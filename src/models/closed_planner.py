import torch


from src.models.world_model import WorldModel
from src.models.forecasting_model import ForecastingModel


class ClosedPlanner(ForecastingModel):
    # Closed-Loop Planner (Ours) Approach
    def __init__(
            self,
            *args,
            gamma=0.95,
            plan_T=8,
            num_samples=8,
            speed_coeff=1.,
            coll_coeff=20.,
            route_coeff=0.1,
            red_speed_coeff=4.,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.plan_T = plan_T
        self._num_samples = num_samples
        self.world_model = WorldModel(
            dt=self.dt,
            max_token_distance=self.max_token_distance,
            speed_coeff=speed_coeff,
            coll_coeff=coll_coeff,
            route_coeff=route_coeff,
            red_speed_coeff=red_speed_coeff,
        )

    @torch.inference_mode()
    def get_action(self, full_obs):
        # gets the best expected action according to the closed-loop planner
        obs = {}
        for k in full_obs:
            if k in self._dynamic_feature_names:
                obs[k] = full_obs[k][:, -self.H:]
            else:
                obs[k] = full_obs[k]
        actions = self._closed_loop_planner(obs)
        return actions

    def _pick_modes(self, logits):
        # enumerates modes for ego-vehicle
        # samples modes for other agents
        B, _, N = logits.shape
        shaped_logits = logits.permute(0, 2, 1).reshape((self.num_modes * self._num_samples * N, self.num_modes))
        modes = torch.multinomial(shaped_logits.softmax(dim=1), 1).reshape((self.num_modes, self._num_samples, N))
        modes[:, :, 0] = torch.arange(self.num_modes, device=logits.device).unsqueeze(1).repeat_interleave(self._num_samples, 1)
        return modes

    def _pick_wps(self, wp_diffs, modes):
        # gets the next wp according to the currently selected modes
        _, _, N = modes.shape
        B = self.num_modes * self._num_samples

        picked_wps = wp_diffs[torch.arange(B, device=modes.device).reshape((B, 1)), modes.reshape((B, N)), 0, torch.arange(N, device=modes.device).reshape((1, N))]
        return picked_wps

    def _closed_loop_planner(self, agent_obs):
        # runs the closed-loop planner
        # enumerates and evaluates the different behavior modes for the ego-vehicle
        # returns action corresponding to the behavior mode with the highest expected reward
        obs = {}
        for k in agent_obs:
            if k == 'town':
                obs[k] = agent_obs[k].repeat(self.num_modes * self._num_samples, 0)
            else:
                obs[k] = agent_obs[k].repeat_interleave(self.num_modes * self._num_samples, 0)

        veh_masks = obs['vehicle_masks'].any(dim=1).any(dim=0)
        obs['vehicle_features'] = obs['vehicle_features'][:, :, veh_masks]
        obs['vehicle_masks'] = obs['vehicle_masks'][:, :, veh_masks]

        obses = [obs]
        rewards = []
        dones = []
        done = None
        is_reds = None
        all_wp_diffs = []
        all_logits = []
        for t in range(self.plan_T):
            pred, logits, _ = self.forward(obs)
            wp_diffs = pred[..., 0:1, :, :4]
            if t == 0:
                first_wp_diffs = pred[..., :4]
                modes = self._pick_modes(logits)
            all_wp_diffs.append(wp_diffs)
            all_logits.append(logits)

            picked_wps = self._pick_wps(wp_diffs, modes)
            obs, reward, done, is_reds = self.world_model.step(obs, picked_wps, dones=done, is_reds=is_reds)
            obses.append(obs)
            rewards.append(reward)
            dones.append(done)

        all_wp_diffs = torch.cat(all_wp_diffs, dim=2)
        rewards = torch.stack(rewards, dim=1).view((self.num_modes, self._num_samples, self.plan_T))
        dones = torch.stack(dones, dim=1).view((self.num_modes, self._num_samples, self.plan_T))
        rewards[..., 1:] = rewards[..., 1:] * (~dones[..., :-1])

        gammas = self.gamma ** torch.arange(self.plan_T, device=rewards.device)
        values = torch.sum(rewards * gammas, dim=-1)
        values = values.mean(dim=-1)

        ego_labels = values.argmax(dim=-1)

        num_agents = obs['vehicle_masks'].shape[-1]
        ego_wp_diffs = first_wp_diffs.reshape((self.num_modes, self._num_samples, self.num_modes, -1, num_agents, 4)).mean(dim=1)[:, :, :, 0]
        next_ego_wps = ego_wp_diffs[ego_labels, ego_labels].unsqueeze(0)

        ego_waypoints = next_ego_wps[0]
        ego_speed = agent_obs['vehicle_features'][0, self.H - 1, 0, 3]
        actions = self.control_pid(ego_waypoints, ego_speed)
        return actions
