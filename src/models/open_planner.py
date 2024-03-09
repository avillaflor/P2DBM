import torch
from einops import rearrange, repeat


from src.models.world_model import WorldModel
from src.models.forecasting_model import ForecastingModel
from src.carla.features.utils import transform_points


class OpenPlanner(ForecastingModel):
    # Open-Loop Planner Approach
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
        # gets the best expected action according to the open-loop planner
        obs = {}
        for k in full_obs:
            if k in self._dynamic_feature_names:
                obs[k] = full_obs[k][:, -self.H:]
            else:
                obs[k] = full_obs[k]
        actions = self._open_loop_planner(obs)
        return actions

    def _convert_preds_to_wp_diffs(self, preds):
        # converts open-loop predictions to per time step wp diffs
        rel_wps = preds[0, ..., :4]
        pos_ori_diffs = transform_points(
            rel_wps[:, 1:, :, :3].unsqueeze(-2),
            rel_wps[:, :-1, :, :3],
            invert=False).squeeze(-2)
        speed_diffs = rel_wps[:, 1:, :, 3:4] - rel_wps[:, :-1, :, 3:4]
        wp_diffs = torch.cat([rel_wps[:, :1], torch.cat([pos_ori_diffs, speed_diffs], dim=-1)], dim=1)
        return wp_diffs

    def _pick_modes(self, logits):
        # enumerates modes for ego-vehicle
        # samples modes for other agents
        _, _, A = logits.shape
        shaped_logits = rearrange(logits, '1 m a -> a m')
        samples = torch.multinomial(shaped_logits.softmax(dim=1), self.num_modes * self._num_samples, replacement=True)
        modes = rearrange(samples, 'a (m s) -> m s a', m=self.num_modes, s=self._num_samples)
        modes[:, :, 0] = torch.arange(self.num_modes, device=logits.device).unsqueeze(1).repeat_interleave(self._num_samples, 1)
        return modes

    def _pick_wps(self, wp_diffs, modes, t):
        # gets the next wp according to the currently selected modes
        _, _, N = modes.shape
        B = self.num_modes * self._num_samples

        picked_wps = wp_diffs[modes.reshape((B, N)), t, torch.arange(N, device=modes.device).reshape((1, N))]
        return picked_wps

    def _open_loop_planner(self, agent_obs):
        # runs the open-loop planner
        # enumerates and evaluates the different behavior modes for the ego-vehicle
        # returns action corresponding to the behavior mode with the highest expected reward
        query_obs = {}
        for k in agent_obs:
            if k == 'town':
                query_obs[k] = agent_obs[k]
            else:
                query_obs[k] = agent_obs[k].clone()

        veh_mask_out = query_obs['vehicle_masks'].any(dim=1).any(dim=0)
        query_obs['vehicle_features'] = query_obs['vehicle_features'][:, :, veh_mask_out]
        query_obs['vehicle_masks'] = query_obs['vehicle_masks'][:, :, veh_mask_out]

        preds, logits, _ = self.forward(query_obs)
        wp_diffs = self._convert_preds_to_wp_diffs(preds)
        first_wp_diffs = preds[..., :4]
        modes = self._pick_modes(logits)

        obs = {}
        for k in query_obs:
            obs[k] = repeat(query_obs[k], '1 ... -> (m s) ...', m=self.num_modes, s=self._num_samples)

        obses = [obs]
        rewards = []
        dones = []
        done = None
        is_reds = None
        for t in range(self.plan_T):
            picked_wps = self._pick_wps(wp_diffs, modes, t)
            obs, reward, done, is_reds = self.world_model.step(obs, picked_wps, dones=done, is_reds=is_reds)
            obses.append(obs)
            rewards.append(reward)
            dones.append(done)

        rewards = rearrange(torch.stack(rewards, dim=1), '(m s) t -> m s t', m=self.num_modes, s=self._num_samples)
        dones = rearrange(torch.stack(dones, dim=1), '(m s) t -> m s t', m=self.num_modes, s=self._num_samples)
        rewards[..., 1:] = rewards[..., 1:] * (~dones[..., :-1])

        gammas = self.gamma ** torch.arange(self.plan_T, device=rewards.device)
        values = torch.sum(rewards * gammas, dim=-1)
        values = values.mean(dim=-1)

        ego_labels = values.argmax(dim=-1)

        ego_wp_diffs = rearrange(first_wp_diffs, '1 m t a d -> m t a d')[:, :, 0]
        ego_waypoints = ego_wp_diffs[ego_labels]

        ego_speed = agent_obs['vehicle_features'][0, self.H - 1, 0, 3]
        actions = self.control_pid(ego_waypoints, ego_speed)
        return actions
