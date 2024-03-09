import torch
from einops import rearrange, repeat
from torch.distributions import Normal


from src.models.world_model import WorldModel
from src.models.vae_forecasting_model import VAEForecastingModel


class VAEClosedPlanner(VAEForecastingModel):
    # Closed-Loop Planner Approach with CVAE model
    def __init__(
            self,
            *args,
            gamma=0.95,
            num_ego_samples=8,
            num_other_samples=8,
            plan_T=8,
            speed_coeff=1.,
            coll_coeff=20.,
            route_coeff=0.1,
            red_speed_coeff=4.,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.num_ego_samples = num_ego_samples
        self.num_other_samples = num_other_samples
        self.plan_T = plan_T
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
        with torch.inference_mode():
            obs = {}
            for k in full_obs:
                if k in self._dynamic_feature_names:
                    obs[k] = full_obs[k][:, -self.H:]
                else:
                    obs[k] = full_obs[k]
            actions = self._closed_loop_planner(obs)
            return actions

    def _get_zs(self, obs):
        # sample latents for the ego-vehicle and other agents
        _, _, A = obs['vehicle_masks'].shape
        device = obs['vehicle_masks'].device
        ego_zs = Normal(
            torch.zeros([self.num_ego_samples, 1, 1, self.vae_dim], device=device),
            torch.ones([self.num_ego_samples, 1, 1, self.vae_dim], device=device),
        ).sample()
        ego_zs = repeat(ego_zs, 'm 1 1 d -> m s 1 d', s=self.num_other_samples)
        other_zs = Normal(
            torch.zeros([1, self.num_other_samples, A-1, self.vae_dim], device=device),
            torch.ones([1, self.num_other_samples, A-1, self.vae_dim], device=device),
        ).sample()
        other_zs = repeat(other_zs, '1 s a d -> m s a d', m=self.num_ego_samples)
        zs = rearrange(torch.cat([ego_zs, other_zs], dim=2), 'm s a d -> (m s) 1 a d')
        return zs

    def _closed_loop_planner(self, agent_obs):
        # runs the closed-loop planner
        # samples and evaluates the different behavior latents for the ego-vehicle
        # returns action corresponding to the sampled behavior latent with the highest expected reward
        obs = {}
        for k in agent_obs:
            obs[k] = repeat(agent_obs[k], 'b ... -> (b s) ...', s=self.num_ego_samples * self.num_other_samples)

        veh_masks = obs['vehicle_masks'].any(dim=1).any(dim=0)
        obs['vehicle_features'] = obs['vehicle_features'][:, :, veh_masks]
        obs['vehicle_masks'] = obs['vehicle_masks'][:, :, veh_masks]

        obses = [obs]
        done = None
        is_reds = None
        rewards = []
        dones = []
        for t in range(self.plan_T):
            if t == 0:
                zs = self._get_zs(obs)
            pred, *_ = self.inference_forward(obs, zs)
            if t == 0:
                first_wp_diffs = pred[..., :4]

            picked_wps = pred[:, 0, 0, :, :4]
            obs, reward, done, is_reds = self.world_model.step(obs, picked_wps, dones=done, is_reds=is_reds)
            obses.append(obs)
            rewards.append(reward)
            dones.append(done)

        rewards = torch.stack(rewards, dim=1)
        dones = torch.stack(dones, dim=1)
        rewards[:, 1:] = rewards[:, 1:] * (~dones[:, :-1])

        gammas = self.gamma ** torch.arange(self.plan_T, device=rewards.device)
        values = torch.sum(rewards * gammas, dim=-1)

        ego_labels = rearrange(values, '(m s) -> m s', m=self.num_ego_samples, s=self.num_other_samples).mean(dim=1).argmax()

        ego_wp_diffs = rearrange(first_wp_diffs, '(m s) 1 t a d -> m s t a d', m=self.num_ego_samples, s=self.num_other_samples)[:, 0, 0]
        ego_waypoints = ego_wp_diffs[ego_labels]

        ego_speed = agent_obs['vehicle_features'][0, self.H - 1, 0, 3]
        actions = self.control_pid(ego_waypoints, ego_speed)

        return actions
