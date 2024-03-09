import numpy as np
import torch


from src.carla.features.utils import process_features, transform_points


class Scenario:
    def __init__(self, **kwargs):
        self.data = {}

    def setup(self, samples, config):
        """
        VEHICLE FEATURES
        """
        self.seq_length = len(samples)

        self.max_token_distance = config.obs_config.max_token_distance
        self.max_route_distance = config.obs_config.max_route_distance
        self.max_z_distance = config.obs_config.max_z_distance
        self.max_actors = config.obs_config.max_actors
        self.max_signs = config.obs_config.max_signs
        self.max_walker_features = config.obs_config.max_walker_features
        self.max_light_features = config.obs_config.max_light_features
        self.max_route_pts = config.obs_config.max_route_pts

        self.ref_dim = config.obs_config.ref_dim
        self.actor_dim = config.obs_config.actor_dim
        self.walker_dim = config.obs_config.walker_dim
        self.light_dim = config.obs_config.light_dim
        self.sign_dim = config.obs_config.sign_dim
        self.route_dim = config.obs_config.route_dim

        vehicle_ids = np.unique(np.concatenate([list(sample['scene_graph']['vehicle_features'].keys()) for sample in samples]))
        key_type = type(vehicle_ids[0]) # because depending on whether loaded directly or not, keys are int or str
        ego_id = key_type(samples[0]['scene_graph']['ego_id'])
        vehicle_ids = np.concatenate([[ego_id], vehicle_ids[vehicle_ids != ego_id]])
        num_vehicles = len(vehicle_ids)

        base_ref = np.array([(
            samples[t]['scene_graph']['vehicle_features'][ego_id]['x'],
            samples[t]['scene_graph']['vehicle_features'][ego_id]['y'],
            samples[t]['scene_graph']['vehicle_features'][ego_id]['z'],
            samples[t]['scene_graph']['vehicle_features'][ego_id]['theta'],
        ) for t in range(self.seq_length)])

        """
        Town
        """
        town = samples[0]['scene_graph']['town']

        """
        REFERENCE FRAME
        """
        ref = torch.tensor(base_ref)
        ref = ref.float()

        """
        VEHICLE FEATURES
        """
        vehicle_features = np.zeros((self.seq_length, num_vehicles, self.actor_dim), dtype=np.float32)
        vehicle_zs = np.zeros((self.seq_length, num_vehicles), dtype=np.float32)
        vehicle_masks = np.zeros((self.seq_length, num_vehicles), dtype=bool)

        for t in range(self.seq_length):
            for i, vehicle_id in enumerate(vehicle_ids):
                if vehicle_id in samples[t]['scene_graph']['vehicle_features']:
                    sample = samples[t]['scene_graph']['vehicle_features'][vehicle_id]
                    vehicle_features[t, i] = sample['x'], sample['y'], sample['theta'], sample['speed'], sample['bbox_extent_x'], sample['bbox_extent_y'], sample['speed_limit']
                    vehicle_zs[t, i] = sample['z']
                    vehicle_masks[t, i] = True
                else:
                    vehicle_features[t, i] = np.nan
                    vehicle_zs[t, i] = np.nan
                    vehicle_masks[t, i] = False

        vehicle_features, vehicle_masks = process_features(
            vehicle_features,
            vehicle_masks,
            base_ref,
            zs=vehicle_zs,
            max_token_distance=self.max_token_distance,
            max_z_distance=self.max_z_distance,
            max_actors=self.max_actors,
            backwards_bias=3.)

        """
        STOP FEATURES
        """
        stop_ids = np.unique(np.concatenate([list(sample['scene_graph']['stop_features'].keys()) for sample in samples]))
        num_stops = len(stop_ids)

        stop_features = np.zeros((self.seq_length, num_stops, self.sign_dim), dtype=np.float32)
        stop_zs = np.zeros((self.seq_length, num_stops), dtype=np.float32)
        stop_masks = np.zeros((self.seq_length, num_stops), dtype=bool)

        for t in range(self.seq_length):
            for i, stop_id in enumerate(stop_ids):
                if stop_id in samples[t]['scene_graph']['stop_features']:
                    sample = samples[t]['scene_graph']['stop_features'][stop_id]
                    stop_features[t, i] = sample['x'], sample['y'], sample['theta'], sample['bbox_extent_x'], sample['bbox_extent_y']
                    stop_zs[t, i] = sample['z']
                    stop_masks[t, i] = True
                else:
                    stop_features[t, i] = np.nan
                    stop_zs[t, i] = np.nan
                    stop_masks[t, i] = False

        stop_features, stop_masks = process_features(
            stop_features,
            stop_masks,
            base_ref,
            zs=stop_zs,
            max_token_distance=self.max_token_distance,
            max_z_distance=self.max_z_distance,
            max_actors=self.max_signs)

        """
        WALKER FEATURES
        """
        walker_ids = np.unique(np.concatenate([list(sample['scene_graph']['walker_features'].keys()) for sample in samples]))
        num_walkers = len(walker_ids)

        walker_features = np.zeros((self.seq_length, num_walkers, self.walker_dim), dtype=np.float32)
        walker_zs = np.zeros((self.seq_length, num_walkers), dtype=np.float32)
        walker_masks = np.zeros((self.seq_length, num_walkers), dtype=bool)

        for t in range(self.seq_length):
            for i, walker_id in enumerate(walker_ids):
                if walker_id in samples[t]['scene_graph']['walker_features']:
                    sample = samples[t]['scene_graph']['walker_features'][walker_id]
                    walker_features[t, i] = sample['x'], sample['y'], sample['theta'], sample['speed'], sample['bbox_extent_x'], sample['bbox_extent_y']
                    walker_zs[t, i] = sample['z']
                    walker_masks[t, i] = True
                else:
                    walker_features[t, i] = np.nan
                    walker_zs[t, i] = np.nan
                    walker_masks[t, i] = False

        walker_features, walker_masks = process_features(
            walker_features,
            walker_masks,
            base_ref,
            zs=walker_zs,
            max_token_distance=self.max_token_distance,
            max_z_distance=self.max_z_distance,
            max_actors=self.max_walker_features)

        """
        LIGHT FEATURES
        """
        light_ids = np.unique(np.concatenate([list(sample['scene_graph']['light_features'].keys()) for sample in samples]))
        num_lights = len(light_ids)

        light_features = np.zeros((self.seq_length, num_lights, self.light_dim), dtype=np.float32)
        light_zs = np.zeros((self.seq_length, num_lights), dtype=np.float32)
        light_masks = np.zeros((self.seq_length, num_lights), dtype=bool)

        for t in range(self.seq_length):
            for i, light_id in enumerate(light_ids):
                if light_id in samples[t]['scene_graph']['light_features']:
                    sample = samples[t]['scene_graph']['light_features'][light_id]
                    light_features[t, i] = sample['x'], sample['y'], sample['theta'], sample['bbox_extent_x'], sample['bbox_extent_y'], sample['is_red'], sample['is_yellow'], sample['is_green']
                    light_masks[t, i] = True
                else:
                    light_features[t, i] = np.nan
                    light_zs[t, i] = np.nan
                    light_masks[t, i] = False

        light_features, light_masks = process_features(
            light_features,
            light_masks,
            base_ref,
            zs=light_zs,
            max_token_distance=self.max_token_distance,
            max_z_distance=self.max_z_distance,
            max_actors=self.max_light_features)

        """
        ROUTE FEATURES
        """
        route_ids = np.unique(np.concatenate([list(sample['scene_graph']['route_features'].keys()) for sample in samples]))
        num_routes = len(route_ids)

        route_features = np.zeros((self.seq_length, num_routes, self.route_dim), dtype=np.float32)
        route_zs = np.zeros((self.seq_length, num_routes), dtype=np.float32)
        route_masks = np.zeros((self.seq_length, num_routes), dtype=bool)

        for t in range(self.seq_length):
            for i, route_id in enumerate(route_ids):
                if route_id in samples[t]['scene_graph']['route_features']:
                    sample = samples[t]['scene_graph']['route_features'][route_id]
                    route_features[t, i] = sample['x'], sample['y'], sample['theta'], sample['lane_width']
                    route_zs[t, i] = sample['z']
                    route_masks[t, i] = True
                else:
                    route_features[t, i] = np.nan
                    route_zs[t, i] = np.nan
                    route_masks[t, i] = False

        route_features, route_masks = process_features(
            route_features,
            route_masks,
            base_ref,
            zs=route_zs,
            max_token_distance=self.max_route_distance,
            max_z_distance=self.max_z_distance,
            max_actors=self.max_route_pts)

        self.data = {
            'vehicle_features': vehicle_features,
            'vehicle_masks': vehicle_masks,
            'walker_features': walker_features,
            'walker_masks': walker_masks,
            'light_features': light_features,
            'light_masks': light_masks,
            'stop_features': stop_features,
            'stop_masks': stop_masks,
            'route_features': route_features,
            'route_masks': route_masks,
            'town': town,
            'ref': ref
        }

    def to_dict(self):
        """
        Since using Scenario objects directly in DataLoader is bad, we can export to dict and reconstruct
        """
        scenario_dict = {
            'seq_length': self.seq_length,
            'max_token_distance': self.max_token_distance,
            'max_route_distance': self.max_route_distance,
            'max_z_distance': self.max_z_distance,
            'max_actors': self.max_actors,
            'max_signs': self.max_signs,
            'max_walker_features': self.max_walker_features,
            'max_light_features': self.max_light_features,
            'max_route_pts': self.max_route_pts,
            'ref_dim': self.ref_dim,
            'actor_dim': self.actor_dim,
            'walker_dim': self.walker_dim,
            'light_dim': self.light_dim,
            'sign_dim': self.sign_dim,
            'route_dim': self.route_dim,
            'data': self.data
        }
        return scenario_dict

    @staticmethod
    def load_from_dict(scenario_dict):
        scenario = Scenario()
        scenario.max_token_distance = scenario_dict['max_token_distance']
        scenario.max_route_distance = scenario_dict['max_route_distance']
        scenario.max_z_distance = scenario_dict['max_z_distance']
        scenario.max_actors = scenario_dict['max_actors']
        scenario.max_signs = scenario_dict['max_signs']
        scenario.max_walker_features = scenario_dict['max_walker_features']
        scenario.max_light_features = scenario_dict['max_light_features']
        scenario.max_route_pts = scenario_dict['max_route_pts']
        scenario.ref_dim = scenario_dict['ref_dim']
        scenario.actor_dim = scenario_dict['actor_dim']
        scenario.walker_dim = scenario_dict['walker_dim']
        scenario.light_dim = scenario_dict['light_dim']
        scenario.sign_dim = scenario_dict['sign_dim']
        scenario.route_dim = scenario_dict['route_dim']
        scenario.data = scenario_dict['data']
        return scenario


class BatchScenario:
    def __init__(self, scenarios=None, scenario_dict=None):
        assert (scenarios is not None) != (scenario_dict is not None), 'Must specify either scenarios or scenario_dict when initializing BatchScenario'

        if scenarios is not None:
            assert len(scenarios) > 0, 'Must be at least 1 scenario passed into BatchScenario'
            self.num_scenarios = len(scenarios)
            self.max_token_distance = scenarios[0].max_token_distance
            self.max_route_distance = scenarios[0].max_route_distance
            self.max_z_distance = scenarios[0].max_z_distance
            self.max_actors = scenarios[0].max_actors
            self.max_signs = scenarios[0].max_signs
            self.max_walker_features = scenarios[0].max_walker_features
            self.max_light_features = scenarios[0].max_light_features
            self.max_route_pts = scenarios[0].max_route_pts

            self.ref_dim = scenarios[0].ref_dim
            self.actor_dim = scenarios[0].actor_dim
            self.walker_dim = scenarios[0].walker_dim
            self.light_dim = scenarios[0].light_dim
            self.sign_dim = scenarios[0].sign_dim
            self.route_dim = scenarios[0].route_dim

            self.data = {}
            for key in scenarios[0].data:
                values = [scenario.data[key] for scenario in scenarios]
                if isinstance(scenarios[0].data[key], torch.Tensor):
                    values = torch.stack(values, dim=0)
                self.data[key] = values
        else:
            self.num_scenarios = scenario_dict['num_scenarios']
            self.max_token_distance = scenario_dict['max_token_distance']
            self.max_route_distance = scenario_dict['max_route_distance']
            self.max_z_distance = scenario_dict['max_z_distance']
            self.max_actors = scenario_dict['max_actors']
            self.max_signs = scenario_dict['max_signs']
            self.max_walker_features = scenario_dict['max_walker_features']
            self.max_light_features = scenario_dict['max_light_features']
            self.max_route_pts = scenario_dict['max_route_pts']

            self.ref_dim = scenario_dict['ref_dim']
            self.actor_dim = scenario_dict['actor_dim']
            self.walker_dim = scenario_dict['walker_dim']
            self.light_dim = scenario_dict['light_dim']
            self.sign_dim = scenario_dict['sign_dim']
            self.route_dim = scenario_dict['route_dim']

            self.data = scenario_dict['data']


    def get_obs(self, ref_t, f=1, device=None):
        if device is None:
            device = self.data['vehicle_features'].device

        obs = {
            'town': np.array(self.data['town'], copy=True)
        }

        # get features and normalize
        for k in self.data:
            if k in ['ref', 'stop_features', 'stop_masks', 'route_features', 'route_masks', 'map_features', 'map_masks']:
                obs[k] = self.data[k][:, ::f][:, ref_t].to(device=device, copy=True)
            elif k in ['vehicle_features', 'vehicle_masks', 'walker_features', 'walker_masks', 'light_features', 'light_masks']:
                obs[k] = self.data[k][:, ::f].to(device=device, copy=True)

        B, T = obs['vehicle_features'].shape[:2]

        veh_pos_ori = transform_points(obs['vehicle_features'][..., :3].reshape((B, -1, 3)), obs['ref']).reshape((B, T, -1, 3))
        obs['vehicle_features'] = torch.cat([
            veh_pos_ori[..., :2] / self.max_token_distance,
            veh_pos_ori[..., 2:3],
            obs['vehicle_features'][..., 3:] / self.max_token_distance], dim=-1)

        walk_pos_ori = transform_points(obs['walker_features'][..., :3].reshape((B, -1, 3)), obs['ref']).reshape((B, T, -1, 3))
        obs['walker_features'] = torch.cat([
            walk_pos_ori[..., :2] / self.max_token_distance,
            walk_pos_ori[..., 2:3],
            obs['walker_features'][..., 3:] / self.max_token_distance], dim=-1)

        light_pos_ori = transform_points(obs['light_features'][..., :3].reshape((B, -1, 3)), obs['ref']).reshape((B, T, -1, 3))
        obs['light_features'] = torch.cat([
            light_pos_ori[..., :2] / self.max_token_distance,
            light_pos_ori[..., 2:3],
            obs['light_features'][..., 3:5] / self.max_token_distance,
            obs['light_features'][..., 5:]], dim=-1)

        stop_pos_ori = transform_points(obs['stop_features'][..., :3], obs['ref'])
        obs['stop_features'] = torch.cat([
            stop_pos_ori[..., :2] / self.max_token_distance,
            stop_pos_ori[..., 2:3],
            obs['stop_features'][..., 3:] / self.max_token_distance], dim=-1)

        route_pos_ori = transform_points(obs['route_features'][..., :3], obs['ref'])
        obs['route_features'] = torch.cat([
            route_pos_ori[..., :2] / self.max_token_distance,
            route_pos_ori[..., 2:3],
            obs['route_features'][..., 3:] / self.max_token_distance], dim=-1)

        # mask out features
        for k in obs:
            if 'features' in k:
                mask_k = k[:-8] + 'masks'
                obs[k] = torch.where(
                    obs[mask_k].unsqueeze(-1),
                    obs[k],
                    torch.zeros_like(obs[k]))

        return obs

    def get_traj(self, ref_t, f=1):
        # get trajectories in that vehicle's reference frame
        features = self.data['vehicle_features'][:, ::f, :, :4].clone()
        ref = features[:, ref_t]
        masks = self.data['vehicle_masks'][:, ::f].clone()

        pos_ori = features[..., :3]
        transformed_pos_ori = transform_points(pos_ori.transpose(1, 2), ref[..., :3]).transpose(1, 2)
        transformed_speed = features[..., 3:4] - ref[:, None, :, 3:4]
        transformed_features = torch.cat([
            transformed_pos_ori[..., :2] / self.max_token_distance,
            transformed_pos_ori[..., 2:3],
            transformed_speed / self.max_token_distance], dim=-1)

        masks = masks & masks[:, ref_t:ref_t+1]
        return transformed_features, masks
