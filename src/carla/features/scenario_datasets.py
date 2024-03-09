import os
import glob
import json
import torch
from tqdm import tqdm
import h5py


from src.carla.features.scenarios import Scenario, BatchScenario
from src.carla.features.utils import TOWNS


class ScenarioH5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        config,
        towns=TOWNS,
    ):
        self.path = os.path.join(path, 'data.hdf5')
        self.towns = towns

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

        with h5py.File(self.path, 'r') as dataset:
            towns_dataset = dataset['town']
            self.valid_idx = [idx for idx, town in enumerate(tqdm(towns_dataset)) if town.decode() in towns]

    def _get_scenario_dict(self, scenario_data):
        scenario_dict = {
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
            'data': scenario_data
        }
        return scenario_dict


    def __getitem__(self, idx):
        with h5py.File(self.path, 'r') as dataset:
            h5_idx = self.valid_idx[idx]
            scenario_data = {
                'vehicle_features': torch.tensor(dataset['vehicle_features'][h5_idx]),
                'vehicle_masks': torch.tensor(dataset['vehicle_masks'][h5_idx]),
                'walker_features': torch.tensor(dataset['walker_features'][h5_idx]),
                'walker_masks': torch.tensor(dataset['walker_masks'][h5_idx]),
                'light_features': torch.tensor(dataset['light_features'][h5_idx]),
                'light_masks': torch.tensor(dataset['light_masks'][h5_idx]),
                'stop_features': torch.tensor(dataset['stop_features'][h5_idx]),
                'stop_masks': torch.tensor(dataset['stop_masks'][h5_idx]),
                'route_features': torch.tensor(dataset['route_features'][h5_idx]),
                'route_masks': torch.tensor(dataset['route_masks'][h5_idx]),
                'ref': torch.tensor(dataset['ref'][h5_idx]),
                'town': dataset['town'][h5_idx],
            }
        return self._get_scenario_dict(scenario_data)

    def __len__(self):
        return len(self.valid_idx)


class ScenarioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        config,
        horizon_length=1,
        towns=TOWNS,
        skip_every_n=1,
        **scenario_config
    ):
        self.path = path
        self.config = config
        self.seq_length = horizon_length + 1
        self.towns = towns
        self.skip_every_n = skip_every_n
        self.scenario_config = scenario_config

        # Load trajectories
        measurements_trajectory_paths = glob.glob('{}/**/measurements'.format(path), recursive=True)
        trajectory_paths = [os.path.dirname(p) for p in measurements_trajectory_paths]
        assert len(trajectory_paths) > 0, 'No trajectories found in {}'.format(path)

        self.remap_idx_to_sample = []
        self.sample_episodes = []
        self.sample_paths = []

        self.map_pts_dict = {}

        traj_idx = 0
        for traj_path in tqdm(trajectory_paths):
            traj_data = []

            measurement_paths = sorted(glob.glob('{}/measurements/*.json'.format(traj_path)))
            measurement_paths = measurement_paths[::skip_every_n]

            for i, measurement_path in enumerate(measurement_paths):
                with open(measurement_path, 'r') as f:
                    sample = json.load(f)

                if 'scene_graph' in sample:
                    traj_data.append(measurement_path)

            for step_idx in range(len(traj_data)):
                self.remap_idx_to_sample.append((traj_idx, step_idx))

            self.sample_episodes.append(traj_data)
            self.sample_paths.append(traj_path)
            traj_idx += 1

        print('Number of samples: {}'.format(len(self)))

    def __getitem__(self, idx):
        traj_idx, step_idx = self.remap_idx_to_sample[idx]

        measurement_paths = self.sample_episodes[traj_idx][step_idx:step_idx+self.seq_length]
        samples = []
        for measurement_path in measurement_paths:
            with open(measurement_path, 'r') as f:
                sample = json.load(f)
                sample['measurement_path'] = measurement_path

            samples.append(sample)

        scenario = Scenario(**self.scenario_config)
        scenario.setup(samples, self.config)
        scenario_dict = scenario.to_dict()
        return scenario_dict

    def __len__(self):
        return len(self.remap_idx_to_sample)


def scenario_collate_fn(batch):
    scenarios = [Scenario.load_from_dict(scenario_dict) for scenario_dict in batch]
    return BatchScenario(scenarios=scenarios)
