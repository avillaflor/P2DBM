from tqdm import tqdm
import h5py
import numpy as np
import os
import hydra


from src.carla.features.scenario_datasets import ScenarioDataset
from src.carla.config.config import DefaultMainConfig
from src.carla.config.scenario_configs import DefaultScenarioConfig
from src.carla.config.observation_configs import EntityObservationConfig
from src.carla.config.action_configs import ThrottleConfig


@hydra.main(version_base=None, config_path='ours/conf/', config_name='train.yaml')
def main(cfg):
    config = DefaultMainConfig()
    config.populate_config(
        observation_config=EntityObservationConfig(max_route_pts=150, max_actors=30, max_token_distance=50.),
        action_config=ThrottleConfig(),
        scenario_config=DefaultScenarioConfig(),
        testing=False,
    )
    dataset_configs = [cfg.train_dataset, cfg.val_dataset]
    for dataset_config in dataset_configs:
        dataset_path = dataset_config.path
        h5_path = os.path.join(dataset_path, 'data.hdf5')
        if not os.path.exists(h5_path):
            dataset = ScenarioDataset(config=config, path=dataset_config['path'], horizon_length=8, skip_every_n=1)

            num_samples = len(dataset)

            f = h5py.File(h5_path, 'w')

            chunk_size = 1
            vehicle_features = f.create_dataset(
                'vehicle_features',
                shape=(num_samples, dataset.seq_length, config.obs_config.max_actors, config.obs_config.actor_dim),
                chunks=(chunk_size, dataset.seq_length, config.obs_config.max_actors, config.obs_config.actor_dim),
                dtype=np.float32)
            vehicle_masks = f.create_dataset(
                'vehicle_masks',
                shape=(num_samples, dataset.seq_length, config.obs_config.max_actors),
                chunks=(chunk_size, dataset.seq_length, config.obs_config.max_actors),
                dtype=bool)
            walker_features = f.create_dataset(
                'walker_features',
                shape=(num_samples, dataset.seq_length, config.obs_config.max_walker_features, config.obs_config.walker_dim),
                chunks=(chunk_size, dataset.seq_length, config.obs_config.max_walker_features, config.obs_config.walker_dim),
                dtype=np.float32)
            walker_masks = f.create_dataset(
                'walker_masks',
                shape=(num_samples, dataset.seq_length, config.obs_config.max_walker_features),
                chunks=(chunk_size, dataset.seq_length, config.obs_config.max_walker_features),
                dtype=bool)
            light_features = f.create_dataset(
                'light_features',
                shape=(num_samples, dataset.seq_length, config.obs_config.max_light_features, config.obs_config.light_dim),
                chunks=(chunk_size, dataset.seq_length, config.obs_config.max_light_features, config.obs_config.light_dim),
                dtype=np.float32)
            light_masks = f.create_dataset(
                'light_masks',
                shape=(num_samples, dataset.seq_length, config.obs_config.max_light_features),
                chunks=(chunk_size, dataset.seq_length, config.obs_config.max_light_features),
                dtype=bool)
            stop_features = f.create_dataset(
                'stop_features',
                shape=(num_samples, dataset.seq_length, config.obs_config.max_signs, config.obs_config.sign_dim),
                chunks=(chunk_size, dataset.seq_length, config.obs_config.max_signs, config.obs_config.sign_dim),
                dtype=np.float32)
            stop_masks = f.create_dataset(
                'stop_masks',
                shape=(num_samples, dataset.seq_length, config.obs_config.max_signs),
                chunks=(chunk_size, dataset.seq_length, config.obs_config.max_signs),
                dtype=bool)
            route_features = f.create_dataset(
                'route_features',
                shape=(num_samples, dataset.seq_length, config.obs_config.max_route_pts, config.obs_config.route_dim),
                chunks=(chunk_size, dataset.seq_length, config.obs_config.max_route_pts, config.obs_config.route_dim),
                dtype=np.float32)
            route_masks = f.create_dataset(
                'route_masks',
                shape=(num_samples, dataset.seq_length, config.obs_config.max_route_pts),
                chunks=(chunk_size, dataset.seq_length, config.obs_config.max_route_pts),
                dtype=bool)
            town = f.create_dataset(
                'town',
                shape=(num_samples,),
                dtype='S10')
            ref = f.create_dataset(
                'ref',
                shape=(num_samples, dataset.seq_length, config.obs_config.ref_dim),
                chunks=(chunk_size, dataset.seq_length, config.obs_config.ref_dim),
                dtype=np.float32)

            for i, sample in enumerate(tqdm(dataset)):
                T = sample['data']['vehicle_features'].shape[0]
                if T < dataset.seq_length:
                    vehicle_features[i] = np.concatenate([sample['data']['vehicle_features'], np.zeros((dataset.seq_length-T, config.obs_config.max_actors, config.obs_config.actor_dim), dtype=np.float32)], axis=0)
                    vehicle_masks[i] = np.concatenate([sample['data']['vehicle_masks'], np.zeros((dataset.seq_length-T, config.obs_config.max_actors), dtype=bool)], axis=0)
                    walker_features[i] = np.concatenate([sample['data']['walker_features'], np.zeros((dataset.seq_length-T, config.obs_config.max_walker_features, config.obs_config.walker_dim), dtype=np.float32)], axis=0)
                    walker_masks[i] = np.concatenate([sample['data']['walker_masks'], np.zeros((dataset.seq_length-T, config.obs_config.max_walker_features), dtype=bool)], axis=0)
                    light_features[i] = np.concatenate([sample['data']['light_features'], np.zeros((dataset.seq_length-T, config.obs_config.max_light_features, config.obs_config.light_dim), dtype=np.float32)], axis=0)
                    light_masks[i] = np.concatenate([sample['data']['light_masks'], np.zeros((dataset.seq_length-T, config.obs_config.max_light_features), dtype=bool)], axis=0)
                    route_features[i] = np.concatenate([sample['data']['route_features'], np.zeros((dataset.seq_length-T, config.obs_config.max_route_pts, config.obs_config.route_dim), dtype=np.float32)], axis=0)
                    route_masks[i] = np.concatenate([sample['data']['route_masks'], np.zeros((dataset.seq_length-T, config.obs_config.max_route_pts), dtype=bool)], axis=0)
                    stop_features[i] = np.concatenate([sample['data']['stop_features'], np.zeros((dataset.seq_length-T, config.obs_config.max_signs, config.obs_config.sign_dim), dtype=np.float32)], axis=0)
                    stop_masks[i] = np.concatenate([sample['data']['stop_masks'], np.zeros((dataset.seq_length-T, config.obs_config.max_signs), dtype=bool)], axis=0)
                    ref[i] = np.concatenate([sample['data']['ref'], np.zeros((dataset.seq_length-T, config.obs_config.ref_dim), dtype=np.float32)], axis=0)
                else:
                    vehicle_features[i] = sample['data']['vehicle_features']
                    vehicle_masks[i] = sample['data']['vehicle_masks']
                    walker_features[i] = sample['data']['walker_features']
                    walker_masks[i] = sample['data']['walker_masks']
                    light_features[i] = sample['data']['light_features']
                    light_masks[i] = sample['data']['light_masks']
                    stop_features[i] = sample['data']['stop_features']
                    stop_masks[i] = sample['data']['stop_masks']
                    route_features[i] = sample['data']['route_features']
                    route_masks[i] = sample['data']['route_masks']
                    ref[i] = sample['data']['ref']

                town[i] = sample['data']['town']


if __name__ == '__main__':
    main()
