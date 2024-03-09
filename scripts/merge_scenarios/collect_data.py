import os
import json
import datetime
import argparse
import torch
from tqdm import tqdm
import numpy as np


# Environment
from src.carla.config.config import DefaultMainConfig
from src.carla.config.observation_configs import EntityObservationConfig
from src.carla.config.action_configs import ThrottleConfig
from src.carla.merge_scenarios.environment.carla_env import CarlaEnv
from src.carla.merge_scenarios.config.scenario_configs import OnRampTown03Config, RoundAboutTown03Config, RoundAbout2Town03Config, OnRampTown04Config, OnRamp2Town04Config, OnRampTown06Config, OnRamp2Town06Config, OnRamp3Town06Config, OnRamp4Town06Config, OnRamp5Town06Config
# Policy
from src.carla.merge_scenarios.agents.auto_pilot_policies import AutopilotNoisePolicy


def collect_trajectories(env, save_dir, policy, num_agents, save_rate, max_path_length=500):
    now = datetime.datetime.now()
    salt = np.random.randint(10000)
    save_paths = []
    for i in range(num_agents):
        fname = '_'.join(map(lambda x: '%04d' % x, (now.month, now.day, now.hour, now.minute, now.second, salt)))
        fname = '{0}_{1:03d}'.format(fname, i)
        save_path = os.path.join(save_dir, fname)
        measurements_path = os.path.join(save_path, 'measurements')

        # make directories
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        os.mkdir(measurements_path)
        save_paths.append(save_path)

    obses = env.reset()
    obses = env.split_observations(obses)
    actives = np.ones(num_agents, dtype=bool)

    total_steps = 0
    data_steps = 0
    for step in range(max_path_length):

        actions = policy(obses)

        next_obses, dones, infos = env.step(actions)
        next_obses = env.split_observations(next_obses)

        for i in range(num_agents):
            if actives[i]:
                total_steps += 1
                experience = {
                    'action': to_list(infos[i]['control_action']),
                    'done': bool(dones[i]),
                }
                experience.update(to_list(infos[i]))

                if (step % save_rate) == 0:
                    save_env_state(experience, save_paths[i], step)
                    data_steps += 1
                if dones[i]:
                    actives[i] = False

        if not actives.any():
            break

        obses = next_obses

    return data_steps


def save_env_state(measurements, save_path, idx):
    measurements_path = os.path.join(save_path, 'measurements', '{:04d}.json'.format(idx))
    with open(measurements_path, 'w') as out:
        json.dump(measurements, out)


def to_list(d):
    if isinstance(d, dict):
        res = {}
        for k in d:
            res[k] = to_list(d[k])
        return res
    elif isinstance(d, list):
        return d
    elif isinstance(d, np.ndarray):
        return d.tolist()
    elif isinstance(d, torch.Tensor):
        return d.tolist()
    else:
        return d


def main(args):
    if args.scenario == '3_1':
        scenario = OnRampTown03Config
    elif args.scenario == '3_2':
        scenario = RoundAboutTown03Config
    elif args.scenario == '3_3':
        scenario = RoundAbout2Town03Config
    elif args.scenario == '4_1':
        scenario = OnRampTown04Config
    elif args.scenario == '4_2':
        scenario = OnRamp2Town04Config
    elif args.scenario == '6_1':
        scenario = OnRampTown06Config
    elif args.scenario == '6_2':
        scenario = OnRamp2Town06Config
    elif args.scenario == '6_3':
        scenario = OnRamp3Town06Config
    elif args.scenario == '6_4':
        scenario = OnRamp4Town06Config
    elif args.scenario == '6_5':
        scenario = OnRamp5Town06Config
    else:
        raise NotImplementedError

    scenario_name = str(scenario).split('.')[-1].split('Config')[0]
    save_path = os.path.join(args.path, scenario_name)
    config = DefaultMainConfig()
    config.populate_config(
        observation_config=EntityObservationConfig(max_route_pts=150, max_actors=30, max_token_distance=50.),
        action_config=ThrottleConfig(),
        scenario_config=scenario(),
        testing=False,
        carla_gpu=args.gpu,
        render_server=False,
    )
    config.server_fps = args.fps

    ap_kwargs = {'behavior': args.behavior}

    with CarlaEnv(config=config, num_agents=args.num_agents, ap_class=args.ap_class, ap_kwargs=ap_kwargs) as env:
        # Create the policy
        policy = AutopilotNoisePolicy(env, steer_noise_std=1.e-2, speed_noise_std=1.e-2, clip=True)

        total_samples = 0
        with tqdm(total=args.n_samples) as pbar:
            while total_samples < args.n_samples:
                traj_length = collect_trajectories(env, save_path, policy, args.num_agents, args.save_rate, max_path_length=args.max_path_length)
                total_samples += traj_length
                pbar.update(traj_length)

    print('{0} Done'.format(scenario_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('scenario', type=str)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--behavior', type=str, default='data_collection')
    parser.add_argument('--n_samples', type=int, default=int(2.e5))
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--save_rate', type=int, default=5)
    parser.add_argument('--num_agents', type=int, default=1)
    parser.add_argument('--max_path_length', type=int, default=5000)
    parser.add_argument('--ap_class', type=str, default='autopilot')
    args = parser.parse_args()
    main(args)
