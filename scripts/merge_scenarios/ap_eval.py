from tqdm import tqdm
import numpy as np
import argparse
import os
import pytorch_lightning as pl
import json


from src.carla.config.config import DefaultMainConfig
from src.carla.config.observation_configs import EntityObservationConfig
from src.carla.config.action_configs import ThrottleConfig
from src.carla.merge_scenarios.environment.carla_env import CarlaEnv
from src.carla.merge_scenarios.config.scenario_configs import OnRampTown03Config, RoundAboutTown03Config, RoundAbout2Town03Config, OnRampTown04Config, OnRamp2Town04Config, OnRampTown06Config, OnRamp2Town06Config, OnRamp3Town06Config, OnRamp4Town06Config, OnRamp5Town06Config
from src.carla.merge_scenarios.agents.auto_pilot_policies import AutopilotNoisePolicy


def main(args):
    #######################
    ######## setup ########
    #######################
    pl.seed_everything(0)


    if args.scenario == 31:
        scenario = OnRampTown03Config
    elif args.scenario == 32:
        scenario = RoundAboutTown03Config
    elif args.scenario == 33:
        scenario = RoundAbout2Town03Config
    elif args.scenario == 41:
        scenario = OnRampTown04Config
    elif args.scenario == 42:
        scenario = OnRamp2Town04Config
    elif args.scenario == 61:
        scenario = OnRampTown06Config
    elif args.scenario == 62:
        scenario = OnRamp2Town06Config
    elif args.scenario == 63:
        scenario = OnRamp3Town06Config
    elif args.scenario == 64:
        scenario = OnRamp4Town06Config
    elif args.scenario == 65:
        scenario = OnRamp5Town06Config
    else:
        raise NotImplementedError

    # env
    config = DefaultMainConfig()
    config.populate_config(
        observation_config=EntityObservationConfig(camera=False),
        action_config=ThrottleConfig(),
        scenario_config=scenario(),
        carla_gpu=args.gpu,
        testing=True,
        render_server=True,
    )
    # should always be 1
    num_agents = 1
    ap_kwargs = {'behavior': args.behavior}
    env = CarlaEnv(config=config, num_agents=num_agents, ap_class=args.ap_class, ap_kwargs=ap_kwargs)

    policy = AutopilotNoisePolicy(env, steer_noise_std=0., speed_noise_std=0., clip=True)

    #######################
    ###### main loop ######
    #######################

    traj_speeds = []
    successes = 0
    fails = 0
    statics = 0
    crashes = 0

    # should never reach timeout because we track statics
    T = 5000
    num_trials = 20
    num_episodes = int(num_trials / num_agents)

    try:
        for ep in tqdm(range(num_episodes)):

            actives = np.ones(num_agents, dtype=bool)
            observation = env.reset(index=ep)

            for t in range(T):

                action = policy(observation)

                ## execute action in environment
                next_observation, terminal, info = env.step(action)

                for i in range(num_agents):
                    if actives[i]:

                        traj_speeds.append(info[i]['speed'])

                        if terminal[i]:
                            status = info[i]['termination_state']
                            success = (status == 'success')
                            successes += int(success)
                            fails += int(not success)

                            statics += int(status == 'static')
                            crashes += int(status == 'obs_collision')

                            actives[i] = False

                if ~actives.any():
                    break

                observation = next_observation

        # print and save statistics
        results = {}
        results['mean_speed'] = float(np.mean(traj_speeds))
        results['success_rate'] = float(successes) / float(num_episodes)
        results['static_rate'] = float(statics) / float(num_episodes)
        results['crash_rate'] = float(crashes) / float(num_episodes)
        print('Scenario', args.scenario)
        print('Mean Speed', results['mean_speed'])
        print('Success Rate', results['success_rate'])
        print('Static Rate', results['static_rate'])
        print('Crash Rate', results['crash_rate'])

        logging_dir = os.path.join('logs', args.exp_name)
        eval_save_dir = os.path.join(logging_dir, args.behavior)
        os.makedirs(eval_save_dir, exist_ok=True)
        eval_file_name = "scenario{0}.json".format(args.scenario)
        eval_file_path = os.path.join(eval_save_dir, eval_file_name)
        with open(eval_file_path, 'w') as f:
            json.dump(results, f)

    finally:
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--behavior', type=str, default='normal')
    parser.add_argument('--ap_class', type=str, default='autopilot')
    parser.add_argument('--exp_name', type=str, default='ap_merge_normal')
    parser.add_argument('--scenario', type=int, default=31)
    args = parser.parse_args()
    main(args)
