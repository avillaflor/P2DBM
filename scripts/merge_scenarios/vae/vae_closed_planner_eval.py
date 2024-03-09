from tqdm import tqdm
import numpy as np
import hydra
import os
import pytorch_lightning as pl
from PIL import Image
import json


from src.carla.env_util import to
from src.carla.config.config import DefaultMainConfig
from src.carla.config.observation_configs import EntityObservationConfig
from src.carla.config.action_configs import ThrottleConfig
from src.carla.merge_scenarios.environment.carla_env import CarlaEnv
from src.carla.merge_scenarios.config.scenario_configs import OnRampTown03Config, RoundAboutTown03Config, RoundAbout2Town03Config, OnRampTown04Config, OnRamp2Town04Config, OnRampTown06Config, OnRamp2Town06Config, OnRamp3Town06Config, OnRamp4Town06Config, OnRamp5Town06Config
from src.models.vae_closed_planner import VAEClosedPlanner


@hydra.main(version_base=None, config_path='conf/', config_name='train.yaml')
def main(cfg):
    #######################
    ######## setup ########
    #######################
    pl.seed_everything(0)

    # agent
    gpu = cfg.gpu[0]
    model_cls = VAEClosedPlanner

    model = model_cls.load_from_checkpoint(cfg.model_checkpoint, **cfg.model, strict=False)

    model.cuda(gpu).eval()

    if cfg.scenario == 31:
        scenario = OnRampTown03Config
    elif cfg.scenario == 32:
        scenario = RoundAboutTown03Config
    elif cfg.scenario == 33:
        scenario = RoundAbout2Town03Config
    elif cfg.scenario == 41:
        scenario = OnRampTown04Config
    elif cfg.scenario == 42:
        scenario = OnRamp2Town04Config
    elif cfg.scenario == 61:
        scenario = OnRampTown06Config
    elif cfg.scenario == 62:
        scenario = OnRamp2Town06Config
    elif cfg.scenario == 63:
        scenario = OnRamp3Town06Config
    elif cfg.scenario == 64:
        scenario = OnRamp4Town06Config
    elif cfg.scenario == 65:
        scenario = OnRamp5Town06Config
    else:
        raise NotImplementedError

    # env
    config = DefaultMainConfig()
    config.populate_config(
        observation_config=EntityObservationConfig(camera=cfg.save_gif, H=cfg.H, f=cfg.f, max_route_pts=150, max_actors=30, max_token_distance=50.),
        action_config=ThrottleConfig(),
        scenario_config=scenario(),
        carla_gpu=gpu,
        testing=True,
        render_server=True,
    )
    env = CarlaEnv(config=config)

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
    num_episodes = 20

    try:
        for ep in tqdm(range(num_episodes)):

            observation = env.reset(index=ep)

            if cfg.save_gif:
                ims = []

            for t in range(T):

                action = model.get_action(to(observation, device=model.device))
                action = action[None]

                # avoids weird carla issue where car struggles to start moving from initial standstill
                if t<20:
                    action[:, 1] = 0.75

                ## execute action in environment
                next_observation, terminal, info = env.step(action)

                if cfg.save_gif:
                    if 'sensor.camera.rgb/top' in info[0]:
                        im = info[0]['sensor.camera.rgb/top']
                        ims.append(Image.fromarray(im))

                traj_speeds.append(info[0]['speed'])

                if terminal[0]:
                    status = info[0]['termination_state']
                    success = (status == 'success')
                    successes += int(success)
                    fails += int(not success)

                    statics += int(status == 'static')
                    crashes += int(status == 'obs_collision')

                    break

                observation = next_observation

            # save gif of route
            if cfg.save_gif:
                if len(ims) > 0:
                    save_dir = os.path.join(cfg.logging_dir, 'vae_closed_planner_viz')
                    os.makedirs(save_dir, exist_ok=True)
                    file_name = "scenario{0}_episode{1:02d}.gif".format(cfg.scenario, ep)
                    file_path = os.path.join(save_dir, file_name)
                    ims[0].save(file_path, save_all=True, append_images=ims[1:], duration=50, loop=0)

        # print and save statistics
        results = {}
        results['mean_speed'] = float(np.mean(traj_speeds))
        results['success_rate'] = float(successes) / float(num_episodes)
        results['static_rate'] = float(statics) / float(num_episodes)
        results['crash_rate'] = float(crashes) / float(num_episodes)
        print('Scenario', cfg.scenario)
        print('Mean Speed', results['mean_speed'])
        print('Success Rate', results['success_rate'])
        print('Static Rate', results['static_rate'])
        print('Crash Rate', results['crash_rate'])

        eval_save_dir = os.path.join(cfg.logging_dir, 'vae_closed_planner_eval')
        os.makedirs(eval_save_dir, exist_ok=True)
        eval_file_name = "scenario{0}.json".format(cfg.scenario)
        eval_file_path = os.path.join(eval_save_dir, eval_file_name)
        with open(eval_file_path, 'w') as f:
            json.dump(results, f)

    finally:
        env.close()

if __name__ == '__main__':
    main()
