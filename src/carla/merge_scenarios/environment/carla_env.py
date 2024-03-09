"""Gym style environment wrapper for CARLA """
import gym
import traceback
import numpy as np


from src.carla.merge_scenarios.environment.carla_interface import CarlaInterface


class CarlaEnv(gym.Env):
    # Wrapper for CARLA environment
    def __init__(self, config, num_agents=1, ap_class='controlled', ap_kwargs=None):
        self.config = config
        self.num_agents = num_agents
        self.carla_interface = CarlaInterface(config, num_agents=num_agents, ap_class=ap_class, ap_kwargs=ap_kwargs)
        self.action_space = self.config.action_config.action_space

    def step(self, actions):
        carla_obses, carla_dones, carla_infos = self.carla_interface.step(actions)
        dones = np.array(carla_dones)
        return carla_obses, dones, carla_infos

    def split_observations(self, carla_obses):
        # helper function for data collection
        obses = []
        for i in range(self.num_agents):
            obs_output = {}
            for k in carla_obses:
                if k == 'town':
                    obs_output[k] = carla_obses[k]
                else:
                    obs_output[k] = carla_obses[k][i]
            obses.append(obs_output)
        obses = np.stack(obses, axis=0)
        return obses

    def reset(self, index=0):
        carla_obses = self.carla_interface.reset(index=index)
        return carla_obses

    def close(self):
        try:
            if self.carla_interface is not None:
                self.carla_interface.close()
        except Exception as e:
            print("********** Exception in closing env **********")
            print(e)
            print(traceback.format_exc())

    def __del__(self):
        self.close()

    def get_autopilot_action(self):
        return self.carla_interface.get_autopilot_actions()
