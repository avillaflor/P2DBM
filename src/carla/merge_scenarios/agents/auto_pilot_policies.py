import numpy as np


class AutopilotPolicy:
    def __init__(self, env):
        self.env = env

    def __call__(self, obs):
        return self.env.get_autopilot_action()


class AutopilotNoisePolicy:
    def __init__(self, env, steer_noise_std=0., speed_noise_std=0., clip=True):
        self.env = env
        self.steer_noise_std = steer_noise_std
        self.speed_noise_std = speed_noise_std
        self.clip = clip

    def __call__(self, obs):
        res = self.env.get_autopilot_action()
        res[..., 0] += np.random.normal(loc=0.0, scale=self.steer_noise_std, size=1)[0]
        res[..., 1] += np.random.normal(loc=0.0, scale=self.speed_noise_std, size=1)[0]
        if self.clip:
            res = np.clip(res, a_min=-0.999, a_max=0.999)
        return res
