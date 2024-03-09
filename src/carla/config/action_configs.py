import numpy as np
from gym.spaces import Box


from src.carla.config.base_config import BaseConfig


class BaseActionConfig(BaseConfig):
    def __init__(self):
        # What action type to use
        self.action_type = None
        # Gym Action Space
        self.action_space = None


class ThrottleConfig(BaseActionConfig):
    def __init__(self):
        self.action_type = "throttle"
        self.action_space = Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))
