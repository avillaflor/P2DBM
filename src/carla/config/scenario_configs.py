from src.carla.config.base_config import BaseConfig


class BaseScenarioConfig(BaseConfig):
    def __init__(self):
        self.num_npc = None

        # Should two wheelers be allowed
        self.disable_two_wheeler = None

        # City to load
        self.city_name = None

        # Ego vehicle make/model
        self.vehicle_type = None

        # Threshold distance from target transform to consider episode a success
        self.dist_for_success = None

        # Maximum length of episode, episode is terminated if this is exceeded
        self.max_steps = None

        # Maximum number of steps vehicle is allowed to be static
        # If this is exceeded, episode is terminated
        self.max_static_steps = None

        self.disable_collision = None
        self.disable_static = None
        self.disable_traffic_light = None
        self.zero_speed_threshold = None

        # Whether to count a lane invasion as a collision
        self.disable_lane_invasion_collision = None


class DefaultScenarioConfig(BaseScenarioConfig):
    def __init__(self):
        super().__init__()
        self.disable_two_wheeler = True
        self.vehicle_type = 'vehicle.toyota.prius'
        self.dist_for_success = 10.0
        self.max_steps = 10000
        self.max_static_steps = 750
        # Disable episode termination due to vehicle being static
        self.disable_static = False
        self.disable_collision = False
        # Disable episode termination due to traffic light
        self.disable_traffic_light = True
        self.zero_speed_threshold = 1.0
        self.disable_lane_invasion_collision = False
