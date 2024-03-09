from src.carla.config.base_config import BaseConfig


class BaseObservationConfig(BaseConfig):
    def __init__(self):
        # Key is sensor name, value is configuration parameters
        self.sensors = None

        # Whether or not the lane invasion sensor is enabled
        self.disable_lane_invasion_sensor = None


class EntityObservationConfig(BaseObservationConfig):
    def __init__(self, camera=False, H=1, f=1, max_route_pts=20, max_actors=30, max_token_distance=50., max_route_distance=170.):
        self.sensors = {
            "lane_invasion_sensor": None,
            "collision_sensor": None,
        }

        # camera needed if you are going to save gifs
        if camera:
            camera_sensors = {
                "sensor.camera.rgb/top": {
                    'x': 3.0,
                    'z': 25.0,
                    'pitch': 270.0,
                    'sensor_x_res': '480',
                    'sensor_y_res': '480',
                    'fov': '90',
                    'sensor_tick': '0.0',
                },
            }
            self.sensors.update(camera_sensors)

        self.max_token_distance = max_token_distance
        self.max_route_distance = max_route_distance
        self.max_z_distance = 7.5
        self.max_actors = max_actors
        self.max_walker_features = 10
        self.max_light_features = 10
        self.max_signs = 10
        self.max_route_pts = max_route_pts
        #  self.max_map_pts = 200
        #  self.map_precision = 5.
        #  self.max_speed = 90.
        #  self.max_m_speed = self.max_speed / 3.6

        #  self.T = 1
        self.H = H
        self.f = f
        self.ref_dim = 4
        self.actor_dim = 7
        self.walker_dim = 6
        self.light_dim = 8
        self.sign_dim = 5
        self.route_dim = 4
        self.map_dim = 4
        #  self.action_dim = 2

        # TODO remove unnecessary stuff
        self.disable_lane_invasion_sensor = True
