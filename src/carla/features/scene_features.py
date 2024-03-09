import carla
import numpy as np


from src.carla.env_util import get_speed, get_acc


class SceneFeaturizer:
    # Gets all the relevant features from a carla simulation
    def __init__(
            self,
            ego_actor,
            world,
            town,
            token_radius=50.,
            route_radius=170.,
            max_z_distance=7.5,
    ):
        self._ego_actor = ego_actor
        self._world = world
        self._town = town
        self._token_radius = token_radius
        self._route_radius = route_radius
        self._max_z_distance = max_z_distance

    def get_scene_graph(self, route_wps=None, goal_transform=None):
        actors = self._world.get_actors()

        vehicle_features = self._get_vehicle_features(actors.filter('*vehicle*'))
        walker_features = self._get_walker_features(actors.filter('*walker*'))
        light_features = self._get_light_features(actors.filter('*traffic_light*'))
        stop_features = self._get_stop_features(actors.filter('*stop*'))

        scene_graph = {
            'vehicle_features': vehicle_features,
            'walker_features': walker_features,
            'light_features': light_features,
            'stop_features': stop_features,
            'ego_id': self._ego_actor.id,
            'town': self._town,
        }

        if route_wps is not None:
            route_features, goal_features = self._get_route_features(route_wps, goal_transform=goal_transform)

            scene_graph['route_features'] = route_features
            scene_graph['goal_features'] = goal_features

        return scene_graph

    def _get_vehicle_features(self, vehicle_list):
        ego_location = self._ego_actor.get_location()
        vehicle_features = {}

        for vehicle in vehicle_list:
            vehicle_location = vehicle.get_location()
            if (vehicle_location.distance(ego_location) < self._token_radius) and (abs(vehicle_location.z - ego_location.z) < self._max_z_distance):
                veh_vel = vehicle.get_velocity()
                veh_vel = np.array([veh_vel.x, veh_vel.y])
                veh_acc = vehicle.get_acceleration()
                veh_acc = np.array([veh_acc.x, veh_acc.y])
                if np.dot(veh_vel, veh_acc) >= 0:
                    acc_sign = 1.
                else:
                    acc_sign = -1.
                vehicle_features[vehicle.id] = {
                    'x': vehicle_location.x,
                    'y': vehicle_location.y,
                    'z': vehicle_location.z,
                    'theta': np.radians(vehicle.get_transform().rotation.yaw),
                    'speed': get_speed(vehicle, convert=False),
                    'x_vel': veh_vel[0],
                    'y_vel': veh_vel[1],
                    'acc': acc_sign * abs(get_acc(vehicle, convert=False)),
                    'x_acc': veh_acc[0],
                    'y_acc': veh_acc[1],
                    'ang_vel': vehicle.get_angular_velocity().z,
                    'bbox_extent_x': vehicle.bounding_box.extent.x * 2,
                    'bbox_extent_y': vehicle.bounding_box.extent.y * 2,
                    'speed_limit': vehicle.get_speed_limit() / 3.6,
                    'traffic_light': vehicle.get_traffic_light_state() in (carla.libcarla.TrafficLightState.Red,),
                }

        return vehicle_features

    def _get_walker_features(self, walker_list):
        ego_location = self._ego_actor.get_location()
        walker_features = {}

        for walker in walker_list:
            walker_location = walker.get_location()
            if (walker_location.distance(ego_location) < self._token_radius) and (abs(walker_location.z - ego_location.z) < self._max_z_distance):
                walker_vel = walker.get_velocity()
                walker_vel = np.array([walker_vel.x, walker_vel.y])
                walker_acc = walker.get_acceleration()
                walker_acc = np.array([walker_acc.x, walker_acc.y])
                if np.dot(walker_vel, walker_acc) >= 0:
                    acc_sign = 1.
                else:
                    acc_sign = -1.
                walker_features[walker.id] = {
                    'x': walker_location.x,
                    'y': walker_location.y,
                    'z': walker_location.z,
                    'theta': np.radians(walker.get_transform().rotation.yaw),
                    'speed': get_speed(walker, convert=False),
                    'x_vel': walker_vel[0],
                    'y_vel': walker_vel[1],
                    'acc': acc_sign * abs(get_acc(walker, convert=False)),
                    'x_acc': walker_acc[0],
                    'y_acc': walker_acc[1],
                    'ang_vel': walker.get_angular_velocity().z,
                    'bbox_extent_x': walker.bounding_box.extent.x * 2,
                    'bbox_extent_y': walker.bounding_box.extent.y * 2
                }

        return walker_features

    def _get_light_features(self, lights_list):
        ego_location = self._ego_actor.get_location()
        light_features = {}

        for light in lights_list:
            light_location = light.get_transform().transform(light.trigger_volume.location)
            light_location = carla.Location(x=light_location.x, y=light_location.y, z=light_location.z)
            if (light_location.distance(ego_location) < self._token_radius) and (abs(light_location.z - ego_location.z) < self._max_z_distance):
                light_features[light.id] = {
                    'x': light_location.x,
                    'y': light_location.y,
                    'z': light_location.z,
                    'theta': np.radians(light.get_transform().rotation.yaw),
                    'bbox_extent_x': light.trigger_volume.extent.x * 2,
                    'bbox_extent_y': light.trigger_volume.extent.y * 2,
                    'is_red': int(light.state in (carla.libcarla.TrafficLightState.Red,)),
                    'is_yellow': int(light.state in (carla.libcarla.TrafficLightState.Yellow,)),
                    'is_green': int(light.state in (carla.libcarla.TrafficLightState.Green,))
                }

        return light_features

    def _get_stop_features(self, stop_sign_list):
        ego_location = self._ego_actor.get_location()
        stop_features = {}

        for stop_sign in stop_sign_list:
            stop_location = stop_sign.get_transform().transform(stop_sign.trigger_volume.location)
            stop_location = carla.Location(x=stop_location.x, y=stop_location.y, z=stop_location.z)
            if (stop_location.distance(ego_location) < self._token_radius) and (abs(stop_location.z - ego_location.z) < self._max_z_distance):
                stop_features[stop_sign.id] = {
                    'x': stop_location.x,
                    'y': stop_location.y,
                    'z': stop_location.z,
                    'theta': np.radians(stop_sign.get_transform().rotation.yaw),
                    'bbox_extent_x': stop_sign.trigger_volume.extent.x * 2,
                    'bbox_extent_y': stop_sign.trigger_volume.extent.y * 2
                }

        return stop_features

    def _get_route_features(self, route_wps, goal_transform=None):
        ego_location = self._ego_actor.get_location()
        route_features = {}

        for i, wp in enumerate(route_wps):
            route_location = wp.transform.location
            if (route_location.distance(ego_location) < self._route_radius) and (abs(route_location.z - ego_location.z) < self._max_z_distance):
                route_features[i] = {
                    'x': route_location.x,
                    'y': route_location.y,
                    'z': route_location.z,
                    'theta': np.radians(wp.transform.rotation.yaw),
                    'lane_width': wp.lane_width,
                }

        goal_features = {}
        if goal_transform is not None:
            goal_location = goal_transform.location
            goal_features[0] = {
                'x': goal_location.x,
                'y': goal_location.y,
                'z': goal_location.z,
                'theta': np.radians(goal_transform.rotation.yaw),
            }

        return route_features, goal_features
