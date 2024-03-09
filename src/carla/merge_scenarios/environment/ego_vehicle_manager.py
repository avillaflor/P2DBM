import numpy as np
import carla


from src.carla.features.scene_features import SceneFeaturizer
from src.carla.env_util import get_speed_from_velocity
from src.carla.merge_scenarios.environment import sensors
from src.carla.merge_scenarios.agents.controlled_agent import ControlledAgent
from src.carla.merge_scenarios.agents.autopilot import AutoPilot


class EgoVehicleManager():
    # Manages the ego vehicles
    def __init__(self, config, world, tm, ap_class, ap_kwargs):
        self.config = config
        self.world = world
        self.tm = tm
        self.map = self.world.get_map()
        self.ego_agent = None
        self.ego_vehicle = None
        self._ap_class = ap_class
        if ap_kwargs is None:
            self._ap_kwargs = {}
        else:
            self._ap_kwargs = ap_kwargs
        self._num_wp_lookahead = self.config.obs_config.max_route_pts
        self.traffic_managed = False

        ################################################
        # Blueprints
        ################################################
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        if self.config.scenario_config.disable_two_wheeler:
            self.vehicle_blueprints = [x for x in self.vehicle_blueprints if int(x.get_attribute('number_of_wheels')) == 4]

        self.actor_list = []

    def spawn(self, source_transform, destination_transform):
        # Parameters for ego vehicle
        self.ego_vehicle, self.ego_agent = self._spawn_ego_vehicle(source_transform, destination_transform)
        self._goal_transform = destination_transform
        self._scene_featurizer = SceneFeaturizer(
            self.ego_vehicle,
            self.world,
            town=self.map.name,
            token_radius=self.config.obs_config.max_token_distance,
            route_radius=self.config.obs_config.max_route_distance,
            max_z_distance=self.config.obs_config.max_z_distance,
        )

        self._reset_counters()
        self.sensor_manager = self.spawn_sensors()

    def _reset_counters(self):
        self.static_steps = 0
        self.num_steps = 0

    def _spawn_ego_vehicle(self, source_transform, destination_transform):
        '''
        Spawns and return ego vehicle/Agent
        '''
        # Spawn the actor
        # Create an Agent object with that actor
        # Return the agent instance
        vehicle_bp = self.blueprint_library.find(self.config.scenario_config.vehicle_type)
        vehicle_bp.set_attribute('role_name', 'hero')
        vehicle_bp.set_attribute('color', '255,0,0')

        # Spawning vehicle actor with retry logic as it fails to spawn sometimes
        NUM_RETRIES = 5

        for _ in range(NUM_RETRIES):
            vehicle_actor = self.world.try_spawn_actor(vehicle_bp, source_transform)
            if vehicle_actor is not None:
                break
            else:
                print("Unable to spawn vehicle actor at {0}, {1}.".format(source_transform.location.x, source_transform.location.y))
                print("Number of existing actors, {0}".format(len(self.actor_list)))
                self.destroy_actors()              # Do we need this as ego vehicle is the first one to be spawned?

        if vehicle_actor is not None:
            self.actor_list.append(vehicle_actor)
        else:
            raise Exception("Failed in spawning vehicle actor.")

        if self._ap_class == 'controlled':
            vehicle_agent = ControlledAgent(self.config, self.world, vehicle_actor, source_transform, destination_transform)
        elif self._ap_class == 'autopilot':
            vehicle_agent = AutoPilot(vehicle_actor, source_transform, destination_transform, **self._ap_kwargs)
        else:
            raise NotImplementedError

        return vehicle_actor, vehicle_agent

    @property
    def ego_vehicle_transform(self):
        return self.ego_vehicle.get_transform()

    @property
    def ego_vehicle_velocity(self):
        return self.ego_vehicle.get_velocity()

    def get_control(self, action):
        episode_measurements = {}

        if self.config.action_config.action_type == "throttle":
            steer = np.clip(float(action[0]), -1.0, 1.0)
            gas = np.clip(float(action[1]), -1.0, 1.0)
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0

        else:
            raise Exception("Invalid Action Type")

        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
            gear=0)

        episode_measurements['control_steer'] = control.steer
        episode_measurements['control_throttle'] = control.throttle
        episode_measurements['control_brake'] = control.brake
        episode_measurements['control_reverse'] = control.reverse
        episode_measurements['control_hand_brake'] = control.hand_brake

        return control, episode_measurements

    def get_last_action(self):
        control = self.ego_vehicle.get_control()
        control_measurements = {}
        control_measurements['control_steer'] = control.steer
        control_measurements['control_throttle'] = control.throttle
        control_measurements['control_brake'] = control.brake
        control_measurements['control_reverse'] = control.reverse
        control_measurements['control_hand_brake'] = control.hand_brake
        control_measurements['control_action'] = self.get_action(control_measurements)
        return control_measurements

    def get_action(self, control):
        if self.config.action_config.action_type == "throttle":
            steer = control['control_steer']
            if control['control_throttle'] >= control['control_brake']:
                gas = control['control_throttle']
            else:
                gas = -control['control_brake']
            action = np.array([steer, gas], dtype=np.float32)
        else:
            raise NotImplementedError
        return action

    def step(self, action):
        control, ep_measurements = self.get_control(action)
        if not self.traffic_managed:
            self.ego_vehicle.apply_control(control)
        return ep_measurements

    def obs(self, world_frame):
        all_sensor_readings = self.sensor_manager.get_sensor_readings(world_frame)

        (next_orientation,
            dist_to_trajectory,
            distance_to_goal_trajec,
            self.next_waypoints,
            next_wp_angles,
            next_wp_vectors,
            all_waypoints) = self.ego_agent.global_planner.get_next_orientation(self.ego_vehicle_transform, num_next_waypoints=self._num_wp_lookahead)

        ep_measurements = {
            'dist_to_trajectory': dist_to_trajectory,
            'distance_to_goal': self.ego_vehicle_transform.location.distance(self.ego_agent.destination_transform.location),
            'speed': get_speed_from_velocity(self.ego_vehicle_velocity),
        }
        scene_graph = self._scene_featurizer.get_scene_graph(route_wps=self.next_waypoints, goal_transform=self._goal_transform)

        sensor_readings = {}
        for key in all_sensor_readings:
            if 'sensor.camera' in key:
                sensor_readings[key] = np.copy(all_sensor_readings[key]['image'])
            else:
                sensor_readings.update(all_sensor_readings[key])

        episode_measurements = {
            'scene_graph': scene_graph,
            **ep_measurements,
            **sensor_readings}

        self._update_counters(episode_measurements)

        return episode_measurements

    def _update_counters(self, episode_measurements):
        if episode_measurements["speed"] <= self.config.scenario_config.zero_speed_threshold:
            self.static_steps += 1
        else:
            self.static_steps = 0
        self.num_steps += 1

        episode_measurements['static_steps'] = self.static_steps
        episode_measurements['num_steps'] = self.num_steps

    def get_autopilot_action(self):
        if self._ap_class == 'autopilot':
            control = self.ego_agent.get_control()
            steer = control.steer
            throttle = control.throttle
            brake = control.brake
            if brake > throttle:
                gas = -brake
            else:
                gas = throttle
            return np.array([steer, gas])
        else:
            raise NotImplementedError

    def spawn_sensors(self):
        if self.ego_vehicle is None:
            print("Not spawning sensors as the parent actor is not initialized properly")
            return None
        sensor_manager = sensors.SensorManager(self.config, self.ego_vehicle)
        sensor_manager.spawn()
        for k, v in sensor_manager.sensors.items():
            self.actor_list.append(v.sensor)
        return sensor_manager

    def check_for_vehicle_elimination(self):
        # https://github.com/carla-simulator/carla/issues/3860
        if not self.ego_vehicle.is_alive:
            self.ego_vehicle = None

    def destroy_actors(self):
        for _ in range(len(self.actor_list)):
            try:
                actor = self.actor_list.pop()
                actor.destroy()
            except Exception:
                pass
