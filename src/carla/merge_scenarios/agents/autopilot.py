# Taken from https://github.com/autonomousvision/plant with modifications for targeting speed limit and data collection in our merge scenarios
import math
import statistics
import numpy as np
import carla
from collections import deque, defaultdict


from src.carla.merge_scenarios.agents.navigation import planner
from carla_agent_files.nav_planner import PIDController, interpolate_trajectory
from carla_agent_files.nav_planner import RoutePlanner_new as RoutePlanner
from leaderboard.autoagents import autonomous_agent_local
from leaderboard.utils.route_manipulation import interpolate_trajectory as route_interpolate_trajectory
from leaderboard.utils.route_manipulation import downsample_route


class AutoPilot(autonomous_agent_local.AutonomousAgent):
    def __init__(
            self,
            vehicle,
            source_transform,
            destination_transform,
            behavior='normal',
    ):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self.world_map = self._vehicle.get_world().get_map()

        self.step = -1
        self.initialized = False
        self.save_path = None

        self.render_bev = False # TODO

        # self.gps_buffer = deque(maxlen=1) # Stores the last x updated gps signals. #TODO

        # Dynamics models
        self.frame_rate = 20
        self.frame_rate_sim = 20
        self.ego_model     = EgoModel(dt=(1.0 / self.frame_rate))
        self.vehicle_model = EgoModel(dt=(1.0 / self.frame_rate))


        # Controllers
        # self.steer_buffer_size = 1     # Number of elements to average steering over
        self.target_speed_slow = 3.0	# Speed at junctions, m/s
        self.target_speed_fast = 4.0	# Speed outside junctions, m/s
        self.clip_delta = 0.25			# Max angular error for turn controller
        self.clip_throttle = 0.75		# Max throttle (0-1)
        self.steer_damping = 0.5		# Steer multiplicative reduction while braking
        self.slope_pitch = 10.0			# Pitch above which throttle is increased
        self.slope_throttle = 0.4		# Excess throttle applied on slopes
        self.angle_search_range = 0    # Number of future waypoints to consider in angle search
        self.steer_noise = 1e-3         # Noise added to expert steering angle
        # self.steer_buffer = deque(maxlen=self.steer_buffer_size)

        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=20)
        self._turn_controller_extrapolation = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=20)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=20)
        self._speed_controller_extrapolation = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=20)

        # Red light detection
        # Coordinates of the center of the red light detector bounding box. In local coordinates of the vehicle, units are meters
        self.center_bb_light_x = -2.0
        self.center_bb_light_y = 0.0
        self.center_bb_light_z = 0.0

        # Extent of the red light detector bounding box. In local coordinates of the vehicle, units are meters. Size are half of the bounding box
        self.extent_bb_light_x = 4.5
        self.extent_bb_light_y = 1.5
        self.extent_bb_light_z = 2.0

        # Obstacle detection
        self.extrapolation_seconds_no_junction = 1.0    # Amount of seconds we look into the future to predict collisions (>= 1 frame)
        # CHANGED for data collection
        if behavior == 'normal' or behavior == 'conservative':
            self.extrapolation_seconds = 3.0                # Amount of seconds we look into the future to predict collisions at junctions
        elif behavior == 'data_collection':
            self.extrapolation_seconds = 3. * np.random.rand() + (1.1 / self.frame_rate)                # Amount of seconds we look into the future to predict collisions at junctions
        else:
            raise NotImplementedError
        self.waypoint_seconds = 4.0                     # Amount of seconds we look into the future to store waypoint labels
        self.detection_radius = 50.0                    # Distance of obstacles (in meters) in which we will check for collisions
        # CHANGED for faster speed
        self.light_radius = 50.0                        # Distance of traffic lights considered relevant (in meters)

        # Speed buffer for detecting "stuck" vehicles
        self.vehicle_speed_buffer = defaultdict( lambda: {"velocity": [], "throttle": [], "brake": []})
        self.stuck_buffer_size = 30
        self.stuck_vel_threshold = 0.1
        self.stuck_throttle_threshold = 0.1
        self.stuck_brake_threshold = 0.1

        # Initialize controls
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.target_speed = 4.0

        # CHANGED for data collection
        self.target_speed_factor = 1.0
        if behavior == 'normal':
            self.bounding_box_factor = 1.
        elif behavior == 'conservative':
            self.bounding_box_factor = 3.
        elif behavior == 'data_collection':
            self.bounding_box_factor = np.random.choice([1., 3.])

        #  print('look ahead', self.extrapolation_seconds, self.extrapolation_seconds_no_junction, 'speed factor', self.target_speed_factor, 'bb factor', self.bounding_box_factor)

        # Shift applied to the augmentation camera at the current frame [0] and
        # the next frame [1]
        self.augmentation_translation = deque(maxlen=2)
        self.augmentation_translation.append(0.0)  # Shift at the first frame is 0.0
        # Rotation is in degrees
        self.augmentation_rotation = deque(maxlen=2)
        self.augmentation_rotation.append(0.0)  # Rotation at the first frame is 0.0

        # Angle to the next waypoint.
        # Normalized in [-1, 1] corresponding to [-90, 90]
        self.angle                = 0.0   # Angle to the next waypoint. Normalized in [-1, 1] corresponding to [-90, 90]
        self.stop_sign_hazard     = False
        self.traffic_light_hazard = False
        self.walker_hazard        = [False for i in range(int(self.extrapolation_seconds * self.frame_rate))]
        self.vehicle_hazard       = [False for i in range(int(self.extrapolation_seconds * self.frame_rate))]
        self.junction             = False
        self.aim_wp = None  # Waypoint that the expert is steering towards
        self.remaining_route = None  # Remaining route
        self.close_traffic_lights = []
        self.close_stop_signs = []
        self.ignore_stop_signs    = True # Whether to ignore stop signs
        self.cleared_stop_signs = []  # A list of all stop signs that we have cleared
        self.future_states = {}

        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam

        # CHANGED for our setting
        self.global_planner = planner.GlobalPlanner()
        self.destination_transform = destination_transform
        dense_waypoints = self.global_planner.trace_route(self.world_map, source_transform, self.destination_transform)
        self.global_planner.set_global_plan(dense_waypoints)

        self.source_transform = source_transform
        self.destination_transform = destination_transform
        gps_route, route = route_interpolate_trajectory(self._world, [self.source_transform.location, self.destination_transform.location])

        ds_ids = downsample_route(route, 50)
        self._global_plan_world_coord = [(route[x][0], route[x][1]) for x in ds_ids]

        # Near node
        trajectory = [item[0].location for item in self._global_plan_world_coord]
        self.dense_route, _ = interpolate_trajectory(self.world_map, trajectory)

        #  print("Dense Waypoints:", len(self.dense_route))

        # CHANGED for faster speed
        self._waypoint_planner = RoutePlanner(5., 50)
        self._waypoint_planner.set_route(self.dense_route, True)
        self._waypoint_planner.save()

        # CHANGED for faster speed
        self._waypoint_planner_extrapolation = RoutePlanner(5., 50)
        self._waypoint_planner_extrapolation.set_route(self.dense_route, True)
        self._waypoint_planner_extrapolation.save()

        self.keep_ids = None

        self.initialized = True

    def sensors(self):
        result = [{
                    'type': 'sensor.opendrive_map',
                    'reading_frequency': 1e-6,
                    'id': 'hd_map'
                    },
                {
                    'type': 'sensor.other.imu',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'imu'
                    },
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'gps'
                    },
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'speed'
                    }
                ]


        return result

        # CHANGED for data collection
    def step(self):
        control = self.get_control()
        self._vehicle.apply_control(control)

    def get_control(self):

        loc = self._vehicle.get_location()
        pos = np.array([loc.x, loc.y])

        self._waypoint_planner.load()
        waypoint_route = self._waypoint_planner.run_step(pos)
        self._waypoint_planner.save()
        self.waypoint_route = np.array([[node[0][0], node[0][1]] for node in waypoint_route])
        _, near_command = waypoint_route[1] if len(waypoint_route) > 1 else waypoint_route[0]  # needs HD map

        self.remaining_route = waypoint_route

        brake = self._get_brake(near_command=near_command) # privileged

        ego_vehicle_waypoint = self.world_map.get_waypoint(self._vehicle.get_location())
        self.junction = ego_vehicle_waypoint.is_junction

        vel = self._vehicle.get_velocity()
        speed = np.sqrt(vel.x ** 2 + vel.y **2 + vel.z **2)

        target_speed = self.target_speed_factor * self._vehicle.get_speed_limit() / 3.6

        # Update saved route
        self._waypoint_planner_extrapolation.load()
        self.waypoint_route_extrapolation = self._waypoint_planner_extrapolation.run_step(pos)
        self._waypoint_planner_extrapolation.save()

        throttle = self._get_throttle(brake, target_speed, speed)

        # hack for steep slopes
        if (self._vehicle.get_transform().rotation.pitch > self.slope_pitch):
            throttle += self.slope_throttle

        theta = self._vehicle.get_transform().rotation.yaw * np.pi / 180.

        steer = self._get_steer(brake, waypoint_route, pos, theta, speed)

        control = carla.VehicleControl()
        control.steer = steer + self.steer_noise * np.random.randn()
        control.throttle = throttle
        # reducing make brake to make behavior a bit more realistic
        control.brake = 0.25 * float(brake)

        self.steer = control.steer
        self.throttle = control.throttle
        self.brake = control.brake
        self.target_speed = target_speed

        # self._save_waypoints()

        return control

    def destroy(self):
        pass

    def _get_steer(self, brake, route, pos, theta, speed, restore=True):
        if len(route) == 1:
            target = route[0][0]
        else:
            target = route[1][0]

        if self._waypoint_planner.is_last:  # end of route
            angle = 0.0
        elif (speed < 0.01) and brake:  # prevent accumulation
            angle = 0.0
        else:
            angle_unnorm = self._get_angle_to(pos, theta, target)
            angle = angle_unnorm / 90

        self.aim_wp = target
        self.angle = angle

        if restore: self._turn_controller.load()
        steer = self._turn_controller.step(angle)
        if restore: self._turn_controller.save()

        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        if brake:
            steer *= self.steer_damping

        return steer

    def _get_steer_extrapolation(self, route, pos, theta, restore=True):
        if self._waypoint_planner_extrapolation.is_last: # end of route
            angle = 0.0
        else:
            if len(route) == 1:
                target = route[0][0]
            else:
                target = route[1][0]
            angle_unnorm = self._get_angle_to(pos, theta, target)
            angle = angle_unnorm / 90

        self.angle = angle

        if restore:
            self._turn_controller_extrapolation.load()
        steer = self._turn_controller_extrapolation.step(angle)
        if restore:
            self._turn_controller_extrapolation.save()

        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        return steer

    def _get_throttle(self, brake, target_speed, speed, restore=True):
        target_speed = target_speed if not brake else 0.0

        if self._waypoint_planner.is_last: # end of route
            target_speed = 0.0

        delta = np.clip(target_speed - speed, 0.0, self.clip_delta)

        if restore: self._speed_controller.load()
        throttle = self._speed_controller.step(delta)
        if restore: self._speed_controller.save()

        throttle = np.clip(throttle, 0.0, self.clip_throttle)

        if brake:
            throttle = 0.0

        return throttle

    def _get_throttle_extrapolation(self, target_speed, speed, restore=True):
        if self._waypoint_planner_extrapolation.is_last: # end of route
            target_speed = 0.0

        delta = np.clip(target_speed - speed, 0.0, self.clip_delta)

        if restore: self._speed_controller_extrapolation.load()
        throttle = self._speed_controller_extrapolation.step(delta)
        if restore: self._speed_controller_extrapolation.save()

        throttle = np.clip(throttle, 0.0, self.clip_throttle)

        return throttle

    def _get_brake(self, vehicle_hazard=None, light_hazard=None, walker_hazard=None, stop_sign_hazard=None, near_command=None):
        actors = self._world.get_actors()
        speed = self._get_forward_speed()

        vehicle_location = self._vehicle.get_location()
        vehicle_transform = self._vehicle.get_transform()

        all_vehicles = actors.filter('*vehicle*')

        # print(f'expert: {self.keep_ids}')
        vehicles = []
        if self.keep_ids is not None:
            for vehicle in all_vehicles:
                # print(vehicle.id)
                if vehicle.id in self.keep_ids:
                    vehicles.append(vehicle)
        else:
            vehicles = all_vehicles

        # print(f'{len(vehicles)} vehicles')

        # -----------------------------------------------------------
        # Red light detection
        # -----------------------------------------------------------
        if light_hazard is None:
            light_hazard = False
            self._active_traffic_light = None
            _traffic_lights = self.get_nearby_object(vehicle_location, actors.filter('*traffic_light*'), self.light_radius)

            center_light_detector_bb = vehicle_transform.transform(carla.Location(x=self.center_bb_light_x, y=self.center_bb_light_y, z=self.center_bb_light_z))
            extent_light_detector_bb = carla.Vector3D(x=self.extent_bb_light_x, y=self.extent_bb_light_y, z=self.extent_bb_light_z)
            light_detector_bb = carla.BoundingBox(center_light_detector_bb, extent_light_detector_bb)
            light_detector_bb.rotation = vehicle_transform.rotation
            for light in _traffic_lights:

                # box in which we will look for traffic light triggers.
                center_bounding_box = light.get_transform().transform(light.trigger_volume.location)
                center_bounding_box = carla.Location(center_bounding_box.x, center_bounding_box.y, center_bounding_box.z)
                length_bounding_box = carla.Vector3D(light.trigger_volume.extent.x, light.trigger_volume.extent.y, light.trigger_volume.extent.z)
                transform = carla.Transform(center_bounding_box) # can only create a bounding box from a transform.location, not from a location
                bounding_box = carla.BoundingBox(transform.location, length_bounding_box)

                gloabl_rot = light.get_transform().rotation
                bounding_box.rotation = carla.Rotation(pitch = light.trigger_volume.rotation.pitch + gloabl_rot.pitch,
                                                    yaw   = light.trigger_volume.rotation.yaw   + gloabl_rot.yaw,
                                                    roll  = light.trigger_volume.rotation.roll  + gloabl_rot.roll)

                if(self.check_obb_intersection(light_detector_bb, bounding_box) == True):
                    if ((light.state == carla.libcarla.TrafficLightState.Red)
                        or (light.state == carla.libcarla.TrafficLightState.Yellow)):
                        self._active_traffic_light = light
                        light_hazard = True

        #-----------------------------------------------------------
        # Stop sign detection
        #-----------------------------------------------------------
        if stop_sign_hazard is None:
            stop_sign_hazard = False
            if not self.ignore_stop_signs:
                stop_signs     = self.get_nearby_object(vehicle_location, actors.filter('*stop*'), self.light_radius)
                center_vehicle_stop_sign_detector_bb   = vehicle_transform.transform(self._vehicle.bounding_box.location)
                extent_vehicle_stop_sign_detector_bb   = self._vehicle.bounding_box.extent
                vehicle_stop_sign_detector_bb          = carla.BoundingBox(center_vehicle_stop_sign_detector_bb, extent_vehicle_stop_sign_detector_bb)
                vehicle_stop_sign_detector_bb.rotation = vehicle_transform.rotation

                for stop_sign in stop_signs:
                    center_bb_stop_sign    = stop_sign.get_transform().transform(stop_sign.trigger_volume.location)
                    length_bb_stop_sign    = carla.Vector3D(stop_sign.trigger_volume.extent.x, stop_sign.trigger_volume.extent.y, stop_sign.trigger_volume.extent.z)
                    transform_stop_sign    = carla.Transform(center_bb_stop_sign)
                    bounding_box_stop_sign = carla.BoundingBox(transform_stop_sign.location, length_bb_stop_sign)
                    rotation_stop_sign     = stop_sign.get_transform().rotation
                    bounding_box_stop_sign.rotation = carla.Rotation(pitch=stop_sign.trigger_volume.rotation.pitch + rotation_stop_sign.pitch,
                                                                    yaw  =stop_sign.trigger_volume.rotation.yaw   + rotation_stop_sign.yaw,
                                                                    roll =stop_sign.trigger_volume.rotation.roll  + rotation_stop_sign.roll)


                    if (self.check_obb_intersection(vehicle_stop_sign_detector_bb, bounding_box_stop_sign) == True):
                        if(not (stop_sign.id in self.cleared_stop_signs)):
                            if((speed * 3.6) > 0.0): #Conversion from m/s to km/h
                                stop_sign_hazard = True
                            else:
                                self.cleared_stop_signs.append(stop_sign.id)

                # reset past cleared stop signs
                for cleared_stop_sign in self.cleared_stop_signs:
                    remove_stop_sign = True
                    for stop_sign in stop_signs:
                        if(stop_sign.id == cleared_stop_sign):
                            remove_stop_sign = False # stop sign is still around us hence it might be active
                    if(remove_stop_sign == True):
                        self.cleared_stop_signs.remove(cleared_stop_sign)

        # -----------------------------------------------------------
        # Obstacle detection
        # -----------------------------------------------------------
        if vehicle_hazard is None or walker_hazard is None:
            vehicle_hazard = False
            lane_change = ((near_command.value == 5) or (near_command.value == 6))
            self.vehicle_hazard = [False for i in range(int(self.extrapolation_seconds * self.frame_rate))]
            extrapolation_seconds   = self.extrapolation_seconds  # amount of seconds we look into the future to predict collisions
            detection_radius        = self.detection_radius       # distance in which we check for collisions
            number_of_future_frames = int(extrapolation_seconds * self.frame_rate)
            number_of_future_frames_no_junction = int(self.extrapolation_seconds_no_junction * self.frame_rate)

            # -----------------------------------------------------------
            # Walker detection
            # -----------------------------------------------------------
            walkers = actors.filter('*walker*')
            walker_hazard  = False
            self.walker_hazard = [False for i in range(int(self.extrapolation_seconds * self.frame_rate))]
            nearby_walkers = []
            for walker in walkers:
                if (walker.get_location().distance(vehicle_location) < detection_radius):
                    walker_future_bbs = []
                    walker_transform = walker.get_transform()
                    walker_velocity = walker.get_velocity()
                    walker_speed = self._get_forward_speed(transform=walker_transform, velocity=walker_velocity)  # In m/s
                    walker_location = walker_transform.location
                    walker_direction = walker.get_control().direction

                    for i in range(number_of_future_frames):
                        if self.render_bev==False and self.junction==False and i > number_of_future_frames_no_junction:
                            break

                        # NOTE for perf. optimization: Could also just add velocity.x instead might be a bit faster
                        new_x = walker_location.x + walker_direction.x * walker_speed * (1.0 / self.frame_rate)
                        new_y = walker_location.y + walker_direction.y * walker_speed * (1.0 / self.frame_rate)
                        new_z = walker_location.z + walker_direction.z * walker_speed * (1.0 / self.frame_rate)
                        walker_location = carla.Location(new_x, new_y, new_z)

                        transform = carla.Transform(walker_location)
                        bounding_box = carla.BoundingBox(transform.location, walker.bounding_box.extent)
                        bounding_box.rotation = carla.Rotation(pitch = walker.bounding_box.rotation.pitch + walker_transform.rotation.pitch,
                                                            yaw   = walker.bounding_box.rotation.yaw   + walker_transform.rotation.yaw,
                                                            roll  = walker.bounding_box.rotation.roll  + walker_transform.rotation.roll)

                        walker_future_bbs.append(bounding_box)
                    nearby_walkers.append(walker_future_bbs)

            # -----------------------------------------------------------
            # Vehicle detection
            # -----------------------------------------------------------
            nearby_vehicles = {}
            tmp_near_vehicle_id = []
            tmp_stucked_vehicle_id = []
            for vehicle in vehicles:
                if (vehicle.id == self._vehicle.id):
                    continue
                if (vehicle.get_location().distance(vehicle_location) < detection_radius):
                    tmp_near_vehicle_id.append(vehicle.id)
                    veh_future_bbs    = []
                    traffic_transform = vehicle.get_transform()
                    traffic_control   = vehicle.get_control()
                    traffic_velocity  = vehicle.get_velocity()
                    traffic_speed     = self._get_forward_speed(transform=traffic_transform, velocity=traffic_velocity) # In m/s

                    self.vehicle_speed_buffer[vehicle.id]["velocity"].append(traffic_speed)
                    self.vehicle_speed_buffer[vehicle.id]["throttle"].append(traffic_control.throttle)
                    self.vehicle_speed_buffer[vehicle.id]["brake"].append(traffic_control.brake)
                    if len(self.vehicle_speed_buffer[vehicle.id]["velocity"]) > self.stuck_buffer_size:
                        self.vehicle_speed_buffer[vehicle.id]["velocity"] = self.vehicle_speed_buffer[vehicle.id]["velocity"][-self.stuck_buffer_size:]
                        self.vehicle_speed_buffer[vehicle.id]["throttle"] = self.vehicle_speed_buffer[vehicle.id]["throttle"][-self.stuck_buffer_size:]
                        self.vehicle_speed_buffer[vehicle.id]["brake"] = self.vehicle_speed_buffer[vehicle.id]["brake"][-self.stuck_buffer_size:]


                    next_loc   = np.array([traffic_transform.location.x, traffic_transform.location.y])
                    action     = np.array(np.stack([traffic_control.steer, traffic_control.throttle, traffic_control.brake], axis=-1))
                    next_yaw   = np.array([traffic_transform.rotation.yaw / 180.0 * np.pi])
                    next_speed = np.array([traffic_speed])

                    for i in range(number_of_future_frames):
                        if self.render_bev==False and self.junction==False and i > number_of_future_frames_no_junction:
                            break

                        next_loc, next_yaw, next_speed = self.vehicle_model.forward(next_loc, next_yaw, next_speed, action)

                        delta_yaws = next_yaw.item() * 180.0 / np.pi

                        transform             = carla.Transform(carla.Location(x=next_loc[0].item(), y=next_loc[1].item(), z=traffic_transform.location.z))
                        bounding_box          = carla.BoundingBox(transform.location, vehicle.bounding_box.extent)
                        bounding_box.rotation = carla.Rotation(pitch=float(traffic_transform.rotation.pitch),
                                                            yaw=float(delta_yaws),
                                                            roll=float(traffic_transform.rotation.roll))

                        veh_future_bbs.append(bounding_box)

                    if (statistics.mean(self.vehicle_speed_buffer[vehicle.id]["velocity"]) < self.stuck_vel_threshold
                            and statistics.mean(self.vehicle_speed_buffer[vehicle.id]["throttle"]) > self.stuck_throttle_threshold
                            and statistics.mean(self.vehicle_speed_buffer[vehicle.id]["brake"]) < self.stuck_brake_threshold):
                        tmp_stucked_vehicle_id.append(vehicle.id)

                    nearby_vehicles[vehicle.id] = veh_future_bbs

            # delete old vehicles
            to_delete = set(self.vehicle_speed_buffer.keys()).difference(tmp_near_vehicle_id)
            for d in to_delete:
                del self.vehicle_speed_buffer[d]

            # -----------------------------------------------------------
            # Intersection checks with ego vehicle
            # -----------------------------------------------------------

            next_loc_no_brake   = np.array([vehicle_transform.location.x, vehicle_transform.location.y])
            next_yaw_no_brake   = np.array([vehicle_transform.rotation.yaw / 180.0 * np.pi])
            next_speed_no_brake = np.array([speed])

            #NOTE intentionally set ego vehicle to move at the target speed (we want to know if there is an intersection if we would not brake)
            throttle_extrapolation = self._get_throttle_extrapolation(self.target_speed, speed)
            action_no_brake     = np.array(np.stack([self.steer, throttle_extrapolation, 0.0], axis=-1))

            back_only_vehicle_id = []

            for i in range(number_of_future_frames):
                if self.render_bev==False and self.junction==False and i > number_of_future_frames_no_junction:
                    alpha = 255
                    color_value = 50
                    break
                else:
                    alpha = 50
                    color_value = 255

                # calculate ego vehicle bounding box for the next timestep. We don't consider timestep 0 because it is from the past and has already happened.
                next_loc_no_brake, next_yaw_no_brake, next_speed_no_brake = self.ego_model.forward(next_loc_no_brake, next_yaw_no_brake, next_speed_no_brake, action_no_brake)
                waypoint_route_extrapolation_temp = self._waypoint_planner_extrapolation.run_step(next_loc_no_brake)
                steer_extrapolation_temp = self._get_steer_extrapolation(waypoint_route_extrapolation_temp, next_loc_no_brake, next_yaw_no_brake.item(), restore=False)
                throttle_extrapolation_temp = self._get_throttle_extrapolation(self.target_speed, next_speed_no_brake, restore=False)
                brake_extrapolation_temp = 1.0 if self._waypoint_planner_extrapolation.is_last else 0.0
                action_no_brake = np.array(np.stack([steer_extrapolation_temp, float(throttle_extrapolation_temp), brake_extrapolation_temp], axis=-1))

                delta_yaws_no_brake = next_yaw_no_brake.item() * 180.0 / np.pi

                # Outside of lane changes we do not want to consider collisions from the rear as they are likely false positives.
                if (lane_change == False):
                    cosine = np.cos(next_yaw_no_brake.item())
                    sine = np.sin(next_yaw_no_brake.item())

                    extent           = self._vehicle.bounding_box.extent
                    extent.x         = extent.x / 2. * self.bounding_box_factor
                    extent.y = extent.y * self.bounding_box_factor

                    # front half
                    transform             = carla.Transform(carla.Location(x=next_loc_no_brake[0].item()+extent.x*cosine, y=next_loc_no_brake[1].item()+extent.y*sine, z=vehicle_transform.location.z))
                    bounding_box          = carla.BoundingBox(transform.location, extent)
                    bounding_box.rotation = carla.Rotation(pitch=float(vehicle_transform.rotation.pitch), yaw=float(delta_yaws_no_brake), roll=float(vehicle_transform.rotation.roll))

                    # back half
                    transform_back             = carla.Transform(carla.Location(x=next_loc_no_brake[0].item()-extent.x*cosine, y=next_loc_no_brake[1].item()-extent.y*sine, z=vehicle_transform.location.z))
                    bounding_box_back          = carla.BoundingBox(transform_back.location, extent)
                    bounding_box_back.rotation = carla.Rotation(pitch=float(vehicle_transform.rotation.pitch), yaw=float(delta_yaws_no_brake), roll=float(vehicle_transform.rotation.roll))

                    i_stuck = i
                    for id, traffic_participant in nearby_vehicles.items():
                        if self.render_bev==False and self.junction==False and i > number_of_future_frames_no_junction:
                                break
                        if id in tmp_stucked_vehicle_id:
                            i_stuck = 0
                        back_intersect = (self.check_obb_intersection(bounding_box_back, traffic_participant[i_stuck]) == True)
                        front_intersect = (self.check_obb_intersection(bounding_box, traffic_participant[i_stuck]) == True)
                        if id in back_only_vehicle_id:
                            back_only_vehicle_id.remove(id)
                            if back_intersect:
                                back_only_vehicle_id.append(id)
                            continue
                        if back_intersect and not front_intersect:
                            back_only_vehicle_id.append(id)
                        if front_intersect:
                            if self.junction==True or i <= number_of_future_frames_no_junction:
                                vehicle_hazard = True
                            self.vehicle_hazard[i] = True

                # During lane changes we consider the entire bounding box of the car to avoid driving when there is a vehicle coming from behind.
                else:
                    transform             = carla.Transform(carla.Location(x=next_loc_no_brake[0].item(), y=next_loc_no_brake[1].item(), z=vehicle_transform.location.z))
                    bounding_box          = carla.BoundingBox(transform.location, self._vehicle.bounding_box.extent)
                    bounding_box.rotation = carla.Rotation(pitch=float(vehicle_transform.rotation.pitch), yaw=float(delta_yaws_no_brake), roll=float(vehicle_transform.rotation.roll))

                    i_stuck = i
                    for id, traffic_participant in nearby_vehicles.items():
                        if self.render_bev == False and self.junction == False and i > number_of_future_frames_no_junction:
                            break
                        if id in tmp_stucked_vehicle_id:
                            i_stuck = 0 # If the vehicle is stuck we treat him that his current position will stay the same in the future
                        if (self.check_obb_intersection(bounding_box, traffic_participant[i_stuck]) == True):
                            if self.junction == True or i <= number_of_future_frames_no_junction:
                                vehicle_hazard = True
                            self.vehicle_hazard[i] = True

                for walker in nearby_walkers:
                    if self.render_bev==False and self.junction==False and i > number_of_future_frames_no_junction:
                            break
                    if (self.check_obb_intersection(bounding_box, walker[i]) == True):
                        if self.junction==True or i <= number_of_future_frames_no_junction:
                            walker_hazard = True
                        self.walker_hazard[i] = True



            # add safety bounding box in front. If there is anything in there we won't start driving
            bremsweg = ((speed * 3.6) / 10.0)**2 / 2.0 # Bremsweg formula for emergency break
            safety_x = np.clip(bremsweg + 1.0, a_min=2.0, a_max=4.0) # plus one meter is the car.
            
            center_safety_box     = vehicle_transform.transform(carla.Location(x=safety_x, y=0.0, z=0.0))
            bounding_box          = carla.BoundingBox(center_safety_box, self._vehicle.bounding_box.extent)
            bounding_box.rotation = vehicle_transform.rotation
            
            for _, traffic_participant in nearby_vehicles.items():
                if (self.check_obb_intersection(bounding_box, traffic_participant[0]) == True): # check the first BB of the traffic participant. We don't extrapolate into the future here.
                    vehicle_hazard = True
                    self.vehicle_hazard[0] = True

            for walker in nearby_walkers:
                if (self.check_obb_intersection(bounding_box, walker[0]) == True): # check the first BB of the traffic participant. We don't extrapolate into the future here.
                    walker_hazard = True
                    self.walker_hazard[0] = True

            self.future_states = {'walker': nearby_walkers, 'vehicle': nearby_vehicles}

        else:
            self.vehicle_hazard = vehicle_hazard
            self.walker_hazard = walker_hazard

        self.stop_sign_hazard     = stop_sign_hazard
        self.traffic_light_hazard = light_hazard
        
        return (vehicle_hazard or light_hazard or walker_hazard or stop_sign_hazard)

    def _intersection_check(self, ego_wps):
        actors = self._world.get_actors()
        
        vehicle_location = self._vehicle.get_location()
        vehicle_transform = self._vehicle.get_transform()

        all_vehicles = actors.filter('*vehicle*')
        
        vehicles = []
        if self.keep_ids is not None:
            for vehicle in all_vehicles:
                if vehicle.id in self.keep_ids:
                    vehicles.append(vehicle)
        else:
            vehicles = all_vehicles

       
        # -----------------------------------------------------------
        # Obstacle detection
        # -----------------------------------------------------------
        vehicle_hazard = False
        self.vehicle_hazard = [False for i in range(int(self.extrapolation_seconds * self.frame_rate))]
        extrapolation_seconds   = self.extrapolation_seconds  # amount of seconds we look into the future to predict collisions
        detection_radius        = self.detection_radius       # distance in which we check for collisions
        number_of_future_frames = int(extrapolation_seconds * self.frame_rate)
        number_of_future_frames_no_junction = int(self.extrapolation_seconds_no_junction * self.frame_rate)

        # -----------------------------------------------------------
        # Walker detection
        # -----------------------------------------------------------
        walkers = actors.filter('*walker*')
        walker_hazard  = False
        self.walker_hazard = [False for i in range(int(self.extrapolation_seconds * self.frame_rate))]
        nearby_walkers = []
        for walker in walkers:
            if (walker.get_location().distance(vehicle_location) < detection_radius):
                walker_future_bbs = []
                walker_transform = walker.get_transform()
                walker_velocity = walker.get_velocity()
                walker_speed = self._get_forward_speed(transform=walker_transform, velocity=walker_velocity)  # In m/s
                walker_location = walker_transform.location
                walker_direction = walker.get_control().direction

                for i in range(number_of_future_frames):
                    if self.render_bev==False and self.junction==False and i > number_of_future_frames_no_junction:
                        break

                    # NOTE for perf. optimization: Could also just add velocity.x instead might be a bit faster
                    new_x = walker_location.x + walker_direction.x * walker_speed * (1.0 / self.frame_rate)
                    new_y = walker_location.y + walker_direction.y * walker_speed * (1.0 / self.frame_rate)
                    new_z = walker_location.z + walker_direction.z * walker_speed * (1.0 / self.frame_rate)
                    walker_location = carla.Location(new_x, new_y, new_z)

                    transform = carla.Transform(walker_location)
                    bounding_box = carla.BoundingBox(transform.location, walker.bounding_box.extent)
                    bounding_box.rotation = carla.Rotation(pitch = walker.bounding_box.rotation.pitch + walker_transform.rotation.pitch,
                                                        yaw   = walker.bounding_box.rotation.yaw   + walker_transform.rotation.yaw,
                                                        roll  = walker.bounding_box.rotation.roll  + walker_transform.rotation.roll)

                    walker_future_bbs.append(bounding_box)
                nearby_walkers.append(walker_future_bbs)

        # -----------------------------------------------------------
        # Vehicle detection
        # -----------------------------------------------------------
        nearby_vehicles = {}
        tmp_near_vehicle_id = []
        tmp_stucked_vehicle_id = []
        for vehicle in vehicles:
            if (vehicle.id == self._vehicle.id):
                continue
            if (vehicle.get_location().distance(vehicle_location) < detection_radius):
                tmp_near_vehicle_id.append(vehicle.id)
                veh_future_bbs    = []
                traffic_transform = vehicle.get_transform()
                traffic_control   = vehicle.get_control()
                traffic_velocity  = vehicle.get_velocity()
                traffic_speed     = self._get_forward_speed(transform=traffic_transform, velocity=traffic_velocity) # In m/s

                self.vehicle_speed_buffer[vehicle.id]["velocity"].append(traffic_speed)
                self.vehicle_speed_buffer[vehicle.id]["throttle"].append(traffic_control.throttle)
                self.vehicle_speed_buffer[vehicle.id]["brake"].append(traffic_control.brake)
                if len(self.vehicle_speed_buffer[vehicle.id]["velocity"]) > self.stuck_buffer_size:
                    self.vehicle_speed_buffer[vehicle.id]["velocity"] = self.vehicle_speed_buffer[vehicle.id]["velocity"][-self.stuck_buffer_size:]
                    self.vehicle_speed_buffer[vehicle.id]["throttle"] = self.vehicle_speed_buffer[vehicle.id]["throttle"][-self.stuck_buffer_size:]
                    self.vehicle_speed_buffer[vehicle.id]["brake"] = self.vehicle_speed_buffer[vehicle.id]["brake"][-self.stuck_buffer_size:]


                next_loc   = np.array([traffic_transform.location.x, traffic_transform.location.y])
                action     = np.array(np.stack([traffic_control.steer, traffic_control.throttle, traffic_control.brake], axis=-1))
                next_yaw   = np.array([traffic_transform.rotation.yaw / 180.0 * np.pi])
                next_speed = np.array([traffic_speed])
                
                for i in range(number_of_future_frames):
                    if self.render_bev==False and self.junction==False and i > number_of_future_frames_no_junction:
                        break

                    next_loc, next_yaw, next_speed = self.vehicle_model.forward(next_loc, next_yaw, next_speed, action)

                    delta_yaws = next_yaw.item() * 180.0 / np.pi

                    transform             = carla.Transform(carla.Location(x=next_loc[0].item(), y=next_loc[1].item(), z=traffic_transform.location.z))
                    bounding_box          = carla.BoundingBox(transform.location, vehicle.bounding_box.extent)
                    bounding_box.rotation = carla.Rotation(pitch=float(traffic_transform.rotation.pitch),
                                                        yaw=float(delta_yaws),
                                                        roll=float(traffic_transform.rotation.roll))

                if (statistics.mean(self.vehicle_speed_buffer[vehicle.id]["velocity"]) < self.stuck_vel_threshold
                        and statistics.mean(self.vehicle_speed_buffer[vehicle.id]["throttle"]) > self.stuck_throttle_threshold
                        and statistics.mean(self.vehicle_speed_buffer[vehicle.id]["brake"]) < self.stuck_brake_threshold):
                    tmp_stucked_vehicle_id.append(vehicle.id)

                nearby_vehicles[vehicle.id] = veh_future_bbs

        # delete old vehicles
        to_delete = set(self.vehicle_speed_buffer.keys()).difference(tmp_near_vehicle_id)
        for d in to_delete:
            del self.vehicle_speed_buffer[d]

        # -----------------------------------------------------------
        # Intersection checks with ego vehicle
        # -----------------------------------------------------------
        number_of_interpolation_frames = self.frame_rate // 2 #TODO only checked for 20fps
        cur_loc   = np.array([[vehicle_transform.location.x, vehicle_transform.location.y]])
        cur_loc_ego   = np.array([[0, 0]])

        vehicl_yaw = (vehicle_transform.rotation.yaw + 90)* np.pi / 180
        rotation = np.array(
                [[np.cos(vehicl_yaw), np.sin(vehicl_yaw)],
                [-np.sin(vehicl_yaw),  np.cos(vehicl_yaw)]]
            )
        # tp = target_point[0] @ rotation

        future_loc = cur_loc + ego_wps[0]@ rotation
        all_locs = np.append(cur_loc, future_loc, axis=0)
        all_locs_ego = np.append(cur_loc_ego, ego_wps[0], axis=0)
        cur_yaw   = np.array([(vehicle_transform.rotation.yaw) / 180.0 * np.pi])
        prev_yaw = cur_yaw
        # next_speed_no_brake = np.array([speed])

        #NOTE intentionally set ego vehicle to move at the target speed (we want to know if there is an intersection if we would not brake)
        # throttle_extrapolation = self._get_throttle_extrapolation(self.target_speed, speed)
        # action_no_brake     = np.array(np.stack([self.steer, throttle_extrapolation, 0.0], axis=-1))

        back_only_vehicle_id = []
        # ego_future = []

        for i in range(1, 1+ego_wps.shape[1]): # TODO: check dimension!!!
            if self.render_bev==False and self.junction==False and i > number_of_future_frames_no_junction:
                alpha = 255
                color_value = 50
                break
            else:
                alpha = 50
                color_value = 255

            delta_yaw = math.atan2(all_locs_ego[i][0]-all_locs_ego[i-1][0], all_locs_ego[i][1]-all_locs_ego[i-1][1])
            next_yaw = cur_yaw - delta_yaw

            for k in range(number_of_interpolation_frames):

                tmp_loc = all_locs[i-1] + (all_locs[i]-all_locs[i-1])/number_of_interpolation_frames * k
                tmp_yaw = prev_yaw + (next_yaw-prev_yaw)/number_of_interpolation_frames * k
                # cur_yaw = next_yaw # + delta_yaw
                # next_yaw = np.array([delta_yaw]) #+ delta_yaw

                next_yaw_deg = tmp_yaw.item() * 180.0 / np.pi
                cosine = np.cos(tmp_yaw.item())
                sine = np.sin(tmp_yaw.item())

                extent           = self._vehicle.bounding_box.extent
                extent.x         = extent.x / 2.

                # front half
                transform             = carla.Transform(carla.Location(x=tmp_loc[0].item()+extent.x*cosine, y=tmp_loc[1].item()+extent.y*sine, z=vehicle_transform.location.z))
                bounding_box          = carla.BoundingBox(transform.location, extent)
                bounding_box.rotation = carla.Rotation(pitch=float(vehicle_transform.rotation.pitch), yaw=float(next_yaw_deg), roll=float(vehicle_transform.rotation.roll))

                # back half
                transform_back             = carla.Transform(carla.Location(x=tmp_loc[0].item()-extent.x*cosine, y=tmp_loc[1].item()-extent.y*sine, z=vehicle_transform.location.z))
                bounding_box_back          = carla.BoundingBox(transform_back.location, extent)
                bounding_box_back.rotation = carla.Rotation(pitch=float(vehicle_transform.rotation.pitch), yaw=float(next_yaw_deg), roll=float(vehicle_transform.rotation.roll))
                
                index = k + (i-1) * number_of_interpolation_frames
                i_stuck = index

                for id, traffic_participant in nearby_vehicles.items():
                    if self.render_bev==False and self.junction==False and i > number_of_future_frames_no_junction:
                            break
                    if id in tmp_stucked_vehicle_id:
                        i_stuck = 0
                    back_intersect = (self.check_obb_intersection(bounding_box_back, traffic_participant[i_stuck]) == True)
                    front_intersect = (self.check_obb_intersection(bounding_box, traffic_participant[i_stuck]) == True)
                    if id in back_only_vehicle_id:
                        back_only_vehicle_id.remove(id)
                        if back_intersect:
                            back_only_vehicle_id.append(id)
                        continue
                    if back_intersect and not front_intersect:
                        back_only_vehicle_id.append(id)
                    if front_intersect:
                        if self.junction==True or i <= number_of_future_frames_no_junction:
                            vehicle_hazard = True
                        self.vehicle_hazard[i] = True
                    
                for walker in nearby_walkers:
                    if self.render_bev==False and self.junction==False and i > number_of_future_frames_no_junction:
                            break
                    if (self.check_obb_intersection(bounding_box, walker[i]) == True):
                        if self.junction==True or i <= number_of_future_frames_no_junction:
                            walker_hazard = True
                        self.walker_hazard[i] = True

                prev_yaw = next_yaw


        return (vehicle_hazard or walker_hazard)


    def _get_forward_speed(self, transform=None, velocity=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def dot_product(self, vector1, vector2):
        return (vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z)

    def cross_product(self, vector1, vector2):
        return carla.Vector3D(x=vector1.y * vector2.z - vector1.z * vector2.y, y=vector1.z * vector2.x - vector1.x * vector2.z, z=vector1.x * vector2.y - vector1.y * vector2.x)

    def get_separating_plane(self, rPos, plane, obb1, obb2):
        ''' Checks if there is a seperating plane
        rPos Vec3
        plane Vec3
        obb1  Bounding Box
        obb2 Bounding Box
        '''
        return (abs(self.dot_product(rPos, plane)) > (abs(self.dot_product((obb1.rotation.get_forward_vector() * obb1.extent.x), plane)) +
                                                      abs(self.dot_product((obb1.rotation.get_right_vector()   * obb1.extent.y), plane)) +
                                                      abs(self.dot_product((obb1.rotation.get_up_vector()      * obb1.extent.z), plane)) +
                                                      abs(self.dot_product((obb2.rotation.get_forward_vector() * obb2.extent.x), plane)) +
                                                      abs(self.dot_product((obb2.rotation.get_right_vector()   * obb2.extent.y), plane)) +
                                                      abs(self.dot_product((obb2.rotation.get_up_vector()      * obb2.extent.z), plane)))
                )
    
    def check_obb_intersection(self, obb1, obb2):
        RPos = obb2.location - obb1.location
        return not(self.get_separating_plane(RPos, obb1.rotation.get_forward_vector(), obb1, obb2) or
                   self.get_separating_plane(RPos, obb1.rotation.get_right_vector(),   obb1, obb2) or
                   self.get_separating_plane(RPos, obb1.rotation.get_up_vector(),      obb1, obb2) or
                   self.get_separating_plane(RPos, obb2.rotation.get_forward_vector(), obb1, obb2) or
                   self.get_separating_plane(RPos, obb2.rotation.get_right_vector(),   obb1, obb2) or
                   self.get_separating_plane(RPos, obb2.rotation.get_up_vector(),      obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_forward_vector()), obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_right_vector()),   obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_up_vector()),      obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_right_vector()  , obb2.rotation.get_forward_vector()), obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_right_vector()  , obb2.rotation.get_right_vector()),   obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_right_vector()  , obb2.rotation.get_up_vector()),      obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_up_vector()     , obb2.rotation.get_forward_vector()), obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_up_vector()     , obb2.rotation.get_right_vector()),   obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_up_vector()     , obb2.rotation.get_up_vector()),      obb1, obb2))


    # Optimized version of get _get_angle_to. A lot faster, since calculations are done in math and not numpy
    def _get_angle_to(self, pos, theta, target): # 2 - 3 mu
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        diff = target - pos
        aim_0 = ( cos_theta * diff[0] + sin_theta * diff[1])
        aim_1 = (-sin_theta * diff[0] + cos_theta * diff[1])

        angle = -math.degrees(math.atan2(-aim_1, aim_0))
        angle = np.float_(angle) # So that the optimized function has the same datatype as output.
        return angle

    def get_nearby_object(self, vehicle_position, actor_list, radius):
        nearby_objects = []
        for actor in actor_list:
            trigger_box_global_pos = actor.get_transform().transform(actor.trigger_volume.location)
            trigger_box_global_pos = carla.Location(x=trigger_box_global_pos.x, y=trigger_box_global_pos.y, z=trigger_box_global_pos.z)
            if (trigger_box_global_pos.distance(vehicle_position) < radius):
                nearby_objects.append(actor)
        return nearby_objects


class EgoModel():
    def __init__(self, dt=1./4):
        self.dt = dt

        # Kinematic bicycle model. Numbers are the tuned parameters from World on Rails
        self.front_wb    = -0.090769015
        self.rear_wb     = 1.4178275

        self.steer_gain  = 0.36848336
        self.brake_accel = -4.952399
        self.throt_accel = 0.5633837

    def forward(self, locs, yaws, spds, acts):
        # Kinematic bicycle model. Numbers are the tuned parameters from World on Rails
        steer = acts[..., 0:1].item()
        throt = acts[..., 1:2].item()
        brake = acts[..., 2:3].astype(np.uint8)

        if (brake):
            accel = self.brake_accel
        else:
            accel = self.throt_accel * throt

        wheel = self.steer_gain * steer

        beta = math.atan(self.rear_wb / (self.front_wb + self.rear_wb) * math.tan(wheel))
        yaws = yaws.item()
        spds = spds.item()
        next_locs_0 = locs[0].item() + spds * math.cos(yaws + beta) * self.dt
        next_locs_1 = locs[1].item() + spds * math.sin(yaws + beta) * self.dt
        next_yaws = yaws + spds / self.rear_wb * math.sin(beta) * self.dt
        next_spds = spds + accel * self.dt
        next_spds = next_spds * (next_spds > 0.0)  # Fast ReLU

        next_locs = np.array([next_locs_0, next_locs_1])
        next_yaws = np.array(next_yaws)
        next_spds = np.array(next_spds)

        return next_locs, next_yaws, next_spds
