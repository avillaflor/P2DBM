from __future__ import print_function
import collections
import weakref
import carla
import queue
import numpy as np


class SensorManager():
    # Manages the CARLA sensors for an ego-vehicle
    def __init__(self, config, parent_actor):
        self.config = config

        self.parent_actor = parent_actor
        self.sensor_names = list(self.config.obs_config.sensors.keys())
        # Assuming sensors_configs as a list of dictionaries
        self.sensor_configs = list(self.config.obs_config.sensors.values())
        self.sensors = {}

    def spawn(self):
        for idx, (sensor_name, sensor_config) in enumerate(zip(self.sensor_names, self.sensor_configs)):
            if sensor_name=="collision_sensor":
                sensor = CollisionSensor(self.parent_actor)
            elif sensor_name=="lane_invasion_sensor":
                sensor = LaneInvasionSensor(self.parent_actor)
            elif 'camera' in sensor_name:
                sensor_config.update({'name':sensor_name})
                sensor = CameraSensor(self.parent_actor, sensor_config)
            else:
                raise Exception("Sensor {} not supported".format(sensor_name))

            self.sensors[sensor_name] = sensor

    def get_sensor_readings(self, world_frame=None):
        sensor_readings = {}
        for idx, k in enumerate(self.sensor_names):
            if k=="collision_sensor":
                sensor_readings[k] = {'num_collisions': self.sensors[k].num_collisions,\
                                        'collision_actor_id': self.sensors[k].actor_id,\
                                        'collision_actor_type': self.sensors[k].actor_type}
            elif k=="lane_invasion_sensor":
                sensor_readings[k] = {'num_lane_intersections': self.sensors[k].num_laneintersections,\
                                        'out_of_road': self.sensors[k].out_of_road}
            elif 'camera' in k:
                if world_frame is None:
                    print("No world frame found! Skipping reading from camera sensor!!")
                else:
                    camera_image = self.sensors[k]._read_data(world_frame)
                    sensor_readings[k] = {'image': camera_image}
            else:
                print("Uninitialized sensor!")

        return sensor_readings


class CollisionSensor(object):
    def __init__(self, parent_actor, config=None):
        self.sensor = None
        self.num_collisions = 0
        self.actor_id = None
        self.actor_type = None
        self._history = []
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self._history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = event.other_actor.type_id
        if 'road' not in actor_type:
            self.actor_id = event.other_actor.id
            self.actor_type = actor_type
            self.num_collisions += 1


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, config=None):
        self.sensor = None
        self._parent = parent_actor
        self.num_laneintersections = 0
        self.out_of_road = False
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_changes = set(x.lane_change for x in event.crossed_lane_markings)
        if carla.libcarla.LaneChange.NONE in lane_changes or carla.libcarla.LaneChange.Right in lane_changes:
            self.num_laneintersections += 1

        lane_types = set(x.type for x in event.crossed_lane_markings)
        if carla.libcarla.LaneMarkingType.NONE in lane_types:
            self.out_of_road = True


class CameraSensor(object):
    def __init__(self, parent_actor,config=None, verbose = False):
        self.sensor = None
        self._parent = parent_actor
        self.transform = carla.Transform(carla.Location(x=config['x'], z=config['z']), carla.Rotation(pitch=config['pitch']))

        self.verbose = verbose
        self.config = config
        self.camera_queue = queue.Queue()
        self.name = config['name']
        world = self._parent.get_world()
        blueprint_library = world.get_blueprint_library()
        sensor_bp = blueprint_library.find(config['name'].split('/')[0])
        sensor_bp.set_attribute('image_size_x', config['sensor_x_res'])
        sensor_bp.set_attribute('image_size_y', config['sensor_y_res'])
        sensor_bp.set_attribute('sensor_tick', config['sensor_tick'])
        sensor_bp.set_attribute('fov', config['fov'])

        self.sensor = world.spawn_actor(sensor_bp, self.transform, attach_to=self._parent)

        self.sensor.listen(self.camera_queue.put)

    def _read_data(self, world_frame, timeout=240.0):
        cam_image = self._retrieve_data(world_frame, timeout)
        cam_image_p = self._preprocess_image(cam_image)
        return cam_image_p

    def _retrieve_data(self, world_frame, timeout):
        while True:
            data = self.camera_queue.get(timeout=timeout)
            if data.frame == world_frame:
                return data
            else:
                if self.verbose:
                    print("difference in frames, world_frame={0}, data_frame={1}".format(world_frame, data.frame))

    def _preprocess_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array
