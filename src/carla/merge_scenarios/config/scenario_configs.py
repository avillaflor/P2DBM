import carla
import numpy as np


from src.carla.config.scenario_configs import DefaultScenarioConfig


class OnRampTown03Config(DefaultScenarioConfig):
    def __init__(self):
        super().__init__()
        self.scenarios = "merge"
        self.num_npc = 10

        self.city_name = "Town03"

        self.max_static_steps = 60

        spawn_location = carla.Location(-104.0, 19.8, 0.3)
        spawn_rotation = carla.Rotation(0.0, 135., 0.0)
        self.ego_spawn_transform = carla.Transform(spawn_location, spawn_rotation)

        dest_location = carla.Location(-145.5, 7.1, 0.0)
        dest_rotation = carla.Rotation(0.0, -90.0, 0.0)
        self.ego_dest_transform = carla.Transform(dest_location, dest_rotation)

        npc_location = carla.Location(-145.5, 7.1, 0.)
        npc_rotation = carla.Rotation(0.0, -90., 0.0)
        self.npc_destination = carla.Transform(npc_location, npc_rotation)

        min_y = 7.
        max_y = 117.
        min_x = -145.5
        max_x = -145.5
        min_z = 0.29
        max_z = 0.29
        min_yaw = -90.0
        max_yaw = -90.0

        poss_y = np.arange(min_y, max_y, 7.5)
        n = len(poss_y)
        poss_x = np.linspace(min_x, max_x, n)
        poss_z = np.linspace(min_z, max_z, n)
        poss_yaw = np.linspace(min_yaw, max_yaw, n)
        spawn_points = []
        for i in range(n):
            transform = carla.Transform(
                carla.Location(poss_x[i], poss_y[i], poss_z[i]),
                carla.Rotation(0., poss_yaw[i], 0.))
            spawn_points.append(transform)

        self.npc_spawn_points = spawn_points

        min_y = 30.
        max_y = 44.4
        max_x = -128.4
        min_x = -114.0
        min_z = 0.3
        max_z = 0.3
        min_yaw = 135.
        max_yaw = 135.

        poss_y = np.arange(min_y, max_y, 7.5)
        n = len(poss_y)
        poss_x = np.linspace(min_x, max_x, n)
        poss_z = np.linspace(min_z, max_z, n)
        poss_yaw = np.linspace(min_yaw, max_yaw, n)
        front_spawn_points = []
        for i in range(n):
            transform = carla.Transform(
                carla.Location(poss_x[i], poss_y[i], poss_z[i]),
                carla.Rotation(0., poss_yaw[i], 0.))
            front_spawn_points.append(transform)

        self.npc_spawn_points = self.npc_spawn_points + front_spawn_points


class RoundAboutTown03Config(DefaultScenarioConfig):
    def __init__(self):
        super().__init__()
        self.scenarios = "merge"
        self.num_npc = 12

        self.city_name = "Town03"

        self.max_static_steps = 60

        spawn_location = carla.Location(64.6, -7.8, 5.0)
        spawn_rotation = carla.Rotation(0.0, 180.0, 0.0)
        self.ego_spawn_transform = carla.Transform(spawn_location, spawn_rotation)

        dest_location = carla.Location(7.6, -47.3, 0.0)
        dest_rotation = carla.Rotation(0.0, -90.0, 0.0)
        self.ego_dest_transform = carla.Transform(dest_location, dest_rotation)

        npc_location = carla.Location(210, -195.3, 0.)
        npc_rotation = carla.Rotation(0.0, 0.0, 0.0)
        self.npc_destination = carla.Transform(npc_location, npc_rotation)

        radius = 23.
        center = np.array((0.0, 0.0))

        n = 24

        angles = np.linspace(90., 450., n + 1)[:-1]

        poss_x = np.cos(angles * np.pi / 180.) * radius + center[0]
        poss_y = np.sin(angles * np.pi / 180.) * radius + center[1]
        poss_z = np.linspace(0.3, 0.3, n)
        poss_yaw = (angles - 90.)

        spawn_points = []
        for i in range(n):
            transform = carla.Transform(
                carla.Location(poss_x[i], poss_y[i], poss_z[i]),
                carla.Rotation(0., poss_yaw[i], 0.))
            spawn_points.append(transform)

        self.npc_spawn_points = spawn_points


class RoundAbout2Town03Config(DefaultScenarioConfig):
    def __init__(self):
        super().__init__()
        self.scenarios = "merge"
        self.num_npc = 12

        self.city_name = "Town03"

        self.max_static_steps = 60

        spawn_location = carla.Location(-53.3, 0.6, 1.0)
        spawn_rotation = carla.Rotation(0.0, 0.0, 0.0)
        self.ego_spawn_transform = carla.Transform(spawn_location, spawn_rotation)

        dest_location = carla.Location(-10.5, 43.6, 0.0)
        dest_rotation = carla.Rotation(0.0, 90.0, 0.0)
        self.ego_dest_transform = carla.Transform(dest_location, dest_rotation)

        npc_location = carla.Location(210, -195.3, 0.)
        npc_rotation = carla.Rotation(0.0, 0.0, 0.0)
        self.npc_destination = carla.Transform(npc_location, npc_rotation)

        radius = 23.
        center = np.array((0.0, 0.0))

        n = 24

        angles = np.linspace(90., 450., n + 1)[:-1]

        poss_x = np.cos(angles * np.pi / 180.) * radius + center[0]
        poss_y = np.sin(angles * np.pi / 180.) * radius + center[1]
        poss_z = np.linspace(0.3, 0.3, n)
        poss_yaw = (angles - 90.)

        spawn_points = []
        for i in range(n):
            transform = carla.Transform(
                carla.Location(poss_x[i], poss_y[i], poss_z[i]),
                carla.Rotation(0., poss_yaw[i], 0.))
            spawn_points.append(transform)

        self.npc_spawn_points = spawn_points


class OnRampTown04Config(DefaultScenarioConfig):
    def __init__(self):
        super().__init__()
        self.scenarios = "merge"
        self.num_npc = 26

        self.city_name = "Town04"

        self.max_static_steps = 60

        spawn_location = carla.Location(-100.2, -85.7, 4.9)
        spawn_rotation = carla.Rotation(-4.3, -42.0, 0.0)
        self.ego_spawn_transform = carla.Transform(spawn_location, spawn_rotation)

        dest_location = carla.Location(-16.16, -30.5, 0.0)
        dest_rotation = carla.Rotation(0.0, 90.0, 0.0)
        self.ego_dest_transform = carla.Transform(dest_location, dest_rotation)

        npc_location = carla.Location(385.7, -187.7, 0.3)
        npc_rotation = carla.Rotation(0.0, 90.5, 0.0)
        self.npc_destination = carla.Transform(npc_location, npc_rotation)

        min_y = -212.5
        max_y = 101.6
        min_x = -16.9
        max_x = -15.6
        min_z = 0.29
        max_z = 0.29
        min_yaw = 90.0
        max_yaw = 90.0

        poss_y = np.arange(min_y, max_y, 10.)
        n = len(poss_y)
        poss_x = np.linspace(min_x, max_x, n)
        poss_z = np.linspace(min_z, max_z, n)
        poss_yaw = np.linspace(min_yaw, max_yaw, n)
        spawn_points = []
        for i in range(n):
            transform = carla.Transform(
                carla.Location(poss_x[i], poss_y[i], poss_z[i]),
                carla.Rotation(0., poss_yaw[i], 0.))
            spawn_points.append(transform)

        self.npc_spawn_points = spawn_points


class OnRamp2Town04Config(DefaultScenarioConfig):
    def __init__(self):
        super().__init__()
        self.scenarios = "merge"
        self.num_npc = 26

        self.city_name = "Town04"

        self.max_static_steps = 300

        spawn_location = carla.Location(395.1, 67.9, 1.0)
        spawn_rotation = carla.Rotation(-4.3, -70.0, 0.0)
        self.ego_spawn_transform = carla.Transform(spawn_location, spawn_rotation)

        dest_location = carla.Location(412.2, -35.1, 0.0)
        dest_rotation = carla.Rotation(0.0, -90.0, 0.0)
        self.ego_dest_transform = carla.Transform(dest_location, dest_rotation)

        npc_location = carla.Location(385.7, -187.7, 0.3)
        npc_rotation = carla.Rotation(0.0, 90.5, 0.0)
        self.npc_destination = carla.Transform(npc_location, npc_rotation)

        min_x = 140.
        max_x = 360.
        min_y = 42.0
        max_y = 42.0
        min_z = 2.0
        max_z = 4.0
        min_yaw = 0.0
        max_yaw = 0.0

        poss_x = np.arange(min_x, max_x, 7.5)
        n = len(poss_x)
        poss_y = np.linspace(min_y, max_y, n)
        poss_z = np.linspace(min_z, max_z, n)
        poss_yaw = np.linspace(min_yaw, max_yaw, n)
        spawn_points = []
        for i in range(n):
            transform = carla.Transform(
                carla.Location(poss_x[i], poss_y[i], poss_z[i]),
                carla.Rotation(0., poss_yaw[i], 0.))
            spawn_points.append(transform)

        self.npc_spawn_points = spawn_points

        min_x = 397.6
        max_x = 406.
        min_y = 59.1
        max_y = 9.1
        min_z = 1.0
        max_z = 1.0
        min_yaw = -70.
        max_yaw = -70.

        poss_x = np.arange(min_x, max_x, 2.5)
        n = len(poss_x)
        poss_y = np.linspace(min_y, max_y, n)
        poss_z = np.linspace(min_z, max_z, n)
        poss_yaw = np.linspace(min_yaw, max_yaw, n)
        front_spawn_points = []
        for i in range(n):
            transform = carla.Transform(
                carla.Location(poss_x[i], poss_y[i], poss_z[i]),
                carla.Rotation(0., poss_yaw[i], 0.))
            front_spawn_points.append(transform)

        self.npc_spawn_points = self.npc_spawn_points + front_spawn_points


class OnRampTown06Config(DefaultScenarioConfig):
    def __init__(self):
        super().__init__()
        self.scenarios = "merge"
        self.num_npc = 12

        self.city_name = "Town06"

        self.max_static_steps = 60

        spawn_location = carla.Location(174.2, 93.1, 0.3)
        spawn_rotation = carla.Rotation(0.0, -35.0, 0.0)
        self.ego_spawn_transform = carla.Transform(spawn_location, spawn_rotation)

        dest_location = carla.Location(280.0, 52.4, 0.0)
        dest_rotation = carla.Rotation(0.0, 0.0, 0.0)
        self.ego_dest_transform = carla.Transform(dest_location, dest_rotation)

        npc_location = carla.Location(300.0, 52.4, 0.3)
        npc_rotation = carla.Rotation(0.0, 0.0, 0.0)
        self.npc_destination = carla.Transform(npc_location, npc_rotation)

        min_y = 52.4
        max_y = 52.4
        min_x = 150.
        max_x = 300.
        min_z = 0.3
        max_z = 0.3
        min_yaw = 0.0
        max_yaw = 0.0

        poss_x = np.arange(min_x, max_x, 10.)
        n = len(poss_x)
        poss_y = np.linspace(min_y, max_y, n)
        poss_z = np.linspace(min_z, max_z, n)
        poss_yaw = np.linspace(min_yaw, max_yaw, n)
        spawn_points = []
        for i in range(n):
            transform = carla.Transform(
                carla.Location(poss_x[i], poss_y[i], poss_z[i]),
                carla.Rotation(0., poss_yaw[i], 0.))
            spawn_points.append(transform)

        self.npc_spawn_points = spawn_points


class OnRamp2Town06Config(DefaultScenarioConfig):
    def __init__(self):
        super().__init__()
        self.scenarios = "merge"
        self.num_npc = 15

        self.city_name = "Town06"

        self.max_static_steps = 60

        spawn_location = carla.Location(141.8, 14.3, 0.3)
        spawn_rotation = carla.Rotation(0.0, -90., 0.0)
        self.ego_spawn_transform = carla.Transform(spawn_location, spawn_rotation)

        dest_location = carla.Location(83.4, -12.9, 0.0)
        dest_rotation = carla.Rotation(0.0, 180.0, 0.0)
        self.ego_dest_transform = carla.Transform(dest_location, dest_rotation)

        npc_location = carla.Location(60.0, -12.9, 0.3)
        npc_rotation = carla.Rotation(0.0, 180.0, 0.0)
        self.npc_destination = carla.Transform(npc_location, npc_rotation)

        min_y = -12.9
        max_y = -12.9
        min_x = 83.
        max_x = 253.
        min_z = 0.3
        max_z = 0.3
        min_yaw = 180.0
        max_yaw = 180.0

        poss_x = np.arange(min_x, max_x, 10.)
        n = len(poss_x)
        poss_y = np.linspace(min_y, max_y, n)
        poss_z = np.linspace(min_z, max_z, n)
        poss_yaw = np.linspace(min_yaw, max_yaw, n)
        spawn_points = []
        for i in range(n):
            transform = carla.Transform(
                carla.Location(poss_x[i], poss_y[i], poss_z[i]),
                carla.Rotation(0., poss_yaw[i], 0.))
            spawn_points.append(transform)

        self.npc_spawn_points = spawn_points


class OnRamp3Town06Config(DefaultScenarioConfig):
    def __init__(self):
        super().__init__()
        self.scenarios = "merge"
        self.num_npc = 20

        self.city_name = "Town06"

        self.max_static_steps = 60

        spawn_location = carla.Location(502.6, 33.4, 0.3)
        spawn_rotation = carla.Rotation(0.0, -30., 0.0)
        self.ego_spawn_transform = carla.Transform(spawn_location, spawn_rotation)

        dest_location = carla.Location(449.8, -9.9, 0.0)
        dest_rotation = carla.Rotation(0.0, 180.0, 0.0)
        self.ego_dest_transform = carla.Transform(dest_location, dest_rotation)

        npc_location = carla.Location(400., -9.9, 0.3)
        npc_rotation = carla.Rotation(0.0, 180.0, 0.0)
        self.npc_destination = carla.Transform(npc_location, npc_rotation)

        min_y = -9.9
        max_y = -9.9
        min_x = 400.
        max_x = 620.
        min_z = 0.3
        max_z = 0.3
        min_yaw = 180.0
        max_yaw = 180.0

        poss_x = np.arange(min_x, max_x, 10.)
        n = len(poss_x)
        poss_y = np.linspace(min_y, max_y, n)
        poss_z = np.linspace(min_z, max_z, n)
        poss_yaw = np.linspace(min_yaw, max_yaw, n)
        spawn_points = []
        for i in range(n):
            transform = carla.Transform(
                carla.Location(poss_x[i], poss_y[i], poss_z[i]),
                carla.Rotation(0., poss_yaw[i], 0.))
            spawn_points.append(transform)

        self.npc_spawn_points = spawn_points


class OnRamp4Town06Config(DefaultScenarioConfig):
    def __init__(self):
        super().__init__()
        self.scenarios = "merge"
        self.num_npc = 27

        self.city_name = "Town06"

        self.max_static_steps = 60

        spawn_location = carla.Location(-73.4, 192.6, 0.3)
        spawn_rotation = carla.Rotation(0.0, -90., 0.0)
        self.ego_spawn_transform = carla.Transform(spawn_location, spawn_rotation)

        dest_location = carla.Location(-210., 145.7, 0.0)
        dest_rotation = carla.Rotation(0.0, 180.0, 0.0)
        self.ego_dest_transform = carla.Transform(dest_location, dest_rotation)

        npc_location = carla.Location(-250., 145.7, 0.3)
        npc_rotation = carla.Rotation(0.0, 180.0, 0.0)
        self.npc_destination = carla.Transform(npc_location, npc_rotation)

        min_y = 145.7
        max_y = 145.7
        min_x = -240.
        max_x = -20.
        min_z = 0.3
        max_z = 0.3
        min_yaw = 180.0
        max_yaw = 180.0

        poss_x = np.arange(min_x, max_x, 7.5)
        n = len(poss_x)
        poss_y = np.linspace(min_y, max_y, n)
        poss_z = np.linspace(min_z, max_z, n)
        poss_yaw = np.linspace(min_yaw, max_yaw, n)
        spawn_points = []
        for i in range(n):
            transform = carla.Transform(
                carla.Location(poss_x[i], poss_y[i], poss_z[i]),
                carla.Rotation(0., poss_yaw[i], 0.))
            spawn_points.append(transform)

        self.npc_spawn_points = spawn_points


class OnRamp5Town06Config(DefaultScenarioConfig):
    def __init__(self):
        super().__init__()
        self.scenarios = "merge"
        self.num_npc = 18

        self.city_name = "Town06"

        self.max_static_steps = 60

        spawn_location = carla.Location(495.3, 119.6, 0.3)
        spawn_rotation = carla.Rotation(0.0, -45., 0.0)
        self.ego_spawn_transform = carla.Transform(spawn_location, spawn_rotation)

        dest_location = carla.Location(573., 52.4, 0.0)
        dest_rotation = carla.Rotation(0.0, 0.0, 0.0)
        self.ego_dest_transform = carla.Transform(dest_location, dest_rotation)

        npc_location = carla.Location(590., 52.4, 0.3)
        npc_rotation = carla.Rotation(0.0, 0.0, 0.0)
        self.npc_destination = carla.Transform(npc_location, npc_rotation)

        min_y = 52.4
        max_y = 52.4
        min_x = 480.0
        max_x = 630.
        min_z = 0.3
        max_z = 0.3
        min_yaw = 0.0
        max_yaw = 0.0

        poss_x = np.arange(min_x, max_x, 7.5)
        n = len(poss_x)
        poss_y = np.linspace(min_y, max_y, n)
        poss_z = np.linspace(min_z, max_z, n)
        poss_yaw = np.linspace(min_yaw, max_yaw, n)
        spawn_points = []
        for i in range(n):
            transform = carla.Transform(
                carla.Location(poss_x[i], poss_y[i], poss_z[i]),
                carla.Rotation(0., poss_yaw[i], 0.))
            spawn_points.append(transform)

        self.npc_spawn_points = spawn_points
