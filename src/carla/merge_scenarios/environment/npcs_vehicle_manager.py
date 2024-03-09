import numpy as np


class NPCsVehicleManager():
    # Manages the other vehicles in the scene
    def __init__(self, config, world, tm):
        self.config = config
        self.world = world
        self.tm = tm
        self.map = self.world.get_map()

        if hasattr(self.config.scenario_config, 'npc_spawn_points'):
            self.spawn_points = self.config.scenario_config.npc_spawn_points
        else:
            self.spawn_points = self.world.get_map().get_spawn_points()

        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        if self.config.scenario_config.disable_two_wheeler:
            self.vehicle_blueprints = [x for x in self.vehicle_blueprints if int(x.get_attribute('number_of_wheels')) == 4]

        self.tm_actor_list = []
        self.ap_actor_list = []

    def spawn(self, index=0):
        if self.config.testing:
            curr_state = np.random.get_state()
            np.random.seed(index)

        number_of_vehicles = self.config.scenario_config.num_npc

        self.spawn_npc(number_of_vehicles)
        self.world.reset_all_traffic_lights()

        if self.config.testing:
            np.random.set_state(curr_state)

    def step(self):
        self.check_for_vehicle_elimination()
        for ap in self.ap_actor_list:
            ap.step()

    def check_for_vehicle_elimination(self):
        # https://github.com/carla-simulator/carla/issues/3860
        self.tm_actor_list = [actor for actor in self.tm_actor_list if actor.is_alive]
        self.ap_actor_list = [actor for actor in self.ap_actor_list if actor.is_alive]

    def spawn_npc(self, number_of_vehicles):
        npc_spawn_points = self.pick_npc_spawn_points()
        count = number_of_vehicles
        if count > 0:
            for spawn_point in npc_spawn_points:
                if self.try_spawn_random_vehicle_at(self.vehicle_blueprints, spawn_point):
                    count -= 1
                if count <= 0:
                    break

    def pick_npc_spawn_points(self):
        spawn_points = np.random.permutation(self.spawn_points)
        return spawn_points

    def try_spawn_random_vehicle_at(self, blueprints, transform):
        # To spawn same type of vehicle
        blueprint = blueprints[0]
        if blueprint.has_attribute('color'):
            blueprint.set_attribute('color', '255,255,255')

        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)

        if vehicle is not None:
            self.tm_actor_list.append(vehicle)
            tm_port = self.tm.get_port()
            vehicle.set_autopilot(True, tm_port)
            self.tm.ignore_lights_percentage(vehicle, 100.)
            self.tm.ignore_signs_percentage(vehicle, 0)
            self.tm.ignore_vehicles_percentage(vehicle, 0)
            self.tm.ignore_walkers_percentage(vehicle, 0)
            self.tm.vehicle_percentage_speed_difference(vehicle, -30. + (np.random.random() * 60.))
            self.tm.distance_to_leading_vehicle(vehicle, 2. + 4 * np.random.random())

            return True
        return False

    def destroy_actors(self):
        for _ in range(len(self.tm_actor_list)):
            try:
                actor = self.tm_actor_list.pop()
                actor.destroy()
            except Exception:
                pass

        for _ in range(len(self.ap_actor_list)):
            try:
                actor = self.ap_actor_list.pop()
                actor.destroy()
            except Exception:
                pass
