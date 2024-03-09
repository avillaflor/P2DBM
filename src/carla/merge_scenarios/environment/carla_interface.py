import time
import carla
import numpy as np


from src.carla.features.scenarios import Scenario, BatchScenario
from src.carla.merge_scenarios.environment.utils import compute_done_condition
from src.carla.merge_scenarios.environment.server import CarlaServer
from src.carla.merge_scenarios.environment.ego_vehicle_manager import EgoVehicleManager
from src.carla.merge_scenarios.environment.npcs_vehicle_manager import NPCsVehicleManager


class CarlaInterface():
    # Actual interface with CARLA
    def __init__(self, config, num_agents, ap_class, ap_kwargs):
        self.config = config

        # Instantiate and start server
        self.server = CarlaServer(config)

        self.client = None

        self.num_agents = num_agents
        self.ego_agents = [None for _ in range(num_agents)]
        self.actives = np.zeros(self.num_agents, dtype=bool)
        self.npcs_vehicle_manager = None

        self._f = self.config.obs_config.f
        self._history_len = self._f * self.config.obs_config.H + 1

        self._ap_class = ap_class
        self._ap_kwargs = ap_kwargs

        self.setup()

    def setup(self):
        # Start the carla server and get a client
        self.server.start()
        self.client = self._spawn_client()

        # Get the world
        self.world = self.client.load_world(self.config.scenario_config.city_name)

        # Update the settings from the config
        settings = self.world.get_settings()
        if(self.config.sync_mode):
            settings.synchronous_mode = True
        if self.config.server_fps is not None and self.config.server_fps != 0:
            settings.fixed_delta_seconds = 1.0 / float(self.config.server_fps)

        # Enable rendering
        settings.no_rendering_mode = not self.config.render_server

        self.world.apply_settings(settings)

        # pseudo random so we can set seed
        tm_port = np.random.randint(10000, 50000) + (int(time.time() * 1e9) % 10000)
        self.tm = self.client.get_trafficmanager(tm_port)
        self.tm.set_synchronous_mode(True)
        if self.config.testing:
            self.tm.set_random_device_seed(0)

        # Sleep to allow for settings to update
        time.sleep(5)

        # Retrieve map
        self.map = self.world.get_map()

        # Get blueprints
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()

        self.npcs_vehicle_manager = NPCsVehicleManager(self.config, self.world, self.tm)
        # Instantiate a vehicle manager to handle other actors
        for i in range(self.num_agents):
            self.ego_agents[i] = EgoVehicleManager(self.config, self.world, self.tm, self._ap_class, self._ap_kwargs)

        # Get traffic lights
        self.traffic_actors = self.world.get_actors().filter("*traffic_light*")

        print("server_version", self.client.get_server_version())

    def _spawn_client(self, hostname='localhost', port_number=None):
        port_number = self.server.server_port
        client = carla.Client(hostname, port_number)
        client.set_timeout(self.config.client_timeout_seconds)

        return client

    def _set_scenario(self):
        for index in range(self.num_agents):
            self.ego_agents[index].spawn(self.config.scenario_config.ego_spawn_transform, self.config.scenario_config.ego_dest_transform)

    # Assuming this is a full reset
    def reset(self, index=0):
        # Delete old actors
        self.destroy_actors()

        self._set_scenario()

        # Spawn new actors
        # best way to get reproducible results
        self.npcs_vehicle_manager.spawn(index=index)

        # Tick for 15 frames to handle car initialization in air
        for _ in range(15):
            world_frame = self.world.tick()

        default_control = {
            "target_speed": 0.0,
            "control_steer": 0.0,
            "control_throttle": 0.0,
            "control_brake": 0.0,
            "control_reverse": False,
            "control_hand_brake": False
        }

        obses = []
        scenarios = []
        infos = []
        self._obs_history = []
        for i, ego_agent in enumerate(self.ego_agents):
            ego_obs = ego_agent.obs(world_frame)
            obs = {**ego_obs, **default_control}
            self._obs_history.append([obs])
            scenario = Scenario()
            scenario.setup(
                self._obs_history[i],
                config=self.config,
            )
            scenarios.append(scenario)
            infos.append(obs)

        batch_scenario = BatchScenario(scenarios)
        obses = batch_scenario.get_obs(ref_t=0, f=self._f)

        self.prev_infos = infos
        self.actives = np.ones(self.num_agents, dtype=bool)
        return obses

    def step(self, actions):
        controls = []
        for i in range(self.num_agents):
            if self.actives[i]:
                self.ego_agents[i].step(actions[i])

        world_frame = self.world.tick()

        for i in range(self.num_agents):
            if self.actives[i]:
                self.ego_agents[i].check_for_vehicle_elimination()
                control = self.ego_agents[i].get_last_action()
                controls.append(control)
            else:
                controls.append(None)

        self.npcs_vehicle_manager.step()

        obses = []
        dones = []
        infos = []

        scenarios = []
        for i in range(self.num_agents):
            if self.actives[i]:
                ego_obs = self.ego_agents[i].obs(world_frame)
                obs = {**ego_obs, **controls[i]}
                self._obs_history[i] = [*self._obs_history[i][-(self._history_len-1):], obs]
                scenario = Scenario()
                scenario.setup(
                    self._obs_history[i],
                    config=self.config,
                )
                done = compute_done_condition(
                    prev_episode_measurements=self.prev_infos[i],
                    curr_episode_measurements=obs,
                    config=self.config)
                self.actives[i] = not done
                self.prev_infos = infos
                if done:
                    self.ego_agents[i].destroy_actors()
            else:
                obs = self._obs_history[i][-1]
                scenario = Scenario()
                scenario.setup(
                    self._obs_history[i],
                    config=self.config,
                )
                done = True

            scenarios.append(scenario)
            infos.append(obs)
            dones.append(done)

        batch_scenario = BatchScenario(scenarios)
        obses = batch_scenario.get_obs(ref_t=-1, f=self._f)

        return obses, dones, infos

    def get_autopilot_actions(self):
        actions = []
        for i in range(self.num_agents):
            if self.actives[i]:
                actions.append(self.ego_agents[i].get_autopilot_action())
            else:
                actions.append(np.array([0., -1.]))
        actions = np.stack(actions, axis=0)
        return actions

    @property
    def actor_list(self):
        actor_list = []
        for ego_agent in self.ego_agents:
            actor_list += ego_agent.actor_list
        actor_list += self.npcs_vehicle_manager.actor_list
        return actor_list

    def destroy_actors(self):
        for ego_agent in self.ego_agents:
            ego_agent.destroy_actors()
        self.npcs_vehicle_manager.destroy_actors()

    def close(self):
        self.destroy_actors()

        if self.server is not None:
            self.server.close()
