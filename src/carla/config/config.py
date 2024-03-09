import os


from src.carla.config.base_config import BaseConfig
from src.carla.config.observation_configs import BaseObservationConfig
from src.carla.config.action_configs import BaseActionConfig
from src.carla.config.scenario_configs import BaseScenarioConfig


class BaseMainConfig(BaseConfig):
    """Base Class defining the parameters required in main config.

    DO NOT instantiate this directly. Instead, using DefaultMainConfig
    """
    def __init__(self):
        self.scenario_config = None
        self.obs_config = None
        self.action_config = None

        # Are we testing?
        self.testing = None

        #### Server Setup ####
        self.server_path = None
        self.server_binary = None
        self.server_fps = None
        self.server_port = None
        self.server_retries = None
        self.sync_mode = None
        self.client_timeout_seconds = None
        self.carla_gpu = None


    def populate_config(
            self,
            observation_config,
            action_config,
            scenario_config,
            testing=False,
            carla_gpu=0,
            render_server=False,
    ):
        """Fill in the config parameters that are not set by default
        """
        # Observation Config
        if(isinstance(observation_config, BaseObservationConfig)):
            self.obs_config = observation_config
        else:
            # Invalid Argument
            raise Exception("Invalid argument for observation_config")

        # Action Config
        if(isinstance(action_config, BaseActionConfig)):
            self.action_config = action_config
        else:
            # Invalid Argument
            raise Exception("Invalid argument for action_config")

        # Scenario Config
        if(isinstance(scenario_config, BaseScenarioConfig)):
            # Just save object, since it is already instantiated
            self.scenario_config = scenario_config
        else:
            # Invalid Argument
            raise Exception("Invalid argument for scenario_config")

        # Testing
        self.testing = testing

        # Carla GPU
        self.carla_gpu = carla_gpu

        # whether to display server
        self.render_server = render_server


class DefaultMainConfig(BaseMainConfig):
    """Default Config for the server
    """
    def __init__(self):
        super().__init__()
        #### Server Setup ####
        self.server_path = os.environ.get("CARLA_PATH")
        self.server_binary = self.server_path + '/CarlaUE4.sh'
        self.server_fps = 10
        self.server_port = -1
        self.server_retries = 5
        self.sync_mode = True
        self.client_timeout_seconds = 30
