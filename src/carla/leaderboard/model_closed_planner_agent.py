import omegaconf
from lightning.fabric import Fabric


from src.carla.leaderboard.autopilot import AutoPilot
from src.carla.leaderboard.model_agent import ModelAgent
from src.carla.config.config import DefaultMainConfig
from src.carla.config.scenario_configs import DefaultScenarioConfig
from src.carla.config.observation_configs import EntityObservationConfig
from src.carla.config.action_configs import ThrottleConfig
from src.models.closed_planner import ClosedPlanner


def get_entry_point():
    return 'ModelClosedPlannerAgent'


class ModelClosedPlannerAgent(ModelAgent):
    # Agent for Closed-Planner (Ours)
    def setup(self, path_to_conf_file, route_index=None, cfg=None, exec_or_inter=None):
        AutoPilot.setup(self, path_to_conf_file, route_index, cfg, exec_or_inter)

        self.cfg = cfg
        self.agent_cfg = omegaconf.OmegaConf.load(self.config_path)

        self._scene_featurizer = None

        # agent
        gpu = self.cfg.gpu
        model_cls = ClosedPlanner
        self._fabric = Fabric(accelerator='cuda', precision="16")

        self.model = model_cls.load_from_checkpoint(self.cfg.model_checkpoint, **self.agent_cfg.model, strict=False)

        self.model.cuda(gpu).eval()

        self.config = DefaultMainConfig()
        self.config.populate_config(
            observation_config=EntityObservationConfig(
                camera=self.cfg.save_gif,
                H=self.agent_cfg.H,
                f=self.agent_cfg.f,
                max_route_pts=150,
                max_actors=100,
                max_token_distance=self.cfg.max_token_radius),
            action_config=ThrottleConfig(),
            scenario_config=DefaultScenarioConfig(),
        )

        self._im_save_freq = 20

        self._f = self.config.obs_config.f
        self._history_len = self._f * self.config.obs_config.H + 1
        self._obs_history = []
