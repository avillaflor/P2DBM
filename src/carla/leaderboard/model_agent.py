import os
import carla
import omegaconf
import numpy as np
import torch
from PIL import Image
import lightning.pytorch as pl
from lightning.fabric import Fabric


from src.carla.leaderboard.autopilot import AutoPilot
from src.carla.config.config import DefaultMainConfig
from src.carla.config.scenario_configs import DefaultScenarioConfig
from src.carla.config.observation_configs import EntityObservationConfig
from src.carla.config.action_configs import ThrottleConfig
from src.carla.features.scenarios import Scenario, BatchScenario
from src.carla.features.scene_features import SceneFeaturizer
from src.models.forecasting_model import ForecastingModel


def get_entry_point():
    return 'ModelAgent'


class ModelAgent(AutoPilot):
    # Agent for IL agent
    def setup(self, path_to_conf_file, route_index=None, cfg=None, exec_or_inter=None):
        AutoPilot.setup(self, path_to_conf_file, route_index, cfg, exec_or_inter)

        self.cfg = cfg
        self.agent_cfg = omegaconf.OmegaConf.load(self.config_path)

        self._scene_featurizer = None

        # agent
        gpu = self.cfg.gpu
        model_cls = ForecastingModel
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

    def _init(self, hd_map):
        AutoPilot._init(self, hd_map)
        self._town_map = self._world.get_map()
        self._scene_featurizer = SceneFeaturizer(
            self._vehicle,
            self._world,
            town=self._town_map.name,
            token_radius=self.cfg.max_token_radius,
            route_radius=self.cfg.max_route_radius,
        )

        pl.seed_everything(0)

        self._im_save_dir = os.path.join(self.cfg.data_save_path, self.route_index)
        self._ims = []
        os.makedirs(self._im_save_dir, exist_ok=True)

    def sensors(self):
        result = super().sensors()
        if self.cfg.save_gif:
            result += [{
                'type': 'sensor.camera.rgb',
                'x': self.cfg.camera_pos[0],
                'y': self.cfg.camera_pos[1],
                'z': self.cfg.camera_pos[2],
                'roll': self.cfg.camera_rot_0[0],
                'pitch': self.cfg.camera_rot_0[1],
                'yaw': self.cfg.camera_rot_0[2],
                'width': self.cfg.camera_width,
                'height': self.cfg.camera_height,
                'fov': self.cfg.camera_fov_data_collection,
                'id': 'rgb_front'
            }]
        return result

    def tick(self, input_data):
        result = super().tick(input_data)
        if self.cfg.save_gif:
            if (self.step % self._im_save_freq) == 0:
                im = Image.fromarray(input_data['rgb_front'][1])
                self._ims.append(im)

        if self._scene_featurizer is not None:
            route_wps = []
            veh_loc = self._vehicle.get_location()
            z = veh_loc.z
            for i, pos in enumerate(self.waypoint_route):
                loc = carla.Location(pos[0], pos[1], z)
                if (loc.distance(veh_loc) > self.cfg.max_route_radius) or (i >= 150):
                    break
                wp = self.world_map.get_waypoint(loc)
                route_wps.append(wp)

            goal_loc = carla.Location(self.waypoint_route[-1][0], self.waypoint_route[-1][1], 0.)
            goal_transform = self.world_map.get_waypoint(goal_loc).transform
            scene_graph = self._scene_featurizer.get_scene_graph(route_wps=route_wps, goal_transform=goal_transform)
            result['scene_graph'] = scene_graph

        return result

    def get_action(self, tick_data):
        if len(self._obs_history) < 1:
            self._obs_history = [tick_data]
        else:
            self._obs_history = [*self._obs_history[-(self._history_len-1):], tick_data]
        scenario = Scenario()
        scenario.setup(
            self._obs_history,
            config=self.config,
        )
        batch_scenario = BatchScenario([scenario])
        obses = batch_scenario.get_obs(device=self.model.device, ref_t=-1, f=self._f)

        action = self.model.get_action(obses)

        steer = np.clip(float(action[0]), -1.0, 1.0)
        gas = np.clip(float(action[1]), -1.0, 1.0)
        if gas < 0:
            throttle = 0.0
            brake = 1.
        else:
            throttle = gas
            brake = 0.0

        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
            gear=0)

        return control

    @torch.inference_mode()
    def run_step(self, input_data, timestamp, sensors=None):
        control = super().run_step(input_data, timestamp)

        tick_data = self.tick(input_data)
        if 'scene_graph' in tick_data:
            with self._fabric.autocast():
                control = self.get_action(tick_data)
        return control

    def destroy(self):
        #  pass
        if self.cfg.save_gif and hasattr(self, '_im_save_dir'):
            im_file = os.path.join(self._im_save_dir, 'vid.gif')
            self._ims[0].save(im_file, save_all=True, append_images=self._ims[1:])
