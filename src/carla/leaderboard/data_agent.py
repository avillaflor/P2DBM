import torch
import numpy as np
import os
import json
import carla


from src.carla.leaderboard.autopilot import AutoPilot
from src.carla.features.scene_features import SceneFeaturizer


def get_entry_point():
    return 'DataAgent'


def to_list(d):
    if isinstance(d, dict):
        res = {}
        for k in d:
            res[k] = to_list(d[k])
        return res
    elif isinstance(d, np.ndarray):
        return d.tolist()
    else:
        return d


class DataAgent(AutoPilot):
    # Agent for data collection
    def setup(self, path_to_conf_file, route_index=None, cfg=None, exec_or_inter=None):
        super().setup(path_to_conf_file, route_index, cfg, exec_or_inter)

        self.cfg = cfg

        self._scene_featurizer = None

        self.save_freq = 10

    def _init(self, hd_map):
        super()._init(hd_map)
        self._town_map = self._world.get_map()
        self._scene_featurizer = SceneFeaturizer(
            self._vehicle,
            self._world,
            town=self._town_map.name,
            token_radius=self.cfg.max_token_radius,
            route_radius=self.cfg.max_route_radius,
        )

    def sensors(self):
        result = super().sensors()
        return result

    def tick(self, input_data):
        result = super().tick(input_data)

        if self._scene_featurizer is not None:
            route_wps = []
            veh_loc = self._vehicle.get_location()
            z = veh_loc.z
            for pos in self.waypoint_route:
                loc = carla.Location(pos[0], pos[1], z)
                if loc.distance(veh_loc) > self.cfg.max_route_radius:
                    break
                wp = self.world_map.get_waypoint(loc)
                route_wps.append(wp)

            goal_loc = carla.Location(self.waypoint_route[-1][0], self.waypoint_route[-1][1], 0.)
            goal_transform = self.world_map.get_waypoint(goal_loc).transform
            scene_graph = self._scene_featurizer.get_scene_graph(route_wps=route_wps, goal_transform=goal_transform)
            result['scene_graph'] = scene_graph

        return result

    def save_data(self, data, control):
        steer = control.steer
        throttle = control.throttle
        brake = control.brake
        if throttle >= brake:
            acc = throttle
        else:
            acc = -brake
        action = np.array([steer, acc])
        data['steer'] = steer
        data['throttle'] = throttle
        data['brake'] = brake
        data['action'] = action

        measurements_path = os.path.join(self.save_path, 'measurements', '{:04d}.json'.format(self.step))
        with open(measurements_path, 'w') as out:
            json.dump(to_list(data), out)

    @torch.inference_mode()
    def run_step(self, input_data, timestamp, sensors=None):
        control = super().run_step(input_data, timestamp)

        if self.step % self.save_freq == 0:
            if self.save_path is not None:
                tick_data = self.tick(input_data)
                if 'scene_graph' in tick_data:
                    self.save_data(tick_data, control)

        return control

    def destroy(self):
        pass
