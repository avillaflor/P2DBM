import carla
import numpy as np
import torch
import os


from src.carla.features.scenarios import transform_points


class CarlaMapFeatures:
    def __init__(
            self,
            map_name,
            world_map=None,
            map_data_path=None,
            map_wps=None,
            map_xodr_path=None,
            precision=5.0,
            max_obs_distance=50.,
            max_token_distance=50.,
            max_z_distance=7.5,
            max_map_pts=None,
            torch_device=None,
    ):
        self._map_name = map_name
        self._precision = precision
        self._max_obs_distance = max_obs_distance
        self._max_token_distance = max_token_distance
        self._max_z_distance = max_z_distance
        self._max_map_pts = max_map_pts
        self._torch_device = torch_device
        if map_data_path is not None:
            map_data_file = os.path.join(map_data_path, '{0}.npy'.format(self._map_name))
            self._map_data = np.load(map_data_file)
        else:
            if map_wps is not None:
                wps = map_wps
            else:
                if map_xodr_path is not None:
                    map_xodr_file = os.path.join(map_xodr_path, '{0}.xodr'.format(self._map_name))
                    with open(map_xodr_file, 'r') as f:
                        map_str = f.read()
                    world_map = carla.Map(map_name, map_str)
                elif world_map is None:
                    raise NotImplementedError
                wps = world_map.generate_waypoints(precision)

            map_data = []
            for wp in wps:
                map_location = wp.transform.location
                map_rotation = wp.transform.rotation
                right_lc = wp.lane_change in (carla.libcarla.LaneChange.Both, carla.libcarla.LaneChange.Right)
                left_lc = wp.lane_change in (carla.libcarla.LaneChange.Both, carla.libcarla.LaneChange.Left)

                theta = np.radians(map_rotation.yaw)
                next_wps = wp.next(precision)
                next_wp = next_wps[0] if len(next_wps) > 0 else None
                prev_wps = wp.previous(precision)
                prev_wp = prev_wps[0] if len(prev_wps) > 0 else None
                if next_wp is not None:
                    x_diff = next_wp.transform.location.x - wp.transform.location.x
                    y_diff = next_wp.transform.location.y - wp.transform.location.y
                    theta = np.arctan2(y_diff, x_diff)
                elif prev_wp is not None:
                    x_diff = wp.transform.location.x - prev_wp.transform.location.x
                    y_diff = wp.transform.location.y - prev_wp.transform.location.y
                    theta = np.arctan2(y_diff, x_diff)

                data = np.array([
                    map_location.x,
                    map_location.y,
                    map_location.z,
                    theta,
                    wp.lane_width,
                    wp.is_junction,
                    right_lc,
                    left_lc,
                    wp.road_id,
                    wp.section_id,
                    wp.lane_id,
                    wp.s,
                ])
                map_data.append(data)
            self._map_data = np.stack(map_data, axis=0)
        if self._torch_device is not None:
            self._torch_map_data = torch.Tensor(self._map_data).to(device=self._torch_device)

    def get_map_features(self, ego_location):
        map_features = {}

        dists = np.linalg.norm(ego_location[:2] - self._map_data[:, :2], axis=-1)
        z_dists = abs(ego_location[2] - self._map_data[:, 2])

        idxs = np.where((dists < self._max_obs_distance) & (z_dists < self._max_z_distance))[0]

        for i, idx in enumerate(idxs):
            data = self._map_data[idx]
            map_features[i] = {
                'x': data[0],
                'y': data[1],
                'z': data[2],
                'theta': data[3],
                'lane_width': data[4],
                'is_junction': data[5],
                'right_lc': data[6],
                'left_lc': data[7],
                'road_id': data[8],
                'section_id': data[9],
                'lane_id': data[10],
                's': data[11],
            }

        return map_features

    def get_model_features(self, refs):
        B = refs.shape[0]
        if refs.device != self._torch_device:
            self._torch_device = refs.device
            self._torch_map_data = self._torch_map_data.to(device=self._torch_device)

        base_refs = refs

        # get closest map pts
        dists = torch.norm(base_refs[:, None, :2] - self._torch_map_data[None, :, :2], dim=-1)
        z_dists = abs(base_refs[:, None, 2] - self._torch_map_data[None, :, 2])
        masks = (dists < self._max_obs_distance) & (z_dists < self._max_z_distance)
        dists = torch.where(
            masks,
            dists,
            torch.ones_like(dists) * torch.inf)

        sorted_idxs = torch.argsort(dists, dim=-1)
        sorted_masks = masks[torch.arange(B, device=sorted_idxs.device).unsqueeze(1), sorted_idxs]
        sorted_map_features = self._torch_map_data[sorted_idxs]
        if self._max_map_pts is not None:
            sorted_masks = sorted_masks[:, :self._max_map_pts]
            sorted_masks = torch.cat([sorted_masks, torch.zeros((B, self._max_map_pts - sorted_masks.shape[1]), dtype=sorted_masks.dtype, device=sorted_masks.device)], dim=1)
            sorted_map_features = sorted_map_features[:, :self._max_map_pts]
            sorted_map_features = torch.cat([sorted_map_features, torch.zeros((B, self._max_map_pts - sorted_map_features.shape[1], 12), dtype=sorted_map_features.dtype, device=sorted_map_features.device)], dim=1)

        map_features = sorted_map_features[:, :, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]].clone()
        map_masks = sorted_masks.clone()

        # normalize features
        map_pos_ori = transform_points(map_features[..., :3], refs)
        map_features = torch.cat([
            map_pos_ori[..., :2] / self._max_token_distance,
            map_pos_ori[..., 2:3],
            map_features[..., 3:4] / self._max_token_distance,
            map_features[..., 4:],
        ], dim=-1)

        map_features = torch.where(
            map_masks.unsqueeze(-1),
            map_features,
            torch.zeros_like(map_features))

        return map_features, map_masks

    def save_data(self, save_path):
        save_file = os.path.join(save_path, '{0}.npy'.format(self._map_name))
        np.save(save_file, self._map_data)
