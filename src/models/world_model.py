import torch


from src.carla.features.utils import transform_points


class WorldModel:
    # World model on our feature set that is used for closed-loop forecasting and evaluation in planners
    def __init__(
            self,
            dt,
            max_token_distance=50.0,
            speed_coeff=1.0,
            coll_coeff=20.0,
            route_coeff=0.1,
            red_speed_coeff=4.0,
    ):
        self.dt = dt
        self.max_token_distance = max_token_distance
        self.speed_coeff = speed_coeff
        self.coll_coeff = coll_coeff
        self.route_coeff = route_coeff
        self.red_speed_coeff = red_speed_coeff

    def get_features(self, obs, wp_diffs, new_walkers=None, new_lights=None):
        new_obs = {}

        B, _, N, d = obs['vehicle_features'].shape
        curr_vehicles = obs['vehicle_features'][:, -1].view((B, N, d))

        next_pos_ori = transform_points(
            wp_diffs[..., :3].unsqueeze(-2),
            curr_vehicles[..., :3],
            invert=True).view((-1, N, 3))
        next_speed = wp_diffs[..., 3:4] + curr_vehicles[..., 3:4]
        next_vehicle_features = torch.cat([next_pos_ori, next_speed, curr_vehicles[..., 4:]], dim=-1)

        new_obs['vehicle_features'] = torch.cat([obs['vehicle_features'][:, 1:], next_vehicle_features.unsqueeze(1)], dim=1)
        new_obs['vehicle_masks'] = obs['vehicle_masks'].clone()

        if new_walkers is None:
            walker_thetas = obs['walker_features'][..., 2]
            walker_speeds = obs['walker_features'][..., 3]
            next_walker_pos = obs['walker_features'][..., :2] + self.dt * walker_speeds.unsqueeze(-1) * torch.stack([torch.cos(walker_thetas), torch.sin(walker_thetas)], dim=-1)
            new_obs['walker_features'] = torch.cat([next_walker_pos, obs['walker_features'][..., 2:]], dim=-1)
            new_obs['walker_masks'] = obs['walker_masks'].clone()
        else:
            new_obs['walker_features'] = new_walkers.clone()
            new_obs['walker_masks'] = obs['walker_masks'].clone()

        if new_lights is None:
            new_obs['light_features'] = obs['light_features'].clone()
            new_obs['light_masks'] = obs['light_masks'].clone()
        else:
            new_obs['light_features'] = new_lights.clone()
            new_obs['light_masks'] = obs['light_masks'].clone()

        new_obs['stop_features'] = obs['stop_features'].clone()
        new_obs['stop_masks'] = obs['stop_masks'].clone()

        new_obs['route_features'] = obs['route_features'].clone()
        new_obs['route_masks'] = obs['route_masks'].clone()

        new_obs['ref'] = obs['ref'].clone()
        new_obs['town'] = obs['town'].copy()

        return new_obs

    def get_reward(self, obs, new_obs, dones=None, is_reds=None):
        B, H, N, d = new_obs['vehicle_features'].shape
        speed = new_obs['vehicle_features'][:, -1, 0, 3]
        speed_limit = torch.clamp(new_obs['vehicle_features'][:, -1, 0, 6], min=(1. / self.max_token_distance))

        speed_pen = 1. - (abs(speed - speed_limit) / speed_limit)

        # > 0.5 because want a bit of buffer room
        #  bbox_factor = 0.5
        bbox_factor = 0.5
        ego = new_obs['vehicle_features'][:, -1, 0, :3]
        ego_bboxes = bbox_factor * new_obs['vehicle_features'][:, -1, 0, 4:6]

        if N > 1:
            others = new_obs['vehicle_features'][:, -1, 1:, :3]
            other_bboxes = bbox_factor * new_obs['vehicle_features'][:, -1, 1:, 4:6]
            colls = check_collisions(others, other_bboxes, ego, ego_bboxes)
            colls = colls & new_obs['vehicle_masks'][:, -1, 1:]
            colls = colls.any(dim=-1)
        else:
            colls = torch.zeros((B,), dtype=bool, device=speed.device)

        walker_N = new_obs['walker_features'].shape[2]
        if walker_N > 0:
            walkers = new_obs['walker_features'][:, -1, :, :3]
            walker_bboxes = bbox_factor * new_obs['walker_features'][:, -1, :, 4:6]
            walker_colls = check_collisions(walkers, walker_bboxes, ego, ego_bboxes)
            walker_colls = walker_colls & new_obs['walker_masks'][:, -1]
            walker_colls = walker_colls.any(dim=-1)
            colls = colls | walker_colls

        if dones is None:
            dones = colls
        else:
            dones = dones | colls

        coll_pen = -1. * dones.float()

        route_pts = new_obs['route_features'][..., :2]
        route_dists = torch.where(
            new_obs['route_masks'],
            torch.norm(route_pts - new_obs['vehicle_features'][:, -1, 0, :2].unsqueeze(1), dim=-1),
            torch.inf)

        _, closest_inds = route_dists.topk(2, dim=-1, largest=False)

        closest_ind = closest_inds[..., 0]
        other_closest_ind = closest_inds[..., 1]

        close_pt = route_pts[torch.arange(B), closest_ind]
        other_pt = route_pts[torch.arange(B), other_closest_ind]

        a = other_pt - close_pt
        b = close_pt - new_obs['vehicle_features'][:, -1, 0, :2]
        traj_dist = (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]) / (torch.norm(a, dim=-1) + 1.e-6)

        lane_width = 2. / self.max_token_distance
        route_diff = torch.abs(traj_dist) / lane_width
        route_diff = torch.where(
            new_obs['route_masks'].sum(dim=-1) > 1,
            route_diff,
            0.)
        route_pen = 1. - route_diff

        light_N = obs['light_features'].shape[2]
        if light_N > 0:
            light_bbox_factor = 0.5
            lights = torch.stack([obs['light_features'][:, -1, :, :3], new_obs['light_features'][:, -1, :, :3]], dim=1)
            light_bboxes = light_bbox_factor * torch.stack([obs['light_features'][:, -1, :, 3:5], new_obs['light_features'][:, -1, :, 3:5]], dim=1)
            light_masks = torch.stack([obs['light_masks'][:, -1], new_obs['light_masks'][:, -1]], dim=1)
            # red is 5, yellow is 6
            is_lit = torch.stack([obs['light_features'][:, -1, :, 5:7], new_obs['light_features'][:, -1, :, 5:7]], dim=1).bool().any(dim=-1)

            light_ego_bboxes = bbox_factor * torch.stack([obs['vehicle_features'][:, -1, 0, 4:6], new_obs['vehicle_features'][:, -1, 0, 4:6]], dim=1)
            light_ego_bboxes = light_ego_bboxes.clone()
            # from planT
            light_ego_bboxes[..., 0] = 4.5 / self.max_token_distance
            light_ego_bboxes[..., 1] = 1.5 / self.max_token_distance
            lights = lights.clone()

            light_ego = torch.stack([obs['vehicle_features'][:, -1, 0, :3], new_obs['vehicle_features'][:, -1, 0, :3]], dim=1)

            light_intersect = check_collisions(lights, light_bboxes, light_ego, light_ego_bboxes)
            all_is_reds = torch.any(light_intersect & light_masks & is_lit, dim=-1)

            if is_reds is None:
                is_reds = all_is_reds.any(dim=1)
            else:
                is_reds = is_reds | all_is_reds.any(dim=1)

            red_speed_pen = -(torch.abs(speed) / speed_limit) * is_reds.float()

        reward = self.speed_coeff * speed_pen + self.coll_coeff * coll_pen + self.route_coeff * route_pen + self.red_speed_coeff * red_speed_pen

        return reward, dones, is_reds

    def step(self, obs, wp_diffs, new_walkers=None, new_lights=None, dones=None, is_reds=None):
        new_obs = self.get_features(obs, wp_diffs, new_walkers=new_walkers, new_lights=new_lights)
        reward, dones, is_reds = self.get_reward(obs, new_obs, dones=dones, is_reds=is_reds)
        return new_obs, reward, dones, is_reds


def do_bboxes_intersect(other_bboxes, ego_bbox):
    # other_bboxes B, N, 4, 2
    # ego_bbox B, 1, 4, 2
    B, N = other_bboxes.shape[:2]

    other_normals = []
    ego_normals = []
    for i in range(4):
        for j in range(i, 4):
            other_p1 = other_bboxes[..., i, :]
            other_p2 = other_bboxes[..., j, :]
            other_normals.append(torch.stack([other_p2[..., 1] - other_p1[..., 1], other_p1[..., 0] - other_p2[..., 0]], dim=-1))

            ego_p1 = ego_bbox[..., i, :]
            ego_p2 = ego_bbox[..., j, :]
            ego_normals.append(torch.stack([ego_p2[..., 1] - ego_p1[..., 1], ego_p1[..., 0] - ego_p2[..., 0]], dim=-1))

    # other_normals B, N, 10, 2
    other_normals = torch.stack(other_normals, dim=-2)
    # ego_normals B, 1, 10, 2
    ego_normals = torch.stack(ego_normals, dim=-2)

    # a_other_projects (B, N, 10, 4)
    a_other_projects = torch.sum(other_normals.view((B, N, 10, 1, 2)) * other_bboxes.view((B, N, 1, 4, 2)), dim=-1)
    # b_other_projects (B, N, 10, 4)
    b_other_projects = torch.sum(other_normals.view((B, N, 10, 1, 2)) * ego_bbox.view((B, 1, 1, 4, 2)), dim=-1)
    # other_separates (B, N, 10)
    other_separates = (a_other_projects.max(dim=-1)[0] < b_other_projects.min(dim=-1)[0]) | (b_other_projects.max(dim=-1)[0] < b_other_projects.min(dim=-1)[0])

    # a_ego_projects (B, 1, 10, 4)
    a_ego_projects = torch.sum(ego_bbox.view((B, 1, 1, 4, 2)) * ego_normals.view((B, 1, 10, 1, 2)), dim=-1)

    # b_ego_projects (B, N, 10, 4)
    b_ego_projects = torch.sum(other_bboxes.view((B, N, 1, 4, 2)) * ego_normals.view((B, 1, 10, 1, 2)), dim=-1)

    # ego_separates (B, N, 10)
    ego_separates = (a_ego_projects.max(dim=-1)[0] < b_ego_projects.min(dim=-1)[0]) | (b_ego_projects.max(dim=-1)[0] < b_ego_projects.min(dim=-1)[0])

    # separates (B, N)
    separates = other_separates.any(dim=-1) | ego_separates.any(dim=-1)

    return ~separates


def check_collisions(P2, extent2, P1, extent1):
    # P1, P2: (..., 1x3), (..., Nx3) object poses
    # Returns (..., N) collision mask (True if collision)
    N = P2.shape[-2]
    bboxes = torch.stack([
        torch.stack([-extent2[..., 0], -extent2[..., 1]], dim=-1),
        torch.stack([extent2[..., 0], -extent2[..., 1]], dim=-1),
        torch.stack([extent2[..., 0], extent2[..., 1]], dim=-1),
        torch.stack([-extent2[..., 0], extent2[..., 1]], dim=-1),
    ], dim=-2)

    theta2 = P2[..., 2]
    rot_matrix2 = torch.stack([
        torch.stack([torch.cos(theta2), torch.sin(theta2)], dim=-1),
        torch.stack([-torch.sin(theta2), torch.cos(theta2)], dim=-1)
    ], dim=-2)
    bboxes = (bboxes @ rot_matrix2)
    bboxes += P2[..., None, :2]

    ego_bboxes = torch.stack([
        torch.stack([-extent1[..., 0], -extent1[..., 1]], dim=-1),
        torch.stack([extent1[..., 0], -extent1[..., 1]], dim=-1),
        torch.stack([extent1[..., 0], extent1[..., 1]], dim=-1),
        torch.stack([-extent1[..., 0], extent1[..., 1]], dim=-1),
    ], dim=-2)
    ego_theta = P1[..., 2]
    ego_rot_matrix = torch.stack([
        torch.stack([torch.cos(ego_theta), torch.sin(ego_theta)], dim=-1),
        torch.stack([-torch.sin(ego_theta), torch.cos(ego_theta)], dim=-1)
    ], dim=-2)
    ego_bboxes = (ego_bboxes @ ego_rot_matrix)
    ego_bboxes += P1[..., None, :2]

    bboxes_shape = P2.shape[:-1]
    shaped_is_collision = do_bboxes_intersect(bboxes.view((-1, N, 4, 2)), ego_bboxes.view((-1, 1, 4, 2)))
    is_collision = shaped_is_collision.view(bboxes_shape)

    return is_collision
