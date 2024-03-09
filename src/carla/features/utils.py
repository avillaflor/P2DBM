import numpy as np
import torch


TOWNS = ('Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD')


def process_features(features, masks, ref, max_token_distance=50., zs=None, max_z_distance=7.5, max_actors=np.inf, backwards_bias=1.):
    """
    Filter invalid features w.r.t. range, z, max_actors
    1. If it's a different elevation level, ignore it
    2. If it's out of range (MAX_TOKEN_DISTANCE), ignore it
    3. For the rest, keep the ones closest to the ego
    """
    T = features.shape[0]
    if zs is None:
        same_elevation_as_ego = np.ones(ref[0, None, 2].shape, dtype=bool)
    else:
        same_elevation_as_ego = (np.abs(zs - ref[:, None, 2]) < max_z_distance)
    dist_to_ego = np.linalg.norm(features[:, :, :2] - ref[:, None, :2], axis=-1)
    in_range_of_ego = dist_to_ego < max_token_distance

    valid_mask = same_elevation_as_ego & in_range_of_ego

    num_entities = valid_mask.shape[1]

    if num_entities > max_actors:
        rel_pos = transform_points(torch.Tensor(features[:, :, :2]), torch.Tensor(ref))
        biased_pos_x = torch.where(
            rel_pos[..., 0] >= 0.,
            rel_pos[..., 0],
            backwards_bias * rel_pos[..., 0])
        biased_dist = np.linalg.norm(np.stack([biased_pos_x.numpy(), rel_pos[..., 1]], axis=-1), axis=-1)

        min_dists_to_ego = np.where(valid_mask, biased_dist, np.inf).min(axis=0)
        inds = np.argsort(min_dists_to_ego)[None, :max_actors]
    else:
        inds = np.arange(num_entities)[None]

    valid_features = np.where(valid_mask[..., None], features, 0.)
    valid_masks = np.where(valid_mask, masks, False)

    masked_features = np.zeros((features.shape[0], max_actors, features.shape[-1]), dtype=np.float32)
    masked_masks = np.zeros((masks.shape[0], max_actors), dtype=bool)

    masked_features[:, np.arange(inds.shape[1])] = valid_features[np.arange(T)[:, None], inds]
    masked_masks[:, np.arange(inds.shape[1])] = valid_masks[np.arange(T)[:, None], inds]

    masked_features, masked_masks = torch.tensor(masked_features), torch.tensor(masked_masks)

    masked_features = torch.where(masked_features.isnan(), torch.zeros_like(masked_features), masked_features)

    return masked_features, masked_masks


def transform_points(points, ref, invert=False):
    """
    points: (...,N,3)
    ref: (...,3)
    """
    # transform points into relevant reference frame
    assert points.shape[:-2] == ref.shape[:-1], 'Shape mismatch in transform_points. Got {} and {}'.format(points.shape[:-2], ref.shape[:-1])

    if not invert:
        pos = points[..., :2] - ref[..., None, :2]

        theta = ref[..., -1]
        rot_matrix = torch.stack([
            torch.stack([torch.cos(theta), -torch.sin(theta)], dim=-1),
            torch.stack([torch.sin(theta),  torch.cos(theta)], dim=-1)
        ], dim=-2)[..., None, :, :]
        transformed_pos = torch.matmul(pos[..., None, :2], rot_matrix).squeeze(-2)

        transformed_theta = wrap_angle(points[..., -1:] - ref[..., None, -1:])

        transformed = torch.cat([transformed_pos, points[..., 2:-1], transformed_theta], dim=-1)
    else:
        transformed_theta = wrap_angle(points[..., -1:] + ref[..., None, -1:])

        theta = ref[..., -1]
        rot_matrix = torch.stack([
            torch.stack([ torch.cos(theta), torch.sin(theta)], dim=-1),
            torch.stack([-torch.sin(theta), torch.cos(theta)], dim=-1)
        ], dim=-2)[..., None, :, :]
        pos = torch.matmul(points[..., None, :2], rot_matrix).squeeze(-2)

        transformed_pos = pos + ref[..., None, :2]

        transformed = torch.cat([transformed_pos, points[..., 2:-1], transformed_theta], dim=-1)

    return transformed


def wrap_angle(angle, min_val=-torch.pi, max_val=torch.pi):
    max_min_diff = max_val - min_val
    return min_val + torch.remainder(angle + max_val, max_min_diff)
