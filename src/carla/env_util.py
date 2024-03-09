import carla
import numpy as np
import torch


def to(x, device):
    if isinstance(x, dict):
        new_x = {}
        for k in x:
            new_x[k] = to(x[k], device)
    elif isinstance(x, list):
        new_x = []
        for i in range(len(x)):
            new_x.append(to(x[i], device))
    elif isinstance(x, torch.Tensor):
        new_x = x.to(device=device)
    else:
        new_x = x
    return new_x


def get_speed_from_velocity(velocity, convert=True):
    if isinstance(velocity, carla.Vector3D):
        speed = np.sqrt(velocity.x ** 2 + velocity.y **2 + velocity.z **2)
    elif isinstance(velocity, np.ndarray):
        speed = np.linalg.norm(velocity)
    else:
        raise NotImplementedError
    if convert:
        return 3.6 * speed
    else:
        return speed


def get_speed(vehicle, convert=True):
    vel = vehicle.get_velocity()
    return get_speed_from_velocity(vel, convert=convert)


def get_acc(vehicle, convert=True):
    acc = vehicle.get_acceleration()
    return get_speed_from_velocity(acc, convert=convert)
