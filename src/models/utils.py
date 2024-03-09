import math
import numpy as np
import torch
from torch import nn
from collections import deque
from torch.distributions import Laplace, Normal
from torch.optim.lr_scheduler import LambdaLR


class TimeEncoding(nn.Module):
    # Standard positional encoding.
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, start_t=0):
        '''
        :param x: must be (B, H, d)
        :return:
        '''
        x = x + self.pe[:, start_t:start_t+x.shape[1], :]
        return self.dropout(x)


"""
Copied from https://github.com/roggirg/AutoBots/blob/master/utils/train_helpers.py
"""
def get_laplace_dist(pred):
    d = pred.shape[-1] // 2
    return Laplace(pred[..., :d], pred[..., d:2*d])


def get_normal_dist(pred):
    d = pred.shape[-1] // 2
    return Normal(pred[..., :d], pred[..., d:2*d])


def nll_pytorch_dist(pred, data, dist='laplace'):
    if dist == 'laplace':
        biv_lapl_dist = get_laplace_dist(pred)
    elif dist == 'normal':
        biv_lapl_dist = get_normal_dist(pred)
    else:
        raise NotImplementedError

    return -biv_lapl_dist.log_prob(data).sum(dim=-1)


"""
Copied from HuggingFace: https://github.com/huggingface/transformers/blob/v4.19.4/src/transformers/optimization.py#L75
"""
def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1, final_factor=0.0,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * (1. - final_factor) + final_factor

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class PIDController(object):
    # Basic PID Controller
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative
