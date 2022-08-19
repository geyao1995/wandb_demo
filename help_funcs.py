import random
import warnings
from torch.optim.lr_scheduler import CyclicLR, MultiStepLR

import numpy as np
import torch


def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    # elif torch.backends.mps.is_available():
    #     # https://pytorch.org/docs/master/notes/mps.html
    #     device = torch.device('mps')
    else:
        device = torch.device('cpu')
        warnings.warn("You are using CPU!")

    return device


def get_lr_scheduler(optimizer, schedule: str,
                     steps: int | float, lr_min: float, lr_max: float):
    if schedule == 'step':
        milestones = [int(steps / 2), int(steps * 3 / 4)]
        gamma = (lr_min / lr_max) ** (1 / len(milestones))
        lr_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif schedule == 'cyclic':
        step_size_up = int(steps * 2 / 5)
        step_size_down = int(steps * 3 / 5)
        lr_scheduler = CyclicLR(optimizer, base_lr=lr_min, max_lr=lr_max,
                                step_size_up=step_size_up, step_size_down=step_size_down)
    else:
        raise NotImplementedError

    return lr_scheduler
