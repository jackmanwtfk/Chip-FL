import itertools
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .constants import *

# ==================================================
#                       log
# ==================================================

def log_model(m):
    " print model information"
    logging.info(m)

    total_params = sum(p.numel() for p in m.parameters())
    logging.info(f"Total parameters: {total_params/1e6:.2f} Million")

# ==================================================
#                   find
# ==================================================

def find_device():
    # Prefer CUDA if available
    try:
        if torch.cuda.is_available():
            return torch.device('cuda')
    except Exception:
        pass

    # MPS (Apple Metal) support may not exist on all builds/platforms
    try:
        mps_is_avail = False
        if hasattr(torch, 'mps') and callable(getattr(torch.mps, 'is_available', None)):
            mps_is_avail = torch.mps.is_available()
        elif getattr(torch, 'has_mps', False):
            # newer torch versions may expose `has_mps`
            mps_is_avail = bool(torch.has_mps)

        if mps_is_avail:
            return torch.device('mps')
    except Exception:
        pass

    # Fallback to CPU
    return torch.device('cpu')

def find_criterion(name):
    if name == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        raise UserWarning(f"Given criterion {name} is not supported.")
    return criterion

def find_optimizer(model, name, lr:float, momentum:float):
    if name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise UserWarning(f"Given optimizer {name} is not supported.")
    return optimizer

def find_scheduler(optimizer, name):
    if name == 'steplr':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif name == 'reducelronplateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )
    else:
        raise UserWarning(f"Give LR Scheduler {name} is not supported!")
    return scheduler
    
# ==================================================
#                  model stats
# ==================================================

def get_param_size(parameters):
    param_size = 0
    for param in parameters:
        param_size += param.nelement() * param.element_size()

    return param_size

def get_model_size(model):
    parameters = model.parameters()
    return get_param_size(parameters)


# ==================================================
#                  model stats
# ==================================================

def get_random_batches(dataloader: DataLoader, num_batches=8):
    dataset = dataloader.dataset
    batch_size = dataloader.batch_size
    temp_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # 启用随机
        num_workers=0,  # 为避免复杂问题，设为0
        pin_memory=False
    )
    return list(itertools.islice(temp_dataloader, num_batches))
