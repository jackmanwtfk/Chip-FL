import copy

import torch

from common import *
from dataset import *


# ==================================================
#                   Server
# ==================================================

class BaseServer:
    def __init__(self,
        batch_size=64,
        lr=0.05,
        momentum=0.9,

        criterion="ce",
        optimizer="sgd",
        scheduler="steplr",
    ):
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum

        self.criterion_name = criterion.lower()
        self.optimizer_name = optimizer.lower()
        self.scheduler_name = scheduler.lower()

        self.hardware = find_device()
        self.record = Record()

    def set_model(self, model):
        self.model = model
        self.model.to(self.hardware)

        self.optimizer = find_optimizer(self.model, self.optimizer_name, self.lr, self.momentum)
        self.scheduler = find_scheduler(self.optimizer, self.scheduler_name)
        self.criterion = find_criterion(self.criterion_name)

    def lr_step(self, test_acc):
        if self.scheduler_name == "steplr":
            self.scheduler.step()
        elif self.scheduler_name == 'reducelronplateau':
            self.scheduler.step(test_acc)
        else:
            raise NotImplementedError

    def fedavg(self, w):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))

        return w_avg
    
    def distribute(self, clients, state):
        for client in clients:
            client.load_state(state)

    def train_batch(self, smashed, labels):
        raise NotImplementedError
    
    def test_batch(self, smahed, labels):
        raise NotImplementedError
