import torch
from torch.utils.data import DataLoader

from common import *
from dataset import *

# ==================================================
#                   Client
# ==================================================

class BaseClient:
    def __init__(self,
        client_id,
        train_set,
        test_set,

        local_epochs=1,
        batch_size=64,
        lr=0.05,
        momentum=0.9,

        criterion='ce',
        optimizer='sgd',
        scheduler='steplr',
    ):
        self.client_id = client_id
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum

        self.criterion_name = criterion.lower()
        self.optimizer_name = optimizer.lower()
        self.scheduler_name = scheduler.lower()

        self.hardware = find_device()
        self.record = Record()

        self.server = None  # used in SL

    def set_model(self, model):
        self.model = model
        self.model.to(self.hardware)

        self.optimizer = find_optimizer(self.model, self.optimizer_name, self.lr, self.momentum)
        self.scheduler = find_scheduler(self.optimizer, self.scheduler_name)
        self.criterion = find_criterion(self.criterion_name)

    def load_state(self, state):
        self.model.load_state_dict(state)

    def set_server(self, server):
        self.server = server

    def lr_step(self, test_acc):
        if self.scheduler_name == "steplr":
            self.scheduler.step()
        elif self.scheduler_name == 'reducelronplateau':
            self.scheduler.step(test_acc)
        else:
            raise NotImplementedError

    def train_epoch(self):
        raise NotImplementedError

    def test_epoch(self):
        raise NotImplementedError

    def train_test_local(self):
        raise NotImplementedError


def train_client(client: BaseClient):
    print(f'  [{client.client_id}] >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    train_loss, train_acc, test_loss, test_acc, comm_cost = client.train_test_local()
    print(f'  [{client.client_id}] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    res = {
        'id': client.client_id,
        'size' : get_model_size(client.model),
        'state' : client.model.state_dict(),
        'train_loss' : train_loss,
        'train_acc' : train_acc,
        'test_loss' : test_loss,
        'test_acc' : test_acc,
        'comm_cost' : comm_cost
    }
    return res