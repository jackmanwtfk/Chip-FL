import socket

from torch.utils.data import Subset

from common import *
from dataset import *
from model import LeNet12
from network import sendall, recvall

from ...base import BaseDistributedTrainer, train_client
from ..client import FedClient


class FedClientTrainer(BaseDistributedTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)

    # ==================================================
    #                   pretrain
    # ==================================================

    def pretrain(self):
        self.sock.connect(self.host_addr)
        msg, _ = recvall(self.sock)
        self.client_id = msg['id']
        indices = msg['indices']

        self.prepare_dm()
        full_train_set, test_set = self.dm.get_datasets()
        train_set = Subset(full_train_set, indices)

        self.client = FedClient(
            self.client_id,
            train_set,
            test_set,
            batch_size=self.batch_size,
            momentum=self.momentum,
            lr=self.lr,
        )

        model = self.create_model()
        self.client.set_model(model)
        print(f'  {self.client_id} pretain')

    # ==================================================
    #                   train
    # ==================================================

    def train_epoch(self, epoch):
        result = train_client(self.client)
        data = {
            'train_loss' : result['train_loss'],
            'train_acc'  : result['train_acc'],
            'test_loss'  : result['test_loss'],
            'test_acc'   : result['test_acc'],
            'state'      : result['state'],
        }
        sendall(self.sock, data)
        # print(f'  [{self.client_id}] send states')

    def update_state(self):
        agg_state, _ = recvall(self.sock)
        self.client.model.load_state_dict(agg_state)
        print(f'  [{self.client_id}] update states')

    # ==================================================
    #                   posttrain
    # ==================================================

    def posttrain(self):
        self.sock.close()
