import socket

from torch.utils.data import Subset

from common import *
from dataset import *
from model import LeNet12
from network import sendall, recvall

from ...base import BaseDistributedTrainer
from .client import DistributeSplitClient




class SplitClientTrainer(BaseDistributedTrainer):
    def __init__(self, *args, **kwagrs):
        super().__init__(*args, **kwagrs)

        self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)

    def pretrain(self):
        self.sock.connect(self.host_addr)

        rsp, _ = recvall(self.sock)
        self.cid = rsp['id']
        indices = rsp['indices']

        self.prepare_dm()
        full_train_set, test_set = self.dm.get_datasets()
        train_set = Subset(full_train_set, indices)

        self.client = DistributeSplitClient(
            self.cid,
            train_set,
            test_set,
            local_epochs=self.local_epochs,
            batch_size=self.batch_size,
            momentum=self.momentum,
            lr=self.lr,
        )

        cmodel, _ = self.create_model().split_model(self.cut_layer)
        self.client.set_model(cmodel)
        print(f'  {self.cid} connected')

    def train_epoch(self, epoch):
        agg_train_loss, agg_train_acc = 0, 0.0
        agg_test_loss, agg_test_acc = 0, 0.0
        agg_comm_cost = 0.0


        for local_epoch in range(self.local_epochs):
            train_loss, train_acc, comm_cost = self.client.train_epoch(self.sock)
            test_loss, test_acc = self.client.test_epoch(self.sock)
            self.client.lr_step(test_acc)

            data = {
                'action'     : 'finish',
                'train_loss' : train_loss,
                'train_acc'  : train_acc,
                'test_loss'  : test_loss,
                'test_acc'   : test_acc,
            }
            sendall(self.sock, data)

            self.client.record.add(train_loss, train_acc, test_loss, test_acc, comm_cost)
            agg_train_loss += train_loss; agg_train_acc += train_acc
            agg_test_loss += test_loss; agg_test_acc += test_acc
            agg_comm_cost += comm_cost
            print(
                f"  [{self.cid}] ({local_epoch+1}/{self.local_epochs}) "
                f"Train: {train_loss:.4f}/{train_acc:.2f}%, "
                f"Test: {test_loss:.4f}/{test_acc:.2f}%"
            )

        state = self.client.model.state_dict()
        data = {
            'state' : state
        }
        sendall(self.sock, data)
        print(f'  [{self.cid}] local training done, sending state')
        print('')

    def update_state(self):
        agg_state, _ = recvall(self.sock)
        self.client.model.load_state_dict(agg_state)
        print(f'  [{self.cid}] updated')


    def posttrain(self):
        self.sock.close()
            


