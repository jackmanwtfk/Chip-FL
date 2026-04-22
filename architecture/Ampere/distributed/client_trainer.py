import socket

from torch.utils.data import Subset

from common import *
from dataset import *
from network import sendall, recvall

from ...base import BaseDistributedTrainer
from ..client import AmpereClient




class AmpereClientTrainer(BaseDistributedTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)

    def pretrain(self):
        self.sock.connect(self.host_addr)

        rsp, _ = recvall(self.sock)
        self.cid = rsp['id']
        indices = rsp['indices']

        self.prepare_dm()
        full_train_set, test_set = self.dm.get_datasets()
        train_set = Subset(full_train_set, indices)

        self.client = AmpereClient(
            self.cid,
            train_set,
            test_set,
            local_epochs=self.local_epochs,
            batch_size=self.batch_size,
            momentum=self.momentum,
            lr=self.lr
        )

        cmodel = self.create_model(is_client=True)
        self.client.set_model(cmodel)
        print(f'  {self.cid} connected')

    def train_epoch(self, epoch):
        for local_epoch in range(self.local_epochs):
            train_loss, train_acc = self.client.train_epoch()
            test_loss, test_acc = self.client.test_epoch()
            self.client.lr_step(test_acc)
            self.client.record.add(train_loss, train_acc, test_loss, test_acc)
            
            print(
                f"  [{self.cid}] ({local_epoch+1}/{self.local_epochs}) "
                f"Train: {train_loss:.4f}/{train_acc:.2f}%, "
                f"Test: {test_loss:.4f}/{test_acc:.2f}%"
            )

        data = {
            'state' : self.client.model.state_dict()
        }
        sendall(self.sock, data)
        print(f'  [{self.cid}] local training done, sending state')
        print()

    def update_state(self):
        agg_state, _ = recvall(self.sock)
        self.client.model.load_state_dict(agg_state)
        print(f'  [{self.cid}] updated')

        train_smashed, train_labels, test_smashed, test_labels =\
            self.client.gen_activations()

        for smashed, labels in zip(train_smashed, train_labels):
            data = {
                'action'  : 'train',
                'smashed' : smashed,
                'labels'  : labels
            }
            sendall(self.sock, data)

        for smashed, labels in zip(test_smashed, test_labels):
            data = {
                'action'  : 'test',
                'smashed' : smashed,
                'labels'  : labels,
            }
            sendall(self.sock, data)

        data = {
            'action' : 'finished'
        }
        sendall(self.sock, data)
        print(f'  [{self.cid}] sent unified activations')

    def posttrain(self):
        self.sock.close()