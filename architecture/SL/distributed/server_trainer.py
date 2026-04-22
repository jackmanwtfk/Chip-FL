import os
import json
import socket
import datetime

import torch

from common import *
from dataset import *
from network import *
from model import LeNet12

from ...base import BaseDistributedTrainer
from .server import DistributeSplitServer

host_ip = 'localhost'
host_port = 9000




class SplitServerTrainer(BaseDistributedTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clients = {}
        self.server = None
        self.model_states = None

        self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.sock.bind(self.host_addr)
        self.sock.listen(5)

    def pretrain(self):
        print('----- pretrain -----')
        self.server = DistributeSplitServer(batch_size=self.batch_size, lr=self.lr, momentum=self.momentum)
        _, smodel = self.create_model().split_model(self.cut_layer)
        self.server.set_model(smodel)

        self.prepare_dm()
        full_train_set, self.test_set = self.dm.get_datasets()
        self.client_datasets = split_dataset(full_train_set, self.num_clients)

        print('listening for clients ...')
        for idx, dataset in zip(range(self.num_clients), self.client_datasets):
            conn, addr =  self.sock.accept()
            data = {
                'id': idx,
                'indices': dataset.indices
            }
            sendall(conn, data)
            self.clients[idx] = conn
            print(f'  client {idx} : {addr}')

    def train(self):
        print('----- train -----')
        for epoch in range(self.epochs):
            states = []
            comm_cost = 0.0
            agg_train_loss = 0
            agg_train_acc  = 0.0
            agg_test_loss  = 0
            agg_test_acc   = 0.0

            print(f'--- epoch {epoch+1} ---')
            for cid, conn in self.clients.items():
                print(f'  processing {cid}')
                for local_epoch in range(self.local_epochs):
                    nonstop = True

                    while nonstop:
                        rsp, rsplen = recvall(conn)
                        comm_cost += rsplen
                        action = rsp['action']

                        if action == 'train':
                            smashed = rsp['smashed']
                            labels = rsp['labels']
                            grad, loss, correct = self.server.train_batch(smashed, labels)

                            data = {
                                'grad'    : grad,
                                'loss'    : loss,
                                'correct' : correct,
                            }
                            msglen = sendall(conn, data)
                            comm_cost += msglen

                        elif action == 'test':
                            smashed = rsp['smashed']
                            labels = rsp['labels']
                            loss, correct = self.server.test_batch(smashed, labels)

                            data = {
                                'loss' : loss,
                                'correct' : correct,
                            }
                            msglen = sendall(conn, data)
                            comm_cost += msglen

                        elif action == 'finish':
                            nonstop = False
                            agg_train_loss += rsp['train_loss']
                            agg_train_acc += rsp['train_acc']
                            agg_test_loss += rsp['test_loss']
                            agg_test_acc += rsp['test_acc']

                rsp, rsplen = recvall(conn)
                comm_cost += rsplen
                state = rsp['state']
                states.append(state)

            agg_state = self.server.fedavg(states)
            self.model_states = agg_state
            print(f'  distributing agg state')
            for cid, conn in self.clients.items():
                msglen = sendall(conn, agg_state)
                comm_cost += msglen

            agg_train_loss /= (self.num_clients * self.local_epochs)
            agg_train_acc /= (self.num_clients * self.local_epochs)
            agg_test_loss /= (self.num_clients * self.local_epochs)
            agg_test_acc /= (self.num_clients * self.local_epochs)
            self.server.record.add(agg_train_loss, agg_train_acc, agg_test_loss, agg_test_acc)
            print(
                f"Epoch {epoch+1}/{self.epochs}: "
                f"Train : {agg_train_loss:.4f}/{agg_train_acc:.2f}%, "
                f"Test : {agg_test_loss:.4f}/{agg_test_acc:.2f}%, "
                f"Comm : {comm_cost/(1024**2):.2f}MB"
            )

    def posttrain(self, total_time):
        self.sock.close()

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        rdir = os.path.join(DIR_RESULTS, f'SL_{self.dataset.upper()}_{self.num_clients}_{ts}')
        os.makedirs(rdir, exist_ok=True)

        summary_data = {
            'name' : 'Split Learning',
            'date' : datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_time': total_time/60, # minutes
            'total_comm': sum(self.server.record.comm_costs), # MB
            'epochs' : self.epochs,
            'local_epochs' : self.local_epochs,
            'num_clients' : self.num_clients,
            'batch_size' : self.batch_size,
            'lr' : self.lr,
            'momentum' : self.momentum,
            'optimizer' : self.optimizer_name,
            'criterion': self.criterion_name,
            'scheduler' : self.scheduler_name,
            'final_test_acc': self.server.record.test_accs[-1]
        }
        with open(os.path.join(rdir, 'summary.json'), 'w') as f:
            json.dump(summary_data, f)

        self.server.record.saveto(os.path.join(rdir, 'server.csv'))
        torch.save(self.model_states, os.path.join(rdir, 'final_model.pth'))
        print(f'all results saved in {rdir}')