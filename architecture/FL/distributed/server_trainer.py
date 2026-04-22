import os
import json
import time
import socket
import datetime

import torch

from common import *
from dataset import *
from network import *

from ...base import BaseDistributedTrainer
from ..server import FedServer


class FedServerTrainer(BaseDistributedTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server = FedServer()

        self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.sock.bind(self.host_addr)
        self.sock.listen()

    # ==================================================
    #                   pretrain
    # ==================================================

    def pretrain(self):
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
            self.conns[idx] = conn
            print(f'  client {idx} : {conn.getpeername()}')

    # ==================================================
    #                      train
    # ==================================================

    def train(self):
        for epoch in range(self.epochs):
            tic = time.time()
            states = []
            epoch_train_loss = 0
            epoch_train_acc = 0.0
            epoch_test_loss = 0
            epoch_test_acc = 0.0
            epoch_comm_cost = 0.0

            for idx, conn in self.conns.items():
                msg, msglen = recvall(conn)
                print(f'  recv {idx} state')
                train_loss = msg['train_loss']
                train_acc = msg['train_acc']
                test_loss = msg['test_loss']
                test_acc = msg['test_acc']
                state = msg['state']

                epoch_comm_cost += msglen
                epoch_train_loss += train_loss
                epoch_train_acc += train_acc
                epoch_test_loss += test_loss
                epoch_test_acc += test_acc
                states.append(state)

            print('  aggregate states')
            agg_state = self.aggregate_states(states)
            for idx, conn in self.conns.items():
                msglen = sendall(conn, agg_state)
                print(f'  send agg states to {idx}')
                epoch_comm_cost += msglen
            self.model_state = agg_state

            epoch_train_loss /= self.num_clients
            epoch_train_acc /= self.num_clients
            epoch_test_loss /= self.num_clients
            epoch_test_acc /= self.num_clients
            self.server.record.add(
                epoch_train_loss,
                epoch_train_acc,
                epoch_test_loss,
                epoch_test_acc,
                epoch_comm_cost
            )
            print(
                f"Epoch {epoch+1}/{self.epochs}: "
                f"Train : {epoch_train_loss:.4f}/{epoch_train_acc:.2f}%, "
                f"Test : {epoch_test_loss:.4f}/{epoch_test_acc:.2f}%, "
                f"Comm : {epoch_comm_cost/(1024**2):.2f}MB, "
                f"{time.time()-tic:6.2f}s"
            )

    # ==================================================
    #                    posttrain
    # ==================================================

    def posttrain(self, total_time):
        self.sock.close()

        name = "Federated Learning"
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        rdir = os.path.join(DIR_RESULTS, f'FL_{self.dataset.upper()}_{self.num_clients}_{ts}')

        os.makedirs(rdir, exist_ok=True)
        self.save_summary(name, rdir, total_time)
        self.save_records(rdir)
        self.save_model(rdir)
        print(f'All saved to : {rdir}')
