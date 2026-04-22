import socket
import torch

from common import *
from dataset import *
from network import *

from ...base import BaseClient

# ==================================================
#                   Client (Distributed)
# ==================================================


class DistributeSplitClient(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_epoch(self, conn: socket.socket):
        self.model.train()

        running_loss = 0.0
        total = 0
        correct = 0
        comm_cost = 0.0

        for idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.hardware), labels.to(self.hardware)

            # forward
            self.optimizer.zero_grad()
            smashed = self.model(inputs)
            smashed_copy = smashed.clone().detach().requires_grad_(True)

            data = {
                'action' : 'train',
                'smashed': smashed_copy,
                'labels' : labels,
            }
            sendall(conn, data)

            # backward
            rsp, rsplen = recvall(conn)
            smashed_grad = rsp['grad']
            smashed_loss = rsp['loss']
            smashed_correct = rsp['correct']

            smashed.backward(smashed_grad)
            self.optimizer.step()

            running_loss += smashed_loss
            correct += smashed_correct
            total += labels.size(0)

        train_loss = running_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        return train_loss, train_acc, comm_cost

    @torch.no_grad()
    def test_epoch(self, conn: socket.socket):
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in self.test_loader:
            inputs, labels = inputs.to(self.hardware), labels.to(self.hardware)

            smashed = self.model(inputs)
            data = {
                'action'  : 'test',
                'smashed' : smashed,
                'labels'  : labels
            }
            sendall(conn, data)

            rsp, rsplen = recvall(conn)
            smashed_loss = rsp['loss']
            smashed_correct = rsp['correct']

            running_loss += smashed_loss
            correct += smashed_correct
            total += labels.size(0)

        test_loss = running_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        return test_loss, test_acc