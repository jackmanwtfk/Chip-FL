import time
import torch

from common import *
from dataset import *

from ..base import BaseClient
from .server import SplitServer


# ==================================================
#                   Client
# ==================================================

class SplitClient(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_epoch(self, server : SplitServer):
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
            comm_cost += get_param_size(smashed_copy)

            # backward
            smashed_grad, smashed_loss, smashed_correct = server.train_batch(smashed_copy, labels)
            comm_cost += get_param_size(smashed_grad)
            smashed.backward(smashed_grad)
            self.optimizer.step()

            running_loss += smashed_loss
            correct += smashed_correct
            total += labels.size(0)

        train_loss = running_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        return train_loss, train_acc, comm_cost

    @torch.no_grad()
    def test_epoch(self, server: SplitServer):
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in self.test_loader:
            inputs, labels = inputs.to(self.hardware), labels.to(self.hardware)

            smashed = self.model(inputs)
            smashed_loss, smashed_correct = server.test_batch(smashed, labels)

            running_loss += smashed_loss
            correct += smashed_correct
            total += labels.size(0)

        test_loss = running_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        return test_loss, test_acc
    
    def train_test_local(self):
        local_train_loss = 0
        local_train_acc = 0.0
        local_test_loss = 0
        local_test_acc = 0.0
        local_comm_cost = 0.0

        for local_epoch in range(self.local_epochs):
            tic = time.time()

            train_loss, train_acc, comm_cost = self.train_epoch(self.server)
            test_loss, test_acc = self.test_epoch(self.server)
            self.lr_step(test_acc)
            self.server.lr_step(test_acc)

            local_train_loss += train_loss / self.local_epochs
            local_train_acc += train_acc / self.local_epochs
            local_test_loss += test_loss / self.local_epochs
            local_test_acc += test_acc / self.local_epochs
            local_comm_cost += comm_cost

            self.record.add(train_loss, train_acc, test_loss, test_acc, comm_cost)
            print(
                f"  [{self.client_id}] ({local_epoch+1}/{self.local_epochs}) "
                f"Train: {train_loss:.4f}/{train_acc:.2f}%, "
                f"Test: {test_loss:.4f}/{test_acc:.2f}%, "
                f"{time.time()-tic:6.2f}s"
            )

        return (
            local_train_loss,
            local_train_acc,
            local_test_loss,
            local_test_acc,
            local_comm_cost
        )
