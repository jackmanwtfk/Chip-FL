import time
import logging

import torch

from ..base import BaseClient

# ==================================================
#                   Client
# ==================================================

class AmpereClient(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_epoch(self):
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.hardware), labels.to(self.hardware)

            # forward
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # backward
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            total += labels.size(0)
            correct += outputs.max(1)[1].eq(labels).sum().item()

        train_loss = running_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        return train_loss, train_acc
    
    @torch.no_grad()
    def test_epoch(self):
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(self.test_loader):
            inputs, labels = inputs.to(self.hardware), labels.to(self.hardware)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item()
            total += labels.size(0)
            correct += outputs.max(1)[1].eq(labels).sum().item()

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

            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.test_epoch()
            self.lr_step(test_acc)

            local_train_loss += train_loss / self.local_epochs
            local_train_acc += train_acc / self.local_epochs
            local_test_loss += test_loss / self.local_epochs
            local_test_acc += test_acc / self.local_epochs
            self.record.add(train_loss, train_acc, test_loss, test_acc)
            print(
                f"  [{self.client_id}]({local_epoch+1}/{self.local_epochs}) "
                f"Train : {train_loss:.4f}/{train_acc:.2f}%, "
                f"Test : {test_loss:.4f}/{test_acc:.2f}%, "
                f"{time.time()-tic:6.2f}s"
            )

        return (
            local_train_loss,
            local_train_acc,
            local_test_loss,
            local_test_acc,
            local_comm_cost
        )

    @torch.no_grad()
    def gen_activations(self):
        self.model.is_activation = True

        train_smashed = []
        train_labels = []
        test_smashed = []
        test_labels = []

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.hardware), labels.to(self.hardware)
            smashed = self.model(inputs)
            train_smashed.append(smashed)
            train_labels.append(labels)

        for inputs, labels in self.test_loader:
            inputs, labels = inputs.to(self.hardware), labels.to(self.hardware)
            smashed = self.model(inputs)
            test_smashed.append(smashed)
            test_labels.append(labels)

        self.model.is_activation = False
        return train_smashed, train_labels, test_smashed, test_labels