import torch

from common import *
from dataset import *
from network import *

from ...base import BaseServer


# ==================================================
#                   Server (Distributed)
# ==================================================

class DistributeSplitServer(BaseServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_batch(self, smashed, labels):
        self.model.train()

        self.optimizer.zero_grad()
        smashed, labels = smashed.to(self.hardware), labels.to(self.hardware)

        outputs = self.model(smashed)
        loss = self.criterion(outputs, labels)
        loss.backward()

        grad = smashed.grad.clone().detach()
        self.optimizer.step()

        loss = loss.item()
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        return grad, loss, correct

    @torch.no_grad()
    def test_batch(self, smashed, labels):
        self.model.eval()

        smashed, labels = smashed.to(self.hardware), labels.to(self.hardware)
        outputs = self.model(smashed)
        loss = self.criterion(outputs, labels)
            
        smashed_loss = loss.item()
        _, predicted = outputs.max(1)
        smashed_correct = predicted.eq(labels).sum().item()

        return smashed_loss, smashed_correct