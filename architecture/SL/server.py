import torch

from ..base import BaseServer


# ==================================================
#                   Server
# ==================================================

class SplitServer(BaseServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_batch(self, smashed, labels):
        self.model.train()

        self.optimizer.zero_grad()
        smashed, labels = smashed.to(self.hardware), labels.to(self.hardware)

        outputs = self.model(smashed)
        loss = self.criterion(outputs, labels)
        loss.backward()

        smashed_grad = smashed.grad.clone().detach()
        self.optimizer.step()

        smashed_loss = loss.item()
        _, predicted = outputs.max(1)
        smashed_correct = predicted.eq(labels).sum().item()
        return smashed_grad, smashed_loss, smashed_correct

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