import torch

from ..base import BaseServer


# ==================================================
#                   Server
# ==================================================

class AmpereServer(BaseServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_batch(self, local_smashed, local_labels):
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for idx, (smashed, labels) in enumerate(zip(local_smashed, local_labels)):
            smashed, labels = smashed.to(self.hardware), labels.to(self.hardware)

            self.optimizer.zero_grad()

            outputs = self.model(smashed)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        batch_loss = running_loss / (idx + 1) 
        batch_acc = 100. * correct / total
        return batch_loss, batch_acc
    
    @torch.no_grad()
    def test_batch(self, local_smashed, local_labels):
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for idx, (smashed, labels) in enumerate(zip(local_smashed, local_labels)):
            smashed, labels = smashed.to(self.hardware), labels.to(self.hardware)
            outputs = self.model(smashed)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        batch_loss = running_loss / (idx + 1) 
        batch_acc = 100. * correct / total
        return batch_loss, batch_acc
