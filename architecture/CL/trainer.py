# ==================================================
# ==================================================
#               Centralized Learning
# ==================================================
# ==================================================
import os
import copy
import json
import time
import datetime

import torch

from common import *
from dataset import *
from model import LeNet12

from ..base import BaseTrainer


class CLTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.record = Record()

    # ==================================================
    #                   pretrain
    # ==================================================

    def pretrain(self):
        self.prepare_dm()
        self.train_loader, self.test_loader = self.dm.get_dataloaders()

        self.model = self.create_model()
        self.model.to(self.hardware)
        self.criterion = find_criterion(self.criterion_name)
        self.optimizer = find_optimizer(self.model, self.optimizer_name, self.lr, self.momentum)
        self.scheduler = find_scheduler(self.optimizer, self.scheduler_name)

    # ==================================================
    #                   training
    # ==================================================

    def train_epoch(self):
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.hardware), labels.to(self.hardware)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        return train_loss, train_acc

    @torch.no_grad()
    def evaluate_epoch(self):
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.hardware), targets.to(self.hardware)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            running_loss += loss.item()
            total += targets.size(0)
            correct += outputs.max(1)[1].eq(targets).sum().item()

        test_loss = running_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        return test_loss, test_acc

    def train(self):
        for epoch in range(self.epochs):
            tik = time.time()

            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.evaluate_epoch()
            self.scheduler.step()

            self.record.add(train_loss, train_acc, test_loss, test_acc)

            tok = time.time()
            print(
                f"Epoch {epoch+1}/{self.epochs}: "
                f"Train : {train_loss:.4f}/{train_acc:.2f}%, "
                f"Test : {test_loss:.4f}/{test_acc:.2f}%, "
                f"{tok-tik:.2f}s"
            )

            stop_now, _ = self._es_step(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_acc=train_acc,
                test_loss=test_loss,
                test_acc=test_acc,
                snapshot=self.model.state_dict(),
            )
            if stop_now:
                if self.early_stopping_restore_best and self._es_best_snapshot is not None:
                    self.model.load_state_dict(copy.deepcopy(self._es_best_snapshot))
                print(
                    f"[EarlyStop] Stop at epoch {epoch+1}. "
                    f"Best {self.early_stopping_metric}={self._es_best_value:.6f} at epoch {self._es_best_epoch}."
                )
                break

    # ==================================================
    #                   posttrain
    # ==================================================

    def posttrain(self, total_time):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        rdir = os.path.join(DIR_RESULTS, f"CL_{self.dataset.upper()}_{ts}")
        os.makedirs(rdir, exist_ok=True)

        results_path = os.path.join(rdir, "training_results.csv")
        self.record.saveto(results_path)

        summary_data = {
            'Training data' : datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Total Training Time' : total_time,
            'Epochs' : self.epochs,
            'Batch size' : self.batch_size,
            'Learning rate' : self.lr,
            'Momentum' : self.momentum,
            'Optimizer' : self.optimizer_name,
            'Loss Function' : self.criterion_name,
            'LR Scheduler' : self.scheduler_name,
            'Final test accuracy' : self.record.test_accs[-1],
            'Early stopping' : self._es_summary(),
        }
        with open(os.path.join(rdir, "training_summary.txt"), 'w') as f:
            json.dump(summary_data, f)

        torch.save(self.model.state_dict(), os.path.join(rdir, f"{self.dataset}.pth"))
        print(f"All saved to {rdir}")
