# ==================================================
# ==================================================
#               Federated Learning
# ==================================================
# ==================================================
import os
import copy
import time
import logging
import datetime

import torch.multiprocessing as mp

from common import *
from dataset import *
from model import LeNet12

from ..base import BaseTrainer, train_client
from .client import FedClient
from .server import FedServer

# ==================================================
#                   Trainer
# ==================================================

class FedTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ==================================================
    #                   pretrain
    # ==================================================

    def pretrain(self):
        self.prepare_dm()
        self.prepare_data()
        self.prepare_system()
        print(f"Device: {self.hardware}")

    def prepare_system(self):
        self.server = FedServer()

        for cid in range(self.num_clients):
            client = FedClient(
                cid,
                self.client_datasets[cid],
                self.test_set,

                local_epochs=self.local_epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                momentum=self.momentum,

                criterion=self.criterion_name,
                optimizer=self.optimizer_name,
                scheduler=self.scheduler_name,
            )

            model = self.create_model()
            client.set_model(model)
            client.set_server(self.server)
            self.clients.append(client)

    # ==================================================
    #                   train
    # ==================================================

    def train(self):
        for epoch in range(self.epochs):
            print(f"===== epoch {epoch+1}/{self.epochs} =====")
            tic = time.time()

            states = []
            epoch_train_loss = 0
            epoch_train_acc = 0.0
            epoch_test_loss = 0
            epoch_test_acc = 0.0
            epoch_comm_cost = 0.0

            print("----- clients -----")
            selected_clients = self.select_clients()

            if self.parallel:
                with mp.Pool() as pool:
                    results = list(pool.map(train_client, selected_clients))
            else:
                results = [train_client(client) for client in selected_clients]

            for result in results:
                state = result['state']

                states.append(state)
                epoch_train_loss += result['train_loss'] / len(results)
                epoch_train_acc += result['train_acc'] / len(results)
                epoch_test_loss += result['test_loss'] / len(results)
                epoch_test_acc += result['test_acc'] / len(results)
                epoch_comm_cost += result['comm_cost'] + result['size']

            agg_state = self.server.fedavg(states)
            self.model_state = agg_state
            self.distribute_state(self.clients, self.model_state)

            self.server.record.add(
                epoch_train_loss,
                epoch_train_acc,
                epoch_test_loss,
                epoch_test_acc,
                epoch_comm_cost
            )

            tok = time.time()
            print("----- server -----")
            print(
                f"Train : {epoch_train_loss:.4f}/{epoch_train_acc:.2f}%, "
                f"Test : {epoch_test_loss:.4f}/{epoch_test_acc:.2f}%, "
                f"Comm : {epoch_comm_cost/(1024**2):.2f}MB, "
                f"{tok-tic:.2f}s"
            )

            stop_now, _ = self._es_step(
                epoch=epoch + 1,
                train_loss=epoch_train_loss,
                train_acc=epoch_train_acc,
                test_loss=epoch_test_loss,
                test_acc=epoch_test_acc,
                snapshot=self.model_state,
            )
            if stop_now:
                if self.early_stopping_restore_best and self._es_best_snapshot is not None:
                    self.model_state = copy.deepcopy(self._es_best_snapshot)
                    self.distribute_state(self.clients, self.model_state)
                print(
                    f"[EarlyStop] Stop at epoch {epoch+1}. "
                    f"Best {self.early_stopping_metric}={self._es_best_value:.6f} at epoch {self._es_best_epoch}."
                )
                break

    def posttrain(self, total_time):
        name = "Federated Learning"
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        rdir = os.path.join(DIR_RESULTS, f"FL_{self.dataset.upper()}_{self.num_clients}_{ts}")

        os.makedirs(rdir, exist_ok=True)
        self.save_summary(name, rdir, total_time)
        self.save_records(rdir)
        self.save_model(rdir)
        print(f"All saved to : {rdir}")
