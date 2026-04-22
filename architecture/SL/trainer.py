# ==================================================
# ==================================================
#               Split Learning
# ==================================================
# ==================================================
import os
import copy
import time
import logging
import datetime

import torch
import torch.multiprocessing as mp

from common import *
from dataset import *
from model import LeNet12

from ..base import BaseTrainer, train_client
from .client import SplitClient
from .server import SplitServer

# ==================================================
#                   Trainer
# ==================================================

class SplitTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ==================================================
    #                   pretrain
    # ==================================================

    def pretrain(self):
        self.prepare_dm()
        self.prepare_data()
        self.prepare_system()

        print(f"model : {self.model_name}")
        print(f"dataset : {self.dataset}")
        print(f"hardware : {self.hardware}")

    def prepare_system(self):
        model = self.create_model()
        _, smodel = model.split_model(self.cut_layer)

        self.server = SplitServer(
            batch_size=self.batch_size,
            lr=self.lr,
            momentum=self.momentum,

            criterion=self.criterion_name,
            optimizer=self.optimizer_name,
            scheduler=self.scheduler_name,
        )
        self.server.set_model(smodel)

        for client_id in range(self.num_clients):
            model = self.create_model()
            cmodel, _ = model.split_model(self.cut_layer)

            client = SplitClient(
                client_id,
                self.client_datasets[client_id],
                self.test_set,

                local_epochs=self.local_epochs,
                batch_size=self.batch_size,
                momentum=self.momentum,
                lr=self.lr,

                criterion=self.criterion_name,
                optimizer=self.optimizer_name,
                scheduler=self.scheduler_name,
            )
            client.set_model(cmodel)
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
                epoch_comm_cost += result['comm_cost']

            agg_state = self.server.fedavg(states)
            self.model_state = agg_state
            self.distribute_state(self.clients, agg_state)

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

            snapshot = {
                'client_state': self.model_state,
                'server_state': self.server.model.state_dict(),
            }
            stop_now, _ = self._es_step(
                epoch=epoch + 1,
                train_loss=epoch_train_loss,
                train_acc=epoch_train_acc,
                test_loss=epoch_test_loss,
                test_acc=epoch_test_acc,
                snapshot=snapshot,
            )
            if stop_now:
                if self.early_stopping_restore_best and self._es_best_snapshot is not None:
                    best = copy.deepcopy(self._es_best_snapshot)
                    self.model_state = best['client_state']
                    self.distribute_state(self.clients, self.model_state)
                    self.server.model.load_state_dict(best['server_state'])
                print(
                    f"[EarlyStop] Stop at epoch {epoch+1}. "
                    f"Best {self.early_stopping_metric}={self._es_best_value:.6f} at epoch {self._es_best_epoch}."
                )
                break

    # ==================================================
    #                   posttrain
    # ==================================================

    def posttrain(self, total_time):
        name = "Split Learning"
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        rdir = os.path.join(DIR_RESULTS,
            f"SL_{self.dataset.upper()}_{self.num_clients}_{self.cut_layer}_{ts}"
        )

        os.makedirs(rdir, exist_ok=True)
        self.save_summary(name, rdir, total_time)
        self.save_records(rdir)
        self.save_model(rdir)
        print(f"All saved to : {rdir}")

    def save_model(self, rdir):
        for client in self.clients:
            torch.save(
                client.model.state_dict(),
                os.path.join(rdir,f"client_{client.client_id}_model.pth")
            )
        torch.save(self.server.model.state_dict(), os.path.join(rdir, "server_model.pth"))
