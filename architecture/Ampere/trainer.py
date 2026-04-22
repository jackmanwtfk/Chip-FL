# ==================================================
# ==================================================
#        Split Federated Learning - Ampere
# ==================================================
# ==================================================
import os
import copy
import json
import time
import logging
import datetime

import torch
import torch.multiprocessing as mp

from common import *
from dataset import *

from ..base import BaseTrainer, train_client
from .client import AmpereClient
from .server import AmpereServer

# ==================================================
#                   Trainer
# ==================================================

def train_server(client: AmpereClient):
    train_smashed, train_labels, test_smashed, test_labels = client.gen_activations()

    train_loss, train_acc = client.server.train_batch(train_smashed, train_labels)
    test_loss, test_acc = client.server.test_batch(test_smashed, test_labels)
    client.server.lr_step(test_acc)
    return {
        'id' : client.client_id,
        'size' : get_model_size(client.model),
        'train_loss' : train_loss,
        'train_acc' : train_acc,
        'test_loss' : test_loss,
        'test_acc' : test_acc,
        'comm_cost' : get_param_size(train_smashed)
    }


class AmpereTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ==================================================
    #                   pretrain
    # ==================================================

    def pretrain(self):
        self.prepare_dm()
        self.prepare_data()
        self.prepare_system()

    def prepare_system(self):
        self.server = AmpereServer(
            batch_size=self.batch_size, lr=self.lr, momentum=self.momentum,
        )

        smodel = self.create_model(is_client=False)
        self.server.set_model(smodel)

        for client_id in range(self.num_clients):
            client = AmpereClient(
                client_id,
                self.client_datasets[client_id],
                self.test_set,
                local_epochs=self.local_epochs,
                batch_size=self.batch_size,
                momentum=self.momentum,
                lr=self.lr,
            )

            cmodel = self.create_model(is_client=True)
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
                epoch_comm_cost += result['size']

            agg_state = self.server.fedavg(states)
            self.model_state = agg_state
            self.distribute_state(self.clients, agg_state)

            if self.parallel:
                with mp.Pool() as pool:
                    results = list(pool.map(train_server, selected_clients))
            else:
                results = [train_server(client) for client in selected_clients]

            for result in results:
                epoch_train_loss += result['train_loss'] / len(results)
                epoch_train_acc += result['train_acc'] / len(results)
                epoch_test_loss += result['test_loss'] / len(results)
                epoch_test_acc += result['test_acc'] / len(results)
                epoch_comm_cost += result['comm_cost']

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
        name = "Ampere (SFL) Training"
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        rdir = os.path.join(
            DIR_RESULTS,
            f"Ampere_{self.dataset.upper()}_{self.num_clients}_{ts}"
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
