import os
import copy
import math
import json
import random
import datetime

import torch
import numpy as np

from common import *
from dataset import *
from model import *


# ==================================================
#                   Trainer
# ==================================================

class BaseTrainer:
    def __init__(self,
        dataset="mnist",
        model_name='lenet',
        num_clients=1,
        cut_layer=0,
        local_epochs=0,
        epochs=20,
        batch_size=64,
        lr=0.01,
        momentum=0.9,
        parallel=False,
        num_select=0,
        criterion='ce',
        optimizer='sgd',
        scheduler='steplr',
        early_stopping=True,
        early_stopping_metric='test_loss',
        early_stopping_mode='min',
        early_stopping_patience=20,
        early_stopping_min_delta=0.0,
        early_stopping_warmup=0,
        early_stopping_restore_best=True,
    ):
        # set manual seed (for reproduce)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        # client & server
        self.clients = []
        self.server = None

        # train
        self.dataset = dataset
        self.dataset_config = DatasetConfig[self.dataset]
        self.model_name = model_name
        self.model_config = ModelConfig[self.model_name]

        self.parallel = parallel
        self.hardware = find_device()
        self.model_state = None

        # hyperparameter
        self.epochs = epochs
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.cut_layer = cut_layer  # only used in SL
        self.num_clients = num_clients
        self.num_select = num_select

        self.criterion_name = criterion.lower()
        self.optimizer_name = optimizer.lower()
        self.scheduler_name = scheduler.lower()

        # early stopping
        self.early_stopping = bool(early_stopping)
        self.early_stopping_metric = str(early_stopping_metric)
        self.early_stopping_mode = str(early_stopping_mode).lower()
        if self.early_stopping_mode not in ('min', 'max'):
            self.early_stopping_mode = 'min'
        self.early_stopping_patience = max(0, int(early_stopping_patience))
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.early_stopping_warmup = max(0, int(early_stopping_warmup))
        self.early_stopping_restore_best = bool(early_stopping_restore_best)
        self._es_best_value = None
        self._es_best_epoch = 0
        self._es_bad_epochs = 0
        self._es_best_snapshot = None
        self._es_stopped_epoch = None
        self._es_monitor_missing_warned = False

    # ==================================================
    #                   general
    # ==================================================

    def select_clients(self):
        if self.num_select == 0:
            return self.clients
        else:
            return random.sample(self.clients, self.num_select)

    def distribute_state(self, clients, state):
        for client in clients:
            client.load_state(state)

    def aggregate_states(self, w):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))

        return w_avg

    def _es_get_value(self, train_loss, train_acc, test_loss, test_acc, extra_metrics=None):
        metric = self.early_stopping_metric
        if metric == 'train_loss':
            return float(train_loss)
        if metric == 'train_acc':
            return float(train_acc)
        if metric == 'test_loss':
            return float(test_loss)
        if metric == 'test_acc':
            return float(test_acc)
        if isinstance(extra_metrics, dict) and metric in extra_metrics:
            try:
                return float(extra_metrics[metric])
            except Exception:
                return None
        return None

    def _es_is_better(self, current, best):
        if self.early_stopping_mode == 'max':
            return current > (best + self.early_stopping_min_delta)
        return current < (best - self.early_stopping_min_delta)

    def _es_step(
        self,
        epoch,
        train_loss,
        train_acc,
        test_loss,
        test_acc,
        snapshot=None,
        extra_metrics=None,
    ):
        if not self.early_stopping:
            return False, None

        value = self._es_get_value(train_loss, train_acc, test_loss, test_acc, extra_metrics=extra_metrics)
        if value is None:
            if not self._es_monitor_missing_warned:
                print(
                    f"[EarlyStop] monitor '{self.early_stopping_metric}' not found, "
                    "early stopping is disabled for this run."
                )
                self._es_monitor_missing_warned = True
            self.early_stopping = False
            return False, None

        if not math.isfinite(value):
            return False, value

        if self._es_best_value is None or self._es_is_better(value, self._es_best_value):
            self._es_best_value = value
            self._es_best_epoch = int(epoch)
            self._es_bad_epochs = 0
            if snapshot is not None:
                self._es_best_snapshot = copy.deepcopy(snapshot)
            return False, value

        if int(epoch) > self.early_stopping_warmup:
            self._es_bad_epochs += 1

        if self._es_bad_epochs >= self.early_stopping_patience:
            self._es_stopped_epoch = int(epoch)
            return True, value

        return False, value

    def _es_summary(self):
        return {
            'enabled': self.early_stopping,
            'metric': self.early_stopping_metric,
            'mode': self.early_stopping_mode,
            'patience': self.early_stopping_patience,
            'min_delta': self.early_stopping_min_delta,
            'warmup': self.early_stopping_warmup,
            'restore_best': self.early_stopping_restore_best,
            'best_epoch': self._es_best_epoch,
            'best_value': self._es_best_value,
            'stopped_epoch': self._es_stopped_epoch,
        }

    # ==================================================
    #                   pretrain
    # ==================================================

    def pretrain(self):
        raise NotImplementedError

    def prepare_system(self):
        raise NotImplementedError

    def prepare_dm(self):
        batch_size = self.batch_size
        input_size = self.model_config['input_size']

        if self.dataset == 'mnist':
            dm = MnistDM(batch_size, input_size )
        elif self.dataset == 'fmnist':
            dm = FMnistDM(batch_size, input_size)
        elif self.dataset == 'cifar10':
            dm = CIFAR10DM(batch_size, input_size)
        else:
            raise UserWarning(f'Dataset {self.dataset} not supported yet!')

        self.dm = dm

    def prepare_data(self):
        full_train_set, self.test_set = self.dm.get_datasets()
        self.client_datasets = split_dataset(full_train_set, self.num_clients)

    def create_model(self, **kwargs):
        model_name = self.model_name
        channels = self.dataset_config['channels']
        nclasses = self.dataset_config['nclasses']
        return self._create_model(model_name, channels, nclasses, **kwargs)

    def _create_model(self, model_name, channels, nclasses, **kwargs):
        if model_name == 'lenet':
            if 'is_client' in kwargs.keys():
                model = LeNet12Ampere(channels, nclasses, **kwargs)
            else:
                model = LeNet12(channels, nclasses, **kwargs)
        elif model_name == 'alexnet':
            model = AlexNet(channels, nclasses, **kwargs)
        elif model_name == 'vgg16':
            model = VGG16(channels, nclasses, **kwargs)
        else:
            raise UserWarning(f"Model {model_name} Not supported for now.")
        return model

    # ==================================================
    #                     train
    # ==================================================

    def train(self):
        raise NotImplementedError

    # ==================================================
    #                   posttrain
    # ==================================================

    def posttrain(self, total_time):
        raise NotImplementedError
    
    def save_summary(self, name, rdir, total_time):
        summary_data = {
            'name'           : name,
            'date'           : datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_time'     : total_time,

            'dataset'        : self.dataset.upper(),
            'epochs'         : self.epochs,
            'local_epochs'   : self.local_epochs,
            'num_clients'    : self.num_clients,
            'num_select'     : self.num_select,
            'parallel'       : self.parallel,

            'batch_size'     : self.batch_size,
            'lr'             : self.lr,
            'momentum'       : self.momentum,

            'hardware'       : self.hardware.type,
            'optimizer'      : self.optimizer_name,
            'criterion'      : self.criterion_name,
            'scheduler'   : self.scheduler_name,

            'final_train_loss' : self.server.record.train_losses[-1],
            'final_train_acc' : self.server.record.train_accs[-1],
            'final_test_loss' : self.server.record.test_losses[-1],
            'final_test_acc' : self.server.record.test_accs[-1],
            'total_comm'     : sum(self.server.record.comm_costs), # MB
            'early_stopping' : self._es_summary(),
        }
        print(json.dumps(summary_data, indent=2))
        with open(os.path.join(rdir, 'summary.json'), 'w') as f:
            json.dump(summary_data, f, indent=2)

    def save_records(self, rdir):
        for client in self.clients:
            client.record.saveto(os.path.join(rdir, f"client_{client.client_id}.csv"))
        self.server.record.saveto(os.path.join(rdir, "server.csv"))

    def save_model(self, rdir):
        torch.save(self.model_state, os.path.join(rdir, "final_model.pth"))

class BaseDistributedTrainer(BaseTrainer):
    def __init__(self, host_ip, host_port, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conns = {}
        self.host_addr = (host_ip, host_port)
