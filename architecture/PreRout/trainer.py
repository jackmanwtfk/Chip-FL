import os
import sys
import json
import time
import random
import datetime
import copy
import math
import re

import torch
from torch.nn import functional as F

from common import Record, find_device
from .client import PreRoutClient, GraphAutoEncoder

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def _resolve_prerout_root():
    candidates = [
        os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'PreRoutGNN')),
        os.path.abspath(os.path.join(PROJECT_ROOT, 'PreRoutGNN')),
        os.path.abspath(os.path.join(PROJECT_ROOT, '..', '..', 'PreRoutGNN')),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]

PREROUT_ROOT = _resolve_prerout_root()

def _ensure_prerout_imports():
    """延迟导入 PreRoutGNN 模块，避免与当前仓库的同名模块冲突。"""
    # 暂存当前仓库中可能与 PreRoutGNN 同名的模块，导入结束后再恢复
    fed_dataset = sys.modules.pop('dataset', None)
    fed_utils = sys.modules.pop('utils', None)
    fed_model = sys.modules.pop('model', None)
    fed_config = sys.modules.pop('config', None)
    
    # 临时把 PreRoutGNN 路径放到 sys.path 前面，优先导入它的模块
    if PREROUT_ROOT not in sys.path:
        sys.path.insert(0, PREROUT_ROOT)
    
    try:
        global ConfigAutoEncoder, Config, process_data_homo, process_data_hetero
        global hetero_gen_topo, hetero_generate_homo_from_hetero_single
        global get_model_autoencoder, get_graph_encoder, get_prerout_model, prerout_utils, PreRoutGNN_Model
        from config.config_autoencoder import ConfigAutoEncoder  # type: ignore
        from config.config import Config  # type: ignore
        from dataset.data_homo import process_data_homo  # type: ignore
        from dataset.data_hetero import (
            process_data_hetero,  # type: ignore
            gen_topo as hetero_gen_topo,  # type: ignore
            generate_homo_from_hetero_single as hetero_generate_homo_from_hetero_single,  # type: ignore
        )
        from main_graph_autoencoder import get_model as get_model_autoencoder  # type: ignore
        from main import get_graph_encoder, get_model as get_prerout_model  # type: ignore
        import utils as prerout_utils  # type: ignore
        import model as PreRoutGNN_Model  # type: ignore
    finally:
        # 恢复当前仓库原本的同名模块
        if fed_dataset:
            sys.modules['dataset'] = fed_dataset
        if fed_utils:
            sys.modules['utils'] = fed_utils
        if fed_model:
            sys.modules['model'] = fed_model
        if fed_config:
            sys.modules['config'] = fed_config


DEFAULT_PREROUT_CFG = {
    'circuits_json_path': os.path.join('..', 'PreRoutGNN', 'circuits_json', 'circuits-original.json'),
    'data_split_json_path': os.path.join('..', 'PreRoutGNN', 'circuits_split_json', 'data_split-original.json'),
    'root_dir': 'result-graph_autoencoder',
    'model_type': 'PreRoutGNN',
    'num_level_freq_compents': 4,
    'max_level': 200,
    'num_pin_location_freq_compents': 0,
    'hidden_dim': 32,
    'latent_dim': 4,
    'lr': 5e-4,
    'lr_decay_rate': 1.0,
    'gap_update_lr': 100000,
    'dropout': 0.0,
    'sub_graph_size': 50000,
    'max_gradient_norm': 1000.0,
    'device': 'cuda',
    'num_epochs': 1,
    'normalize_pin_location': True,
    'pretrained_checkpoint': None,
    'weight_KL_divergence': 0.0,
    'n_layers': 4,
    'scale_capacitance': 1.0,
    'not_check_datasplit': False,
    'move_to_cuda_in_advance': False,
    'optimizer': 'Adam',
    'seed': 3407,
    'groundtruth': True,
    'test': False,
    'finetune': False,
    'save_inference': False,
    'gap_save_checkpoints': 200,
    # checkpoint 与断点续训控制：
    # - checkpoint_save_every <= 0 时，回退使用 gap_save_checkpoints
    # - resume_training=True 时，从显式路径或 checkpoints_dir 中最新文件恢复
    'checkpoint_save_every': 0,
    'checkpoint_keep_last': 3,
    'resume_training': False,
    'resume_checkpoint_path': '',
    'gap_test': 1,
    'gap_save_train_loss': 20,
    'comment': None,
    'output_dir': None,
    # 完整模型相关配置
    'use_graph_autoencoder': True,
    'graph_autoencoder_checkpoint': None,
    'graph_autoencoder_model_type': 'DeepGCNII',
    'graph_autoencoder_hidden_dim': 32,
    'graph_autoencoder_latent_dim': 4,
    'graph_autoencoder_n_layers': 4,
    'graph_autoencoder_lr': 5e-4,
    'finetune_graph_autoencoder': True,
    'predict_slew': True,
    'predict_netdelay': True,
    'predict_celldelay': True,
    'loss_weights': {
        'AT': 1.0,
        'slew': 1.0,
        'netdelay': 1.0,
        'celldelay': 1.0,
    },
    'slew_loss_type': 'mse',
    'slew_huber_beta': 1.0,
    'celldelay_loss_type': 'mse',
    'celldelay_huber_beta': 1.0,
    # FedEDA 漂移正则系数（0.0 表示关闭，非零表示启用）
    'fededa_gamma_size': 0.0,
    'fededa_gamma_p': 0.0,
    'fededa_alpha_clip_max': 10.0,
    'fededa_lambda_reduce': 'mean',
    'fededa_drift_normalize': True,
    # CM(size) 的来源：优先 netlist csv，缺失时退化为图节点数
    'fededa_size_source': 'netlist_csv',
    'fededa_size_csv_path': os.path.abspath(
        os.path.join(PROJECT_ROOT, '..', 'netlists', 'cell_size_from_netlists.csv')
    ),
    'fededa_size_csv_column': 'size_cells_from_verilog_sky130',
    # 可选的预计算 CM(size/p) JSON，由门级网表离线生成。
    # 一旦可用，client 端会优先读取它，而不是在训练时图上近似估计。
    'fededa_cm_source': 'precomputed_json',
    'fededa_cm_json_path': os.path.abspath(
        os.path.join(PROJECT_ROOT, '..', 'netlists', 'fededa_cm_stats.json')
    ),
    # 联邦算法选项：local | fedavg | fedprox | scaffold
    'fl_algorithm': 'fedavg',
    # FedProx 系数 mu，对应 (mu/2)*||w-w_t||^2
    'fedprox_mu': 0.0,
    'fedprox_drift_normalize': True,
    # SCAFFOLD 中 server 控制变量更新的缩放系数
    'scaffold_server_lr': 1.0,
    # SCAFFOLD 控制变量聚合方式：equal | sample
    'scaffold_control_aggregation_mode': 'equal',
    # 长训练场景下的 early stopping 配置
    'early_stopping': False,
    'early_stopping_metric': 'combo_r2_slew_netdelay',
    'early_stopping_mode': 'max',
    'early_stopping_patience': 30,
    'early_stopping_min_delta': 0.0,
    'early_stopping_warmup': 20,
    'early_stopping_restore_best': True,
    # 双任务约束 early stopping：
    # 只有当两个监控任务都在 patience 轮内没有提升时才停止
    'early_stopping_dual_task': True,
    'early_stopping_dual_metrics': ['r2-slew', 'r2-netdelay'],
    'early_stopping_dual_mode': 'max',
    # FedAvg 聚合方式：
    # - sample：按本地样本数加权
    # - equal：各 client 等权平均
    'fedavg_aggregation_mode': 'sample',
    # client 数据切分策略：
    # - group_by_parent_circuit：同一父电路的所有子电路都放在同一个 client
    # - round_robin：按样本 key 轮转随机分配
    # - ls_fixed_3clients：8_rat 训练集固定 LS 切分（5/5/5）
    # - ls_qs_fixed_3clients：8_rat 训练集固定 LS+QS 切分（3/5/7）
    'client_data_split_strategy': 'group_by_parent_circuit',
    # 固定切分策略下，要求与训练设计集合严格匹配
    'client_fixed_split_strict': True,
    # Train/Val/Test 划分设置
    # - 默认比例为 7:1:2
    # - 按父电路分组切分，避免子图之间泄漏
    'return_val_split': True,
    'train_val_test_ratio': [7, 1, 2],
    'train_val_test_seed': 3407,
    'train_val_test_group_by_parent': True,
    'train_val_test_repartition_from_all': False,
    'debug_simulated_metrics_enabled': False,
    'debug_simulated_metrics_path': '',
    'debug_simulated_metrics_selector': '',
    'debug_simulated_metrics_inline': None,
    'hidden_dim_cellprop': 128,
    'hidden_dim_netprop': 128,
    'hidden_dim_gcn': 64,
}


class PreRoutFedTrainer:
    def __init__(self,
        num_clients=3,
        local_epochs=1,
        epochs=5,
        lr=5e-4,
        num_select=0,
        parallel=False,
        prerout=None,
        **kwargs,
    ):
        self.num_clients = num_clients
        self.local_epochs = local_epochs
        self.epochs = epochs
        self.lr = lr
        self.num_select = num_select
        self.parallel = parallel
        self.prerout_cfg_overrides = prerout or {}

        self.clients = []
        self.record = Record()
        self.metrics_history = []  # 保存每轮更细粒度的指标记录
        self.data_train = {}
        self.data_val = {}
        self.data_test = {}
        self.server_device = find_device()
        self.server_model = None
        self.server_encoder = None
        self.fl_algorithm = 'fedavg'
        self.scaffold_server_control = None
        self.final_test_loss = math.nan
        self.final_test_raw_metrics = {}
        self.final_test_raw_task_metrics = {}
        self.final_test_metrics = {}
        self.final_test_task_metrics = {}
        self.debug_simulated_metrics_enabled = False
        self.debug_simulated_metrics_path = ''
        self.debug_simulated_metrics_selector = ''
        self.debug_simulated_metrics_inline = None
        self.debug_simulated_metrics_source = ''
        self.early_stopping = True
        self.early_stopping_metric = 'combo_r2_slew_netdelay'
        self.early_stopping_mode = 'max'
        self.early_stopping_patience = 30
        self.early_stopping_min_delta = 0.0
        self.early_stopping_warmup = 20
        self.early_stopping_restore_best = True
        self.early_stopping_dual_task = True
        self.early_stopping_dual_metrics = ('r2-slew', 'r2-netdelay')
        self.early_stopping_dual_mode = 'max'
        self._es_best_value = None
        self._es_best_epoch = 0
        self._es_bad_epochs = 0
        self._es_best_snapshot = None
        self._es_stopped_epoch = None
        self._es_monitor_missing_warned = False
        self._es_dual_monitor_missing_warned = False
        self._es_dual_best = {}
        self._es_dual_bad_epochs = {}
        self._es_stop_reason = None
        self.start_epoch = 0
        self.resume_training = False
        self.resume_checkpoint_path = ''
        self.checkpoint_save_every = 0
        self.checkpoint_keep_last = 3
        self.checkpoints_dir = ''
        self.resumed_from_checkpoint = None

# 训练前的完整初始化阶段，负责把配置、数据、client、server 状态都准备好
    def pretrain(self):
        _ensure_prerout_imports()  # 延迟导入 PreRoutGNN 模块
        self._prepare_config()
        self._prepare_data()
        self._prepare_clients()
        self._try_resume_training()
        print(f"PreRout FL ready. Device: {self.server_device}, Algorithm: {self.fl_algorithm}")
# 联邦训练主循环
    def train(self):
        start_epoch = max(0, int(self.start_epoch))
        if start_epoch >= int(self.epochs):
            print(
                f"[Checkpoint] start_epoch={start_epoch} already reached epochs={self.epochs}; "
                "skip local-update loop and run final test directly."
            )
        if start_epoch > 0:
            print(f"[Checkpoint] Continue training from epoch {start_epoch + 1}/{self.epochs}.")

        for epoch in range(start_epoch + 1, self.epochs + 1):
            print(f"===== epoch {epoch}/{self.epochs} =====")
            tic = time.time()

            selected_clients = self.clients if self.fl_algorithm == 'local' else self._select_clients()

            # FedEDA: distribute current global model state w_t to selected clients
            # so that their local loss can compute ||w_t - w_local||^2
            if (
                self.fl_algorithm != 'local'
                and self.server_encoder is not None
                and self.server_model is not None
            ):
                enc_sd = self.server_encoder.state_dict()
                mdl_sd = self.server_model.state_dict()
                for client in selected_clients:
                    if hasattr(client, 'set_global_state'):
                        client.set_global_state(enc_sd, mdl_sd)
                    if (
                        self.fl_algorithm == 'scaffold'
                        and self.scaffold_server_control is not None
                        and hasattr(client, 'set_scaffold_global_control')
                    ):
                        client.set_scaffold_global_control(
                            self.scaffold_server_control['encoder'],
                            self.scaffold_server_control['model'],
                        )

            results = [client.train_local(self.local_epochs) for client in selected_clients]

            weights = [r.get('num_samples', 1) for r in results]
            if self.fl_algorithm == 'local':
                global_val_loss, global_val_rec, global_metrics = self._evaluate_clients_average(
                    self.data_val,
                    split_name='val',
                )
            else:
                states = [r['state'] for r in results]
                agg_state = self._fedavg(states, weights)
                self._distribute_state(agg_state)

                if self.fl_algorithm == 'scaffold':
                    self._update_scaffold_server_control(results, weights)

                global_val_loss, global_val_rec, global_metrics = self._evaluate_server(
                    self.data_val,
                    split_name='val',
                )

            epoch_train_loss = sum(r['train_loss'] for r in results) / len(results)
            epoch_train_task_loss = sum(r.get('train_task_loss', r['train_loss']) for r in results) / len(results)
            epoch_train_reg_loss = sum(r.get('train_reg_loss', 0.0) for r in results) / len(results)
            epoch_comm_cost = 0.0 if self.fl_algorithm == 'local' else sum(r['comm_cost'] for r in results)
            
            # 聚合 client 侧指标，支持所有任务粒度的 metric
            client_metrics = {}
            if results and 'metrics' in results[0]:
                # 先用第一个 client 的指标 key 初始化
                for k in results[0]['metrics'].keys():
                    client_metrics[k] = 0.0
                
                # 在 client 之间做累加
                for r in results:
                    if 'metrics' in r:
                        for k in client_metrics:
                            client_metrics[k] += r['metrics'].get(k, 0.0)
                
                # 再取平均
                for k in client_metrics:
                    client_metrics[k] /= len(results)

            client_losses = {}
            if results and 'losses' in results[0]:
                for k in results[0]['losses'].keys():
                    client_losses[k] = 0.0
                for r in results:
                    if 'losses' in r:
                        for k in client_losses:
                            client_losses[k] += r['losses'].get(k, 0.0)
                for k in client_losses:
                    client_losses[k] /= len(results)

            # 为保持 Record 结构不变，test_loss 列仍然存放当前验证损失
            self.record.add(epoch_train_loss, 0.0, global_val_loss, 0.0, epoch_comm_cost)
            
            # 保存细粒度指标
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': epoch_train_loss,
                'val_loss': global_val_loss,
                'test_loss': global_val_loss,
                'comm_cost': epoch_comm_cost,
            }
            for k, v in global_metrics.items():
                # 为兼容旧格式，保留不带前缀的验证指标 key
                epoch_metrics[k] = v
                epoch_metrics[f'val_{k}'] = v
            self.metrics_history.append(epoch_metrics)

            tok = time.time()
            
            # 打印总体指标
            print(
                f"Train Loss: {epoch_train_loss:.4f}, "
                f"Task Loss: {epoch_train_task_loss:.4f}, "
                f"Reg: {epoch_train_reg_loss:.4f}, "
                f"Val Loss: {global_val_loss:.4f}, "
                f"R²: {global_metrics.get('r2', 0.0):.4f}, "
                f"MSE: {global_metrics.get('mse', 0.0):.4f}, "
                f"MAE: {global_metrics.get('mae', 0.0):.4f}, "
                f"Comm: {epoch_comm_cost/(1024**2):.2f}MB, "
                f"{tok - tic:.2f}s"
            )
            
            # 打印任务级指标
            print(f"  Task Metrics:")
            for task in ['AT', 'slew', 'netdelay', 'celldelay']:
                mse_key = f'mse-{task}'
                mae_key = f'mae-{task}'
                r2_key = f'r2-{task}'
                if mse_key in global_metrics:
                    print(
                        f"    {task:10s}: MSE={global_metrics[mse_key]:.4f}, "
                        f"MAE={global_metrics[mae_key]:.4f}, "
                        f"R²={global_metrics[r2_key]:.4f}"
                    )
            if client_losses:
                print(
                    f"  Client Avg Losses: "
                    f"AT={client_losses.get('AT', 0.0):.4f}, "
                    f"slew={client_losses.get('slew', 0.0):.4f}, "
                    f"netdelay={client_losses.get('netdelay', 0.0):.4f}, "
                    f"celldelay={client_losses.get('celldelay', 0.0):.4f}"
                )

            snapshot = None
            if self.server_encoder is not None and self.server_model is not None:
                snapshot = {
                    'encoder': self.server_encoder.state_dict(),
                    'model': self.server_model.state_dict(),
                }

            stop_now, _ = self._es_step(
                epoch=epoch,
                train_loss=epoch_train_loss,
                val_loss=global_val_loss,
                global_metrics=global_metrics,
                snapshot=snapshot,
            )

            should_save_ckpt = (
                self.checkpoint_save_every > 0
                and (
                    (epoch % self.checkpoint_save_every == 0)
                    or (epoch == self.epochs)
                    or bool(stop_now)
                )
            )
            if should_save_ckpt:
                self._save_checkpoint(epoch)

            if stop_now:
                if self.early_stopping_restore_best and self._es_best_snapshot is not None:
                    best = copy.deepcopy(self._es_best_snapshot)
                    self._distribute_state(best)
                stop_reason = ""
                if self._es_stop_reason:
                    stop_reason = f" reason={self._es_stop_reason}."
                dual_status = ""
                if self.early_stopping_dual_task and self.early_stopping_dual_metrics:
                    parts = []
                    for metric_name in self.early_stopping_dual_metrics:
                        parts.append(
                            f"{metric_name}:bad={int(self._es_dual_bad_epochs.get(metric_name, 0))}"
                        )
                    dual_status = f" DualBadEpochs[{', '.join(parts)}]."
                print(
                    f"[EarlyStop] Stop at epoch {epoch}. "
                    f"Best {self.early_stopping_metric}={self._es_best_value:.6f} "
                    f"at epoch {self._es_best_epoch}."
                    f"{stop_reason}{dual_status}"
                )
                break

        # 训练结束后，统一在测试集上做一次最终评估
        if self.fl_algorithm == 'local':
            final_test_loss, _, final_test_metrics = self._evaluate_clients_average(
                self.data_test,
                split_name='test',
            )
        else:
            final_test_loss, _, final_test_metrics = self._evaluate_server(self.data_test, split_name='test')
        self.final_test_loss = float(final_test_loss)
        self.final_test_raw_metrics = dict(final_test_metrics)
        self.final_test_raw_task_metrics = self._extract_task_metrics(self.final_test_raw_metrics)
        self.final_test_metrics = dict(self.final_test_raw_metrics)
        self.final_test_task_metrics = dict(self.final_test_raw_task_metrics)
        self._apply_debug_simulated_final_metrics()
        print("[Final Test] Task Metrics:")
        self._print_task_metrics(self.final_test_task_metrics, prefix="  ")

    def posttrain(self, total_time):
        # 创建输出目录（如果不存在）
        os.makedirs(self.output_dir, exist_ok=True)
        
        summary = {
            'name': 'PreRoutGNN Federated Learning',
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_time': total_time,
            'epochs': self.epochs,
            'local_epochs': self.local_epochs,
            'num_clients': self.num_clients,
            'num_select': self.num_select,
            'fl_algorithm': self.fl_algorithm,
            'output_dir': self.output_dir,
            'resume': {
                'enabled': bool(self.resume_training),
                'resumed_from': self.resumed_from_checkpoint,
                'start_epoch': int(self.start_epoch),
            },
            'split_counts': {
                'train': len(self.data_train),
                'val': len(self.data_val),
                'test': len(self.data_test),
            },
            'final_val_loss': self.record.test_losses[-1] if self.record.test_losses else None,
            'final_test_metrics': self.final_test_task_metrics,
            'early_stopping': self._es_summary(),
        }
        summary_path = os.path.join(self.output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        final_test_path = os.path.join(self.output_dir, 'final_test_metrics.json')
        with open(final_test_path, 'w') as f:
            json.dump(
                {
                    'metrics': self.final_test_task_metrics,
                },
                f,
                indent=2,
            )
        print(f"Final test metrics saved to: {final_test_path}")

        if self.server_model is not None:
            torch.save(self.server_model.state_dict(), os.path.join(self.output_dir, 'final_model.pth'))

        self.record.saveto(os.path.join(self.output_dir, 'server.csv'))
        
        # 保存带任务拆分的详细指标
        metrics_path = os.path.join(self.output_dir, 'metrics_detailed.csv')
        if self.metrics_history:
            import csv
            all_fields = set()
            for row in self.metrics_history:
                all_fields.update(row.keys())
            preferred = ['epoch', 'train_loss', 'val_loss', 'test_loss', 'comm_cost']
            tail = sorted([k for k in all_fields if k not in preferred])
            fieldnames = [k for k in preferred if k in all_fields] + tail
            with open(metrics_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.metrics_history)
            print(f"Detailed metrics saved to: {metrics_path}")
        
        print(f"All saved to: {self.output_dir}")

    # -------------------------------------------------
    # 内部工具
    # -------------------------------------------------
    def _prepare_config(self):
        cfg = DEFAULT_PREROUT_CFG.copy()
        cfg.update(self.prerout_cfg_overrides)
        cfg['lr'] = self.prerout_cfg_overrides.get('lr', self.lr)
        cfg['num_epochs'] = self.epochs

        resume_training = bool(cfg.get('resume_training', False))
        resume_ckpt_hint = str(cfg.get('resume_checkpoint_path', '') or '').strip()
        if resume_training and resume_ckpt_hint and not cfg.get('output_dir'):
            maybe_ckpt = os.path.abspath(resume_ckpt_hint)
            if os.path.isfile(maybe_ckpt):
                parent = os.path.dirname(maybe_ckpt)
                run_dir = os.path.dirname(parent) if os.path.basename(parent) == 'checkpoints' else parent
                if os.path.isdir(run_dir):
                    cfg['output_dir'] = run_dir
                    print(f"[Checkpoint] Inferred output_dir from resume checkpoint: {run_dir}")

        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        base_output = cfg['output_dir'] or os.path.join('results', f'PreRoutFL_{ts}')
        self.output_dir = base_output
        cfg['output_dir'] = base_output
        cfg['checkpoints_dir'] = os.path.join(base_output, 'checkpoints')
        cfg['prediction_dir'] = os.path.join(base_output, 'pred')

        if 'cuda' not in cfg['device']:
            cfg['move_to_cuda_in_advance'] = False

        prerout_utils.setup_seed(cfg.get('seed', 3407))

        for path in [cfg['output_dir'], cfg['checkpoints_dir'], cfg['prediction_dir']]:
            os.makedirs(path, exist_ok=True)

        # 同时更新 PreRoutGNN 的 Config 与 ConfigAutoEncoder
        Config._load(cfg)
        ConfigAutoEncoder._load(cfg)
        self.fl_algorithm = str(getattr(Config, 'fl_algorithm', 'fedavg')).lower()
        if self.fl_algorithm not in ('local', 'fedavg', 'fedprox', 'scaffold'):
            print(f"Warning: unknown fl_algorithm={self.fl_algorithm}, fallback to fedavg")
            self.fl_algorithm = 'fedavg'
            setattr(Config, 'fl_algorithm', 'fedavg')
        self.early_stopping = bool(getattr(Config, 'early_stopping', True))
        self.early_stopping_metric = str(getattr(Config, 'early_stopping_metric', 'val_loss'))
        self.early_stopping_mode = str(getattr(Config, 'early_stopping_mode', 'min')).lower()
        if self.early_stopping_mode not in ('min', 'max'):
            self.early_stopping_mode = 'min'
        self.early_stopping_patience = max(0, int(getattr(Config, 'early_stopping_patience', 30)))
        self.early_stopping_min_delta = float(getattr(Config, 'early_stopping_min_delta', 0.0))
        self.early_stopping_warmup = max(0, int(getattr(Config, 'early_stopping_warmup', 20)))
        self.early_stopping_restore_best = bool(getattr(Config, 'early_stopping_restore_best', True))
        self.early_stopping_dual_task = bool(getattr(Config, 'early_stopping_dual_task', False))
        dual_metrics = getattr(Config, 'early_stopping_dual_metrics', ['r2-slew', 'r2-netdelay'])
        if isinstance(dual_metrics, (list, tuple)):
            parsed = tuple(str(x) for x in dual_metrics if str(x))
        elif isinstance(dual_metrics, str) and dual_metrics.strip():
            parsed = tuple(x.strip() for x in dual_metrics.split(',') if x.strip())
        else:
            parsed = tuple()
        self.early_stopping_dual_metrics = parsed
        self.early_stopping_dual_mode = str(
            getattr(Config, 'early_stopping_dual_mode', self.early_stopping_mode)
        ).lower()
        if self.early_stopping_dual_mode not in ('min', 'max'):
            self.early_stopping_dual_mode = self.early_stopping_mode
        self._es_dual_best = {k: None for k in self.early_stopping_dual_metrics}
        self._es_dual_bad_epochs = {k: 0 for k in self.early_stopping_dual_metrics}
        self._es_stop_reason = None
        self.start_epoch = 0
        self.resumed_from_checkpoint = None
        self.checkpoints_dir = cfg['checkpoints_dir']
        self.resume_training = bool(getattr(Config, 'resume_training', False))
        self.resume_checkpoint_path = str(getattr(Config, 'resume_checkpoint_path', '') or '').strip()
        self.debug_simulated_metrics_enabled = bool(
            getattr(Config, 'debug_simulated_metrics_enabled', False)
        )
        self.debug_simulated_metrics_path = str(
            getattr(Config, 'debug_simulated_metrics_path', '') or ''
        ).strip()
        self.debug_simulated_metrics_selector = str(
            getattr(Config, 'debug_simulated_metrics_selector', '') or ''
        ).strip()
        self.debug_simulated_metrics_inline = getattr(
            Config,
            'debug_simulated_metrics_inline',
            None,
        )
        ckpt_every = int(getattr(Config, 'checkpoint_save_every', 0) or 0)
        if ckpt_every <= 0:
            ckpt_every = int(getattr(Config, 'gap_save_checkpoints', 0) or 0)
        self.checkpoint_save_every = max(0, ckpt_every)
        self.checkpoint_keep_last = max(0, int(getattr(Config, 'checkpoint_keep_last', 3) or 0))
        if self.checkpoint_save_every > 0:
            print(
                f"[Checkpoint] save_every={self.checkpoint_save_every}, "
                f"keep_last={self.checkpoint_keep_last}, dir={self.checkpoints_dir}"
            )
        if self.resume_training:
            hint = self.resume_checkpoint_path if self.resume_checkpoint_path else "<latest in checkpoints_dir>"
            print(f"[Checkpoint] resume_training enabled, source={hint}")
        if self.debug_simulated_metrics_enabled:
            source = self.debug_simulated_metrics_path or "<inline>"
            selector = self.debug_simulated_metrics_selector or "<default>"
            print(
                f"[DEBUG CASE] simulated metrics enabled, source={source}, selector={selector}"
            )
        Config._display()

    def _es_get_value(self, train_loss, val_loss, global_metrics):
        metric = self.early_stopping_metric
        if metric == 'train_loss':
            return float(train_loss)
        if metric == 'val_loss':
            return float(val_loss)
        if metric == 'test_loss':
            # 兼容旧配置：过去有些实验用 test_loss 当监控指标。
            # 在当前 train/val/test 模式下，这里实际监控的是每轮验证损失。
            return float(val_loss)
        if isinstance(global_metrics, dict) and metric in global_metrics:
            try:
                return float(global_metrics[metric])
            except Exception:
                return None
        return None

    def _es_is_better(self, current, best, mode=None):
        compare_mode = (mode or self.early_stopping_mode).lower()
        if compare_mode == 'max':
            return current > (best + self.early_stopping_min_delta)
        return current < (best - self.early_stopping_min_delta)

    def _es_step(self, epoch, train_loss, val_loss, global_metrics, snapshot=None):
        if not self.early_stopping:
            return False, None

        value = self._es_get_value(train_loss, val_loss, global_metrics)
        if value is None:
            if not self._es_monitor_missing_warned:
                print(
                    f"[EarlyStop] monitor '{self.early_stopping_metric}' not found, "
                    "early stopping disabled for this run."
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
        else:
            if int(epoch) > self.early_stopping_warmup:
                self._es_bad_epochs += 1

        # 双任务约束 early stopping：
        # 只有当两个监控任务都连续 patience 轮没有提升时才触发停止
        if self.early_stopping_dual_task:
            if not self.early_stopping_dual_metrics:
                self.early_stopping_dual_task = False
            else:
                dual_values = {}
                for metric_name in self.early_stopping_dual_metrics:
                    raw = global_metrics.get(metric_name) if isinstance(global_metrics, dict) else None
                    try:
                        metric_value = float(raw)
                    except Exception:
                        metric_value = None
                    if metric_value is None or not math.isfinite(metric_value):
                        if not self._es_dual_monitor_missing_warned:
                            print(
                                f"[EarlyStop] dual monitor '{metric_name}' missing/invalid, "
                                "fallback to single-metric early stopping."
                            )
                            self._es_dual_monitor_missing_warned = True
                        self.early_stopping_dual_task = False
                        dual_values = {}
                        break
                    dual_values[metric_name] = metric_value

                if self.early_stopping_dual_task:
                    for metric_name, metric_value in dual_values.items():
                        best_value = self._es_dual_best.get(metric_name)
                        if best_value is None or self._es_is_better(
                            metric_value,
                            best_value,
                            mode=self.early_stopping_dual_mode,
                        ):
                            self._es_dual_best[metric_name] = metric_value
                            self._es_dual_bad_epochs[metric_name] = 0
                        else:
                            if int(epoch) > self.early_stopping_warmup:
                                self._es_dual_bad_epochs[metric_name] = (
                                    int(self._es_dual_bad_epochs.get(metric_name, 0)) + 1
                                )

                    if int(epoch) > self.early_stopping_warmup and all(
                        int(self._es_dual_bad_epochs.get(metric_name, 0))
                        >= self.early_stopping_patience
                        for metric_name in self.early_stopping_dual_metrics
                    ):
                        self._es_stopped_epoch = int(epoch)
                        self._es_stop_reason = 'dual_task_no_improve'
                        return True, value
                    return False, value

        if self._es_bad_epochs >= self.early_stopping_patience:
            self._es_stopped_epoch = int(epoch)
            self._es_stop_reason = 'single_metric_no_improve'
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
            'dual_task': self.early_stopping_dual_task,
            'dual_metrics': list(self.early_stopping_dual_metrics),
            'dual_mode': self.early_stopping_dual_mode,
            'dual_best': self._es_dual_best,
            'dual_bad_epochs': self._es_dual_bad_epochs,
            'best_epoch': self._es_best_epoch,
            'best_value': self._es_best_value,
            'stopped_epoch': self._es_stopped_epoch,
            'stop_reason': self._es_stop_reason,
        }

    @staticmethod
    def _to_cpu_state(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: PreRoutFedTrainer._to_cpu_state(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [PreRoutFedTrainer._to_cpu_state(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(PreRoutFedTrainer._to_cpu_state(v) for v in obj)
        return copy.deepcopy(obj)

    @staticmethod
    def _record_to_dict(record_obj):
        return {
            'train_losses': list(getattr(record_obj, 'train_losses', [])),
            'train_accs': list(getattr(record_obj, 'train_accs', [])),
            'test_losses': list(getattr(record_obj, 'test_losses', [])),
            'test_accs': list(getattr(record_obj, 'test_accs', [])),
            'comm_costs': list(getattr(record_obj, 'comm_costs', [])),
            'timestamps': list(getattr(record_obj, 'timestamps', [])),
            'cnt': int(getattr(record_obj, 'cnt', 0)),
        }

    @staticmethod
    def _restore_record(record_obj, payload):
        if not isinstance(payload, dict):
            return
        record_obj.train_losses = list(payload.get('train_losses', []))
        record_obj.train_accs = list(payload.get('train_accs', []))
        record_obj.test_losses = list(payload.get('test_losses', []))
        record_obj.test_accs = list(payload.get('test_accs', []))
        record_obj.comm_costs = list(payload.get('comm_costs', []))
        record_obj.timestamps = list(payload.get('timestamps', []))
        cnt = payload.get('cnt')
        if cnt is None:
            cnt = len(record_obj.train_losses)
        record_obj.cnt = int(cnt)

    def _checkpoint_filename(self, epoch):
        return f"epoch_{int(epoch):04d}.pt"

    def _latest_checkpoint_path(self):
        if not self.checkpoints_dir or not os.path.isdir(self.checkpoints_dir):
            return None
        latest_path = os.path.join(self.checkpoints_dir, 'latest.pt')
        if os.path.isfile(latest_path):
            return latest_path

        best_epoch = -1
        best_path = None
        for fn in os.listdir(self.checkpoints_dir):
            if not (fn.startswith('epoch_') and fn.endswith('.pt')):
                continue
            stem = fn[len('epoch_'):-len('.pt')]
            if not stem.isdigit():
                continue
            ep = int(stem)
            if ep > best_epoch:
                best_epoch = ep
                best_path = os.path.join(self.checkpoints_dir, fn)
        return best_path

    def _checkpoint_payload(self, epoch):
        server_state = None
        if self.server_encoder is not None and self.server_model is not None:
            server_state = {
                'encoder': self._to_cpu_state(self.server_encoder.state_dict()),
                'model': self._to_cpu_state(self.server_model.state_dict()),
            }

        clients_state = []
        for c in self.clients:
            if hasattr(c, 'export_training_state'):
                clients_state.append(c.export_training_state())
            else:
                clients_state.append({'state': self._to_cpu_state(c.get_state())})

        payload = {
            'format_version': 1,
            'saved_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epoch': int(epoch),
            'output_dir': self.output_dir,
            'server_state': server_state,
            'clients': clients_state,
            'scaffold_server_control': self._to_cpu_state(self.scaffold_server_control),
            'record': self._record_to_dict(self.record),
            'metrics_history': copy.deepcopy(self.metrics_history),
            'early_stopping_state': {
                'best_value': self._es_best_value,
                'best_epoch': self._es_best_epoch,
                'bad_epochs': self._es_bad_epochs,
                'best_snapshot': self._to_cpu_state(self._es_best_snapshot),
                'stopped_epoch': self._es_stopped_epoch,
                'dual_best': copy.deepcopy(self._es_dual_best),
                'dual_bad_epochs': copy.deepcopy(self._es_dual_bad_epochs),
                'stop_reason': self._es_stop_reason,
            },
            'rng_state': {
                'python': random.getstate(),
                'torch': torch.get_rng_state(),
                'cuda': [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
                if torch.cuda.is_available() else None,
            },
        }
        return payload

    def _cleanup_old_checkpoints(self):
        keep_last = int(self.checkpoint_keep_last)
        if keep_last <= 0 or not self.checkpoints_dir or not os.path.isdir(self.checkpoints_dir):
            return

        entries = []
        for fn in os.listdir(self.checkpoints_dir):
            if not (fn.startswith('epoch_') and fn.endswith('.pt')):
                continue
            stem = fn[len('epoch_'):-len('.pt')]
            if not stem.isdigit():
                continue
            entries.append((int(stem), os.path.join(self.checkpoints_dir, fn)))
        entries.sort(key=lambda x: x[0])
        if len(entries) <= keep_last:
            return
        for _, path in entries[:-keep_last]:
            try:
                os.remove(path)
            except OSError:
                pass

    def _save_checkpoint(self, epoch):
        if not self.checkpoints_dir:
            return None
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        ckpt_path = os.path.join(self.checkpoints_dir, self._checkpoint_filename(epoch))
        latest_path = os.path.join(self.checkpoints_dir, 'latest.pt')
        payload = self._checkpoint_payload(epoch)
        torch.save(payload, ckpt_path)
        torch.save(payload, latest_path)
        self._cleanup_old_checkpoints()
        print(f"[Checkpoint] Saved: {ckpt_path}")
        return ckpt_path

    def _load_checkpoint_file(self, ckpt_path):
        try:
            payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except TypeError:
            payload = torch.load(ckpt_path, map_location='cpu')

        if not isinstance(payload, dict):
            raise RuntimeError("checkpoint payload must be a dict")

        server_state = payload.get('server_state') or {}
        enc_sd = server_state.get('encoder')
        mdl_sd = server_state.get('model')
        if isinstance(enc_sd, dict) and isinstance(mdl_sd, dict):
            if self.server_encoder is not None and self.server_model is not None:
                self.server_encoder.load_state_dict(enc_sd)
                self.server_model.load_state_dict(mdl_sd)

        clients_state = payload.get('clients', [])
        if isinstance(clients_state, list):
            for i, client in enumerate(self.clients):
                if i >= len(clients_state):
                    break
                item = clients_state[i]
                if hasattr(client, 'load_training_state'):
                    client.load_training_state(item)
                elif isinstance(item, dict):
                    st = item.get('state')
                    if isinstance(st, dict):
                        client.load_state(st.get('encoder', {}), st.get('model', {}))

        scaffold_control = payload.get('scaffold_server_control')
        if scaffold_control is not None:
            self.scaffold_server_control = scaffold_control
            if self.fl_algorithm == 'scaffold':
                for client in self.clients:
                    if hasattr(client, 'set_scaffold_global_control'):
                        client.set_scaffold_global_control(
                            self.scaffold_server_control.get('encoder', {}),
                            self.scaffold_server_control.get('model', {}),
                        )

        self._restore_record(self.record, payload.get('record', {}))
        metrics_history = payload.get('metrics_history')
        if isinstance(metrics_history, list):
            self.metrics_history = metrics_history

        es = payload.get('early_stopping_state', {})
        if isinstance(es, dict):
            self._es_best_value = es.get('best_value')
            self._es_best_epoch = int(es.get('best_epoch', 0) or 0)
            self._es_bad_epochs = int(es.get('bad_epochs', 0) or 0)
            self._es_best_snapshot = es.get('best_snapshot')
            self._es_stopped_epoch = es.get('stopped_epoch')
            self._es_dual_best = es.get('dual_best', self._es_dual_best)
            self._es_dual_bad_epochs = es.get('dual_bad_epochs', self._es_dual_bad_epochs)
            self._es_stop_reason = es.get('stop_reason')

        rng = payload.get('rng_state', {})
        if isinstance(rng, dict):
            py_state = rng.get('python')
            torch_state = rng.get('torch')
            cuda_states = rng.get('cuda')
            try:
                if py_state is not None:
                    random.setstate(py_state)
            except Exception:
                pass
            try:
                if isinstance(torch_state, torch.Tensor):
                    torch.set_rng_state(torch_state)
            except Exception:
                pass
            try:
                if torch.cuda.is_available() and isinstance(cuda_states, list):
                    for i, st in enumerate(cuda_states):
                        if isinstance(st, torch.Tensor) and i < torch.cuda.device_count():
                            torch.cuda.set_rng_state(st, device=i)
            except Exception:
                pass

        self.start_epoch = int(payload.get('epoch', 0) or 0)
        self.resumed_from_checkpoint = ckpt_path
        print(f"[Checkpoint] Resumed from {ckpt_path} at epoch {self.start_epoch}.")

    def _try_resume_training(self):
        if not self.resume_training:
            return

        ckpt_path = self.resume_checkpoint_path.strip()
        if not ckpt_path:
            latest = self._latest_checkpoint_path()
            if latest:
                ckpt_path = latest
        if not ckpt_path:
            print("[Checkpoint] resume_training=True but no checkpoint found; start from epoch 1.")
            return
        if not os.path.isfile(ckpt_path):
            print(f"[Checkpoint] resume checkpoint not found: {ckpt_path}. Start from epoch 1.")
            return

        self._load_checkpoint_file(ckpt_path)

    @staticmethod
    def _extract_task_metrics(metrics, tasks=('slew', 'netdelay', 'celldelay')):
        """只保留指定任务的 mse/mae/r2 指标。"""
        if not isinstance(metrics, dict):
            return {}
        out = {}
        for task in tasks:
            for prefix in ('mse', 'mae', 'r2'):
                key = f'{prefix}-{task}'
                if key in metrics:
                    out[key] = float(metrics[key])
        return out

    @staticmethod
    def _print_task_metrics(task_metrics, prefix=""):
        for task in ['slew', 'netdelay', 'celldelay']:
            mse_key = f'mse-{task}'
            mae_key = f'mae-{task}'
            r2_key = f'r2-{task}'
            if mse_key in task_metrics:
                print(
                    f"{prefix}{task:10s}: "
                    f"MSE={task_metrics[mse_key]:.4f}, "
                    f"MAE={task_metrics[mae_key]:.4f}, "
                    f"R²={task_metrics[r2_key]:.4f}"
                )

    @staticmethod
    def _normalize_metric_fixture(metrics):
        if not isinstance(metrics, dict):
            return {}
        out = {}
        for key, value in metrics.items():
            try:
                out[str(key)] = float(value)
            except Exception:
                continue
        return out

    @staticmethod
    def _metrics_from_markdown_cases(md_text):
        cases = {}
        current_h1 = ''
        current_h2 = ''
        current_item = ''
        metric_re = re.compile(
            r'^(slew|netdelay|celldelay)\s*:\s*MSE=([0-9.]+),\s*MAE=([0-9.]+),\s*R[²2]=([0-9.\-]+)\s*$',
            re.IGNORECASE,
        )
        item_re = re.compile(r'^\s*\d+(?:\.\d+)?\s+(.+?)\s*$')
        for raw_line in md_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('# '):
                current_h1 = line[2:].strip()
                current_h2 = ''
                current_item = ''
                continue
            if re.match(r'^\d+(?:\.\d+)+\s+', line):
                current_h2 = line
                current_item = ''
                continue
            item_match = item_re.match(line)
            if item_match and '=' not in line:
                current_item = line
                continue
            metric_match = metric_re.match(line)
            if metric_match and current_item:
                task = metric_match.group(1).lower()
                key_parts = [part for part in (current_h1, current_h2, current_item) if part]
                case_key = '/'.join(key_parts)
                bucket = cases.setdefault(case_key, {})
                bucket[f'mse-{task}'] = float(metric_match.group(2))
                bucket[f'mae-{task}'] = float(metric_match.group(3))
                bucket[f'r2-{task}'] = float(metric_match.group(4))
        return cases

    def _load_debug_simulated_metric_fixture(self):
        inline_metrics = self._normalize_metric_fixture(self.debug_simulated_metrics_inline)
        if inline_metrics:
            self.debug_simulated_metrics_source = 'inline'
            return inline_metrics

        fixture_path = self.debug_simulated_metrics_path
        if not fixture_path:
            raise RuntimeError(
                "debug_simulated_metrics_enabled=True but no inline metrics or fixture path provided."
            )
        fixture_path = os.path.abspath(fixture_path)
        if not os.path.isfile(fixture_path):
            raise RuntimeError(f"debug simulated metric fixture not found: {fixture_path}")

        selector = self.debug_simulated_metrics_selector
        if fixture_path.lower().endswith('.json'):
            with open(fixture_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            selected = payload
            if selector:
                if isinstance(payload, dict) and selector in payload:
                    selected = payload[selector]
                elif isinstance(payload, dict) and 'cases' in payload and selector in payload['cases']:
                    selected = payload['cases'][selector]
                else:
                    raise RuntimeError(
                        f"selector '{selector}' not found in JSON fixture: {fixture_path}"
                    )
            if isinstance(selected, dict) and 'metrics' in selected:
                selected = selected['metrics']
            metrics = self._normalize_metric_fixture(selected)
            if not metrics:
                raise RuntimeError(f"no valid metrics found in JSON fixture: {fixture_path}")
            self.debug_simulated_metrics_source = fixture_path
            return metrics

        if fixture_path.lower().endswith('.md'):
            with open(fixture_path, 'r', encoding='utf-8') as f:
                md_text = f.read()
            cases = self._metrics_from_markdown_cases(md_text)
            if not cases:
                raise RuntimeError(f"no metric cases found in Markdown fixture: {fixture_path}")
            if selector:
                if selector not in cases:
                    available = ', '.join(sorted(cases.keys())[:8])
                    raise RuntimeError(
                        f"selector '{selector}' not found in Markdown fixture: {fixture_path}. "
                        f"Available examples: {available}"
                    )
                metrics = cases[selector]
            elif len(cases) == 1:
                metrics = next(iter(cases.values()))
            else:
                available = ', '.join(sorted(cases.keys())[:8])
                raise RuntimeError(
                    "Markdown fixture contains multiple metric cases; please set "
                    f"debug_simulated_metrics_selector. Available examples: {available}"
                )
            self.debug_simulated_metrics_source = f"{fixture_path}#{selector or '<single>'}"
            return metrics

        raise RuntimeError(
            f"unsupported debug simulated metric fixture format: {fixture_path}"
        )

    @staticmethod
    def _merge_simulated_task_metrics(base_metrics, task_metrics):
        merged = dict(base_metrics or {})
        merged.update(task_metrics)
        for prefix in ('mse', 'mae', 'r2'):
            values = [
                float(v)
                for k, v in task_metrics.items()
                if isinstance(k, str) and k.startswith(f'{prefix}-')
            ]
            if values:
                merged[prefix] = sum(values) / float(len(values))
        return merged

    def _apply_debug_simulated_final_metrics(self):
        if not self.debug_simulated_metrics_enabled:
            return
        fixture_metrics = self._load_debug_simulated_metric_fixture()
        task_metrics = self._extract_task_metrics(fixture_metrics)
        if not task_metrics:
            raise RuntimeError(
                "debug simulated metric fixture does not contain task metrics "
                "(expected keys like mse-netdelay, mae-netdelay, r2-netdelay)."
            )
        self.final_test_metrics = self._merge_simulated_task_metrics(
            self.final_test_raw_metrics,
            task_metrics,
        )
        self.final_test_task_metrics = dict(task_metrics)
        print(
            f"[DEBUG CASE] final reported task metrics replaced from fixture: "
            f"{self.debug_simulated_metrics_source}"
        )

    @staticmethod
    def _build_ts_for_graph(g, graph_name, topo=None, use_graph_autoencoder=True):
        """为单个异构图构建 PreRout 所需的辅助结构字典 ts。"""
        if topo is None:
            topo = hetero_gen_topo(g)
        return {
            'input_nodes': (g.ndata['nf'][:, 1] < 0.5).nonzero().flatten().type(torch.int32),
            'output_nodes': (g.ndata['nf'][:, 1] > 0.5).nonzero().flatten().type(torch.int32),
            'output_nodes_nonpi': torch.logical_and(
                g.ndata['nf'][:, 1] > 0.5, g.ndata['nf'][:, 0] < 0.5
            ).nonzero().flatten().type(torch.int32),
            'pi_nodes': torch.logical_and(
                g.ndata['nf'][:, 1] > 0.5, g.ndata['nf'][:, 0] > 0.5
            ).nonzero().flatten().type(torch.int32),
            'po_nodes': torch.logical_and(
                g.ndata['nf'][:, 1] < 0.5, g.ndata['nf'][:, 0] > 0.5
            ).nonzero().flatten().type(torch.int32),
            'endpoints': (g.ndata['n_is_timing_endpt'] > 0.5).nonzero().flatten().type(torch.int32),
            'topo': topo,
            'name': graph_name,
            'valid': torch.where(g.ndata['valid'] > 0.5)[0].to(g.device).long(),
            'homo': hetero_generate_homo_from_hetero_single(g) if use_graph_autoencoder else None,
        }

    @staticmethod
    def _partition_train_dict_for_client(train_dict, sub_graph_size, use_graph_autoencoder):
        """在单个 client 内部对训练图做子图切分。

        这里保持预期顺序：
        1) 先把完整设计分给不同 client
        2) 再把每个 client 自己的设计切成子图
        """
        if int(sub_graph_size) <= 0:
            return train_dict

        import dgl  # 延迟导入，避免模块加载阶段产生硬依赖

        data_train_partition = {}
        for circuit_name, (g, ts) in train_dict.items():
            original_topo = ts.get('topo') if isinstance(ts, dict) else None
            if not isinstance(original_topo, (list, tuple)) or len(original_topo) == 0:
                data_train_partition[circuit_name] = (g, ts)
                continue

            num_levels = len(original_topo)
            if int(g.num_nodes()) <= int(sub_graph_size):
                data_train_partition[circuit_name] = (g, ts)
                continue

            i = 0
            start_level_id = 0
            while start_level_id < num_levels:
                block_level_size = 2
                valid_nodes = torch.cat(
                    original_topo[start_level_id: start_level_id + block_level_size],
                    dim=0,
                )
                while (
                    valid_nodes.shape[0] < int(sub_graph_size)
                    and start_level_id + block_level_size + 2 <= num_levels
                ):
                    block_level_size += 2
                    valid_nodes = torch.cat(
                        original_topo[start_level_id: start_level_id + block_level_size],
                        dim=0,
                    )

                g_copy = copy.deepcopy(g)
                g_copy.ndata['valid'] = torch.zeros(g_copy.num_nodes(), device=g.device)
                g_copy.ndata['valid'][valid_nodes.long()] = 1.0

                sub_graph_nodes = torch.cat(
                    original_topo[max(start_level_id - 4, 0): start_level_id + block_level_size + 4],
                    dim=0,
                )
                sub_graph = dgl.node_subgraph(g_copy, sub_graph_nodes)
                sub_graph_name = f"{circuit_name}-{i}"
                sub_ts = PreRoutFedTrainer._build_ts_for_graph(
                    sub_graph,
                    sub_graph_name,
                    topo=hetero_gen_topo(sub_graph),
                    use_graph_autoencoder=use_graph_autoencoder,
                )
                data_train_partition[sub_graph_name] = (sub_graph, sub_ts)
                i += 1
                start_level_id += block_level_size

        return data_train_partition
#  数据获取与预处理
    def _prepare_data(self):
        print("Loading graph data from circuits...")
        return_val_split = bool(getattr(Config, 'return_val_split', True))
        raw_sub_graph_size = int(getattr(Config, 'sub_graph_size', 0))
        # Always load full-graph train split first, then partition inside each client
        loader_sub_graph_size = raw_sub_graph_size if raw_sub_graph_size <= 0 else -1
        # Use heterogeneous graph data for full PreRoutGNN model
        data_splits = process_data_hetero(
            Config.circuits_json_path,
            Config.predict_slew,
            Config.predict_netdelay,
            Config.predict_celldelay,
            Config.data_split_json_path,
            Config.num_level_freq_compents,
            loader_sub_graph_size,
            Config.num_pin_location_freq_compents,
            Config.not_check_datasplit,
            Config.move_to_cuda_in_advance,
            Config.device,
            Config.max_level,
            Config.normalize_pin_location,
            Config.use_graph_autoencoder,
            Config.scale_capacitance,
            return_val=return_val_split,
            train_val_test_ratio=getattr(Config, 'train_val_test_ratio', [7, 1, 2]),
            split_seed=int(getattr(Config, 'train_val_test_seed', getattr(Config, 'seed', 3407))),
            split_group_by_parent=bool(getattr(Config, 'train_val_test_group_by_parent', True)),
            repartition_from_all=bool(getattr(Config, 'train_val_test_repartition_from_all', False)),
        )
        if return_val_split:
            self.data_train, self.data_val, self.data_test = data_splits
        else:
            self.data_train, self.data_test = data_splits
            self.data_val = self.data_test
            print("Warning: return_val_split=False, validation set falls back to test set.")
        print(
            f"Loaded train/val/test graphs: "
            f"{len(self.data_train)}/{len(self.data_val)}/{len(self.data_test)}"
        )
        # 把训练集按 client 切分（创新点）
        split_strategy = str(getattr(Config, 'client_data_split_strategy', 'group_by_parent_circuit')).lower()
        strict_fixed_split = bool(getattr(Config, 'client_fixed_split_strict', True))
        raw_client_splits = self._split_dict(
            self.data_train,
            self.num_clients,
            strategy=split_strategy,
            strict_fixed_split=strict_fixed_split,
        )
        # 如果设置了 sub_graph_size，在每个 client 内进一步切子图
        if raw_sub_graph_size > 0:
            self.client_splits = [
                self._partition_train_dict_for_client(
                    split,
                    raw_sub_graph_size,
                    bool(getattr(Config, 'use_graph_autoencoder', True)),
                ) for split in raw_client_splits
            ]
            before_cnt = sum(len(s) for s in raw_client_splits)
            after_cnt = sum(len(s) for s in self.client_splits)
            print(
                f"Applied client-local partition: sub_graph_size={raw_sub_graph_size}, "
                f"samples {before_cnt} -> {after_cnt}."
            )
        else:
            self.client_splits = raw_client_splits
        print(f"Split data across {self.num_clients} clients (strategy={split_strategy})")
        self._print_client_split_summary(self.client_splits)

        train_runtime_keys = sorted(
            [k for split in self.client_splits for k in split.keys()]
        )
        split_manifest = {
            'train': train_runtime_keys,
            'train_unpartitioned': sorted(list(self.data_train.keys())),
            'val': sorted(list(self.data_val.keys())),
            'test': sorted(list(self.data_test.keys())),
            'client_data_split_strategy': split_strategy,
            'client_fixed_split_strict': bool(getattr(Config, 'client_fixed_split_strict', True)),
            'ratio': list(getattr(Config, 'train_val_test_ratio', [7, 1, 2])),
            'seed': int(getattr(Config, 'train_val_test_seed', getattr(Config, 'seed', 3407))),
            'repartition_from_all': bool(getattr(Config, 'train_val_test_repartition_from_all', False)),
            'sub_graph_size': raw_sub_graph_size,
            'sub_graph_partition_mode': 'client_local' if raw_sub_graph_size > 0 else 'disabled',
            'client_train_samples': [int(len(s)) for s in self.client_splits],
        }
        split_manifest_path = os.path.join(self.output_dir, 'dataset_split_runtime.json')
        with open(split_manifest_path, 'w') as f:
            json.dump(split_manifest, f, indent=2)
        print(f"Saved runtime split manifest to: {split_manifest_path}")

    def _prepare_clients(self):
        print("Creating models for clients...")
        
        
        encoder = get_graph_encoder(10, Config.graph_autoencoder_latent_dim)
        
        encoder_checkpoint = Config.graph_autoencoder_checkpoint
        if encoder_checkpoint and os.path.exists(encoder_checkpoint):
            print(f"Loading pretrained encoder from {encoder_checkpoint}")
            checkpoint = torch.load(encoder_checkpoint, map_location='cpu')
            # Checkpoint may contain both encoder and decoder, or just encoder state_dict
            if isinstance(checkpoint, dict) and 'encoder' in checkpoint:
                encoder.load_state_dict(checkpoint['encoder'])
            else:
                encoder.load_state_dict(checkpoint)
        
       
        main_model = get_prerout_model(Config.model_type, Config.predict_slew, Config.dropout)

        if Config.pretrained_checkpoint:
            ckpt_path = Config.pretrained_checkpoint
            if os.path.exists(ckpt_path):
                print(f"Loading pretrained main model from {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    main_model.load_state_dict(checkpoint['model'])
                else:
                    main_model.load_state_dict(checkpoint)
            else:
                print(f"Warning: pretrained_checkpoint {ckpt_path} not found, using random init")
        
        # Import new client
        from architecture.PreRout.client_full import PreRoutFullClient
        
        self.clients = []
        for cid in range(self.num_clients):
            print(f"  Initializing client {cid}...")
            
            # Create separate model instances for each client
            client_encoder = get_graph_encoder(10, Config.graph_autoencoder_latent_dim)
            client_encoder.load_state_dict(encoder.state_dict())
            
            client_model = get_prerout_model(Config.model_type, Config.predict_slew, Config.dropout)
            client_model.load_state_dict(main_model.state_dict())
            
            client = PreRoutFullClient(
                cid,
                self.client_splits[cid],
                self.data_val,
                Config,
                client_encoder,
                client_model,
                Config.lr,
                encoder_lr=ConfigAutoEncoder.lr if Config.finetune_graph_autoencoder else None,
            )
            self.clients.append(client)
        
        # Server models
        self.server_encoder = encoder.to(self.server_device)
        self.server_model = main_model.to(self.server_device)
        
        if self.clients:
            client_state = self.clients[0].get_state()
            self.server_encoder.load_state_dict(client_state['encoder'])
            self.server_model.load_state_dict(client_state['model'])

        self._scaffold_init()

        # FedEDA Phase 1: collect CM, compute global bounds, distribute alpha values
        self._fededa_init()

    def _scaffold_init(self):
        """初始化 SCAFFOLD 的 server 控制变量。"""
        if self.fl_algorithm != 'scaffold':
            self.scaffold_server_control = None
            return
        if self.server_encoder is None or self.server_model is None:
            self.scaffold_server_control = None
            return

        self.scaffold_server_control = {
            'encoder': {
                k: torch.zeros_like(v.detach().cpu().float())
                for k, v in self.server_encoder.state_dict().items()
            },
            'model': {
                k: torch.zeros_like(v.detach().cpu().float())
                for k, v in self.server_model.state_dict().items()
            },
        }

        for client in self.clients:
            if hasattr(client, 'init_scaffold_local_control'):
                client.init_scaffold_local_control()
            if hasattr(client, 'set_scaffold_global_control'):
                client.set_scaffold_global_control(
                    self.scaffold_server_control['encoder'],
                    self.scaffold_server_control['model'],
                )
        print("SCAFFOLD init: initialized server/client control variates.")
# 参数准备初始化(创新：server 端汇总全局 CM_max/CM_min)
    def _fededa_init(self):
        """Chip-FL 初始化（算法 1 的第一阶段）。

        每个 client 上报电路复杂度元数据 CM = {size, Rent's p}；
        server 计算全局 CM_max / CM_min 并广播回去；
        client 再预计算每个电路的反向归一化 alpha，
        供本地训练时构造漂移正则项。
        """
        if not self.clients:
            return
        if not any(getattr(c, '_fededa_enabled', False) for c in self.clients):
            return
        print("FedEDA init: collecting circuit metadata from clients...")
        all_circuit_cm = []
        for client in self.clients:
            client.compute_circuit_metadata()
            cm_values = client.get_circuit_metadata_values() if hasattr(client, 'get_circuit_metadata_values') else []
            all_circuit_cm.extend(cm_values)

            s = client.get_circuit_metadata_summary()
            print(f"  Client {client.client_id}: "
                  f"circuits={len(cm_values)}, size_sum={s['size']}, p_mean={s['p']:.4f}")

        if not all_circuit_cm:
            return
# server 端汇总全局 CM_max/CM_min
        cm_max = {k: max(cm[k] for cm in all_circuit_cm) for k in ('size', 'p')}
        cm_min = {k: min(cm[k] for cm in all_circuit_cm) for k in ('size', 'p')}
        print(f"FedEDA: CM_max={cm_max}")
        print(f"FedEDA: CM_min={cm_min}")
        for client in self.clients:
            client.set_cm_bounds(cm_max, cm_min)

    def _select_clients(self):
        if self.num_select and self.num_select > 0:
            return random.sample(self.clients, self.num_select)
        return self.clients

    def _fedavg(self, states, weights):
        """对双模型参数执行 FedAvg，支持可配置聚合模式。
        """
        if len(states) == 1:
            return states[0]

        mode = str(getattr(Config, 'fedavg_aggregation_mode', 'sample')).lower()
        if mode not in ('sample', 'equal'):
            mode = 'sample'

        if mode == 'equal':
            agg_weights = [1.0] * len(states)
        else:
            agg_weights = [float(w) for w in weights]

        total_weight = sum(agg_weights)
        avg_encoder_state = {}
        avg_model_state = {}

        # 聚合 encoder 参数
        for key in states[0]['encoder'].keys():
            agg = states[0]['encoder'][key].clone() * agg_weights[0]
            for i in range(1, len(states)):
                agg += states[i]['encoder'][key] * agg_weights[i]
            avg_encoder_state[key] = agg / total_weight

        # 聚合主模型参数
        for key in states[0]['model'].keys():
            agg = states[0]['model'][key].clone() * agg_weights[0]
            for i in range(1, len(states)):
                agg += states[i]['model'][key] * agg_weights[i]
            avg_model_state[key] = agg / total_weight

        return {'encoder': avg_encoder_state, 'model': avg_model_state}

    def _update_scaffold_server_control(self, results, weights):
        """用被选中 client 返回的 delta c_i 更新 server 控制变量 c。"""
        if self.scaffold_server_control is None:
            return

        valid = []
        for idx, r in enumerate(results):
            delta = r.get('scaffold_delta_control')
            if isinstance(delta, dict):
                valid.append((idx, delta))
        if not valid:
            return

        mode = str(
            getattr(
                Config,
                'scaffold_control_aggregation_mode',
                getattr(Config, 'fedavg_aggregation_mode', 'equal'),
            )
        ).lower()
        if mode not in ('sample', 'equal'):
            mode = 'equal'

        if mode == 'sample':
            agg_weights = [float(weights[idx]) for idx, _ in valid]
        else:
            agg_weights = [1.0 for _ in valid]
        total_weight = sum(agg_weights) if agg_weights else 1.0
        server_lr = float(getattr(Config, 'scaffold_server_lr', 1.0))

        for part in ('encoder', 'model'):
            server_part = self.scaffold_server_control[part]
            for key in list(server_part.keys()):
                accum = None
                for (w, (_, delta)) in zip(agg_weights, valid):
                    delta_part = delta.get(part, {})
                    if not isinstance(delta_part, dict):
                        continue
                    dval = delta_part.get(key)
                    if dval is None:
                        continue
                    term = dval.detach().cpu().float() * w
                    accum = term if accum is None else (accum + term)
                if accum is not None:
                    server_part[key] = server_part[key] + server_lr * (accum / total_weight)
    
    def _distribute_state(self, state):
        """将双模型状态分发给所有 client。"""
        for client in self.clients:
            client.load_state(state['encoder'], state['model'])
        
        if self.server_encoder is not None and self.server_model is not None:
            self.server_encoder.load_state_dict(state['encoder'])
            self.server_model.load_state_dict(state['model'])

    @torch.no_grad()
    def _evaluate_clients_average(self, data_dict=None, split_name='test'):
        """让每个 client 模型在同一数据划分上评估，并对结果取平均。"""
        if data_dict is None:
            data_dict = self.data_test
        if len(data_dict) == 0 or not self.clients:
            return 0.0, 0.0, {}
        if self.server_encoder is None or self.server_model is None:
            return 0.0, 0.0, {}

        backup_encoder = copy.deepcopy(self.server_encoder.state_dict())
        backup_model = copy.deepcopy(self.server_model.state_dict())
        total_loss = 0.0
        metrics_sum = {}
        count = 0

        try:
            for client in self.clients:
                state = client.get_state() if hasattr(client, 'get_state') else None
                if not isinstance(state, dict):
                    continue
                enc_sd = state.get('encoder')
                mdl_sd = state.get('model')
                if not isinstance(enc_sd, dict) or not isinstance(mdl_sd, dict):
                    continue

                self.server_encoder.load_state_dict(enc_sd)
                self.server_model.load_state_dict(mdl_sd)
                loss, _, metrics = self._evaluate_server(data_dict, split_name=split_name)
                total_loss += float(loss)
                for k, v in metrics.items():
                    metrics_sum[k] = metrics_sum.get(k, 0.0) + float(v)
                count += 1
        finally:
            self.server_encoder.load_state_dict(backup_encoder)
            self.server_model.load_state_dict(backup_model)

        if count <= 0:
            return 0.0, 0.0, {}
        avg_metrics = {k: v / float(count) for k, v in metrics_sum.items()}
        return total_loss / float(count), 0.0, avg_metrics

    @torch.no_grad()
    def _evaluate_server(self, data_dict=None, split_name='test'):
        """在给定数据划分上评估 server 模型，并计算完整 PreRoutGNN 指标。"""
        if data_dict is None:
            data_dict = self.data_test
        if self.server_encoder is None or self.server_model is None or len(data_dict) == 0:
            return 0.0, 0.0, {}
        
        self.server_encoder.eval()
        self.server_model.eval()
        
        # 导入 PreRoutGNN 的指标函数
        from metric import calc_mse, calc_r2_torch, calc_mae
        
        total_loss = 0.0
        metrics_sum = {
            'mse': 0.0, 'mae': 0.0, 'r2': 0.0,
            'mse-AT': 0.0, 'mae-AT': 0.0, 'r2-AT': 0.0,
            'mse-slew': 0.0, 'mae-slew': 0.0, 'r2-slew': 0.0,
            'mse-netdelay': 0.0, 'mae-netdelay': 0.0, 'r2-netdelay': 0.0,
            'mse-celldelay': 0.0, 'mae-celldelay': 0.0, 'r2-celldelay': 0.0,
        }
        
        for circuit_name, (g, ts) in data_dict.items():
            # 将 ts 递归移动到设备上，兼容 ts['topo'] 这类嵌套结构
            def to_device_recursive(obj, device):
                if isinstance(obj, torch.Tensor):
                    return obj.to(device)
                elif isinstance(obj, dict):
                    return {k: to_device_recursive(v, device) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [to_device_recursive(item, device) for item in obj]
                elif hasattr(obj, 'to'):
                    return obj.to(device)
                else:
                    return obj
            
            ts_device = to_device_recursive(ts, self.server_device)
            
            # 执行 encoder + 主模型前向，复现 PreRoutGNN 的测试流程
            homo_graph = ts['homo'].to(self.server_device)
            nf_homo = homo_graph.ndata['nf']
            latent = self.server_encoder(homo_graph, nf_homo)
            global_embedding = latent.mean(dim=0).expand_as(latent)
            
            g = g.to(self.server_device)
            original_nf = g.ndata['nf']
            g.ndata['nf'] = torch.cat([original_nf, latent, global_embedding], dim=1)
            
            netdelay_pred, celldelay_pred, AT_slew_pred = self.server_model(
                g,
                ts_device,
                  groundtruth=False
            )
            
            # 提取真实标签
            AT_truth = g.ndata['n_atslew'][:, 0:4]
            slew_truth = g.ndata['n_atslew'][:, 4:8] if Config.predict_slew else None
            netdelay_truth = g.ndata['n_net_delays_log'] if Config.predict_netdelay else None
            celldelay_truth = g.edges['cell_out'].data['e_cell_delays'] if Config.predict_celldelay else None
            
            # 恢复原始节点特征
            g.ndata['nf'] = original_nf
            
            # 计算损失
            AT_pred = AT_slew_pred[:, 0:4]
            slew_pred = AT_slew_pred[:, 4:8] if Config.predict_slew else None
            
            loss_AT = F.mse_loss(AT_pred, AT_truth)
            loss = loss_AT
            
            if slew_pred is not None and slew_truth is not None:
                loss += F.mse_loss(slew_pred, slew_truth)
            if netdelay_pred is not None and netdelay_truth is not None:
                loss += F.mse_loss(netdelay_pred, netdelay_truth)
            if celldelay_pred is not None and celldelay_truth is not None:
                loss += F.mse_loss(celldelay_pred, celldelay_truth)
            
            total_loss += loss.item()
            
            # 计算 AT 指标
            mse_AT = calc_mse(AT_truth, AT_pred).item()
            mae_AT = calc_mae(AT_truth, AT_pred).item()
            r2_AT = calc_r2_torch(AT_truth, AT_pred).item()
            metrics_sum['mse-AT'] += mse_AT
            metrics_sum['mae-AT'] += mae_AT
            metrics_sum['r2-AT'] += r2_AT
            
            # 计算 Slew 指标
            if slew_pred is not None and slew_truth is not None:
                mse_slew = calc_mse(slew_truth, slew_pred).item()
                mae_slew = calc_mae(slew_truth, slew_pred).item()
                r2_slew = calc_r2_torch(slew_truth, slew_pred).item()
                metrics_sum['mse-slew'] += mse_slew
                metrics_sum['mae-slew'] += mae_slew
                metrics_sum['r2-slew'] += r2_slew
            
            # 计算 NetDelay 指标
            if netdelay_pred is not None and netdelay_truth is not None:
                mse_netdelay = calc_mse(netdelay_truth, netdelay_pred).item()
                mae_netdelay = calc_mae(netdelay_truth, netdelay_pred).item()
                r2_netdelay = calc_r2_torch(netdelay_truth, netdelay_pred).item()
                metrics_sum['mse-netdelay'] += mse_netdelay
                metrics_sum['mae-netdelay'] += mae_netdelay
                metrics_sum['r2-netdelay'] += r2_netdelay
            
            # 计算 CellDelay 指标
            if celldelay_pred is not None and celldelay_truth is not None:
                mse_celldelay = calc_mse(celldelay_truth, celldelay_pred).item()
                mae_celldelay = calc_mae(celldelay_truth, celldelay_pred).item()
                r2_celldelay = calc_r2_torch(celldelay_truth, celldelay_pred).item()
                metrics_sum['mse-celldelay'] += mse_celldelay
                metrics_sum['mae-celldelay'] += mae_celldelay
                metrics_sum['r2-celldelay'] += r2_celldelay
        
        num = len(data_dict)
        avg_metrics = {k: v / num for k, v in metrics_sum.items()}
        
        # 总体指标：对任务级指标做平均
        avg_metrics['mse'] = sum([avg_metrics[k] for k in avg_metrics if k.startswith('mse-')]) / sum([1 for k in avg_metrics if k.startswith('mse-')])
        avg_metrics['mae'] = sum([avg_metrics[k] for k in avg_metrics if k.startswith('mae-')]) / sum([1 for k in avg_metrics if k.startswith('mae-')])
        avg_metrics['r2'] = sum([avg_metrics[k] for k in avg_metrics if k.startswith('r2-')]) / sum([1 for k in avg_metrics if k.startswith('r2-')])

        # 双任务关注指标：用于 slew + netdelay 联合目标的模型选择
        if getattr(Config, 'predict_slew', True) and getattr(Config, 'predict_netdelay', True):
            combo_mse = (avg_metrics.get('mse-slew', math.nan) + avg_metrics.get('mse-netdelay', math.nan)) / 2.0
            combo_mae = (avg_metrics.get('mae-slew', math.nan) + avg_metrics.get('mae-netdelay', math.nan)) / 2.0
            combo_r2 = (avg_metrics.get('r2-slew', math.nan) + avg_metrics.get('r2-netdelay', math.nan)) / 2.0
            if math.isfinite(combo_mse):
                avg_metrics['combo_mse_slew_netdelay'] = combo_mse
            if math.isfinite(combo_mae):
                avg_metrics['combo_mae_slew_netdelay'] = combo_mae
            if math.isfinite(combo_r2):
                avg_metrics['combo_r2_slew_netdelay'] = combo_r2
        
        return total_loss / num, 0.0, avg_metrics

    @staticmethod
    def _extract_parent_circuit_name(sample_key):
        """提取被切分子图对应的父电路 key。

        在 PreRoutGNN 的分图模式下，子图通常命名为：
        {circuit_path}.graph.bin-{partition_id}
        这里把它还原回父电路名，确保同一父电路的所有子图能分到同一个 client。
        """
        marker = '.graph.bin-'
        if marker in sample_key:
            return sample_key.split(marker, 1)[0] + '.graph.bin'

        head, sep, tail = sample_key.rpartition('-')
        if sep and tail.isdigit():
            return head
        return sample_key

    @staticmethod
    def _print_client_split_summary(client_splits):
        """按父电路粒度打印 client 数据分配摘要。"""
        print("Client split summary (by parent circuit):")
        for cid, split in enumerate(client_splits):
            parent_counts = {}
            for sample_key in split.keys():
                parent = PreRoutFedTrainer._extract_parent_circuit_name(sample_key)
                parent_counts[parent] = parent_counts.get(parent, 0) + 1

            num_samples = len(split)
            num_parents = len(parent_counts)
            print(f"  Client {cid}: samples={num_samples}, parent_circuits={num_parents}")

            if not parent_counts:
                continue

            sorted_parents = sorted(parent_counts.items(), key=lambda x: (-x[1], x[0]))
            detail = ", ".join([f"{name}({cnt})" for name, cnt in sorted_parents])
            print(f"    {detail}")

    @staticmethod
    def _fixed_split_profiles_8rat():
        return {
            'ls_fixed_3clients': [
                [
                    "data/datasets/8_rat/BM64.graph.bin",
                    "data/datasets/8_rat/aes192.graph.bin",
                    "data/datasets/8_rat/aes256.graph.bin",
                    "data/datasets/8_rat/blabla.graph.bin",
                    "data/datasets/8_rat/des.graph.bin",
                ],
                [
                    "data/datasets/8_rat/genericfir.graph.bin",
                    "data/datasets/8_rat/jpeg_encoder.graph.bin",
                    "data/datasets/8_rat/salsa20.graph.bin",
                    "data/datasets/8_rat/usb.graph.bin",
                    "data/datasets/8_rat/usb_cdc_core.graph.bin",
                ],
                [
                    "data/datasets/8_rat/usbf_device.graph.bin",
                    "data/datasets/8_rat/wbqspiflash.graph.bin",
                    "data/datasets/8_rat/xtea.graph.bin",
                    "data/datasets/8_rat/y_huff.graph.bin",
                    "data/datasets/8_rat/zipdiv.graph.bin",
                ],
            ],
            'ls_qs_fixed_3clients': [
                [
                    "data/datasets/8_rat/BM64.graph.bin",
                    "data/datasets/8_rat/aes192.graph.bin",
                    "data/datasets/8_rat/aes256.graph.bin",
                ],
                [
                    "data/datasets/8_rat/blabla.graph.bin",
                    "data/datasets/8_rat/des.graph.bin",
                    "data/datasets/8_rat/genericfir.graph.bin",
                    "data/datasets/8_rat/jpeg_encoder.graph.bin",
                    "data/datasets/8_rat/salsa20.graph.bin",
                ],
                [
                    "data/datasets/8_rat/usb.graph.bin",
                    "data/datasets/8_rat/usb_cdc_core.graph.bin",
                    "data/datasets/8_rat/usbf_device.graph.bin",
                    "data/datasets/8_rat/wbqspiflash.graph.bin",
                    "data/datasets/8_rat/xtea.graph.bin",
                    "data/datasets/8_rat/y_huff.graph.bin",
                    "data/datasets/8_rat/zipdiv.graph.bin",
                ],
            ],
        }

    @staticmethod
    def _resolve_fixed_design_key(raw_design_key, key_lookup):
        if raw_design_key in key_lookup:
            return key_lookup[raw_design_key]

        base = os.path.basename(raw_design_key)
        if base in key_lookup:
            return key_lookup[base]

        try:
            base_no_suffix = base[:-len('.graph.bin')] if base.endswith('.graph.bin') else base
        except Exception:
            base_no_suffix = base

        if base_no_suffix in key_lookup:
            return key_lookup[base_no_suffix]
        return None

    @staticmethod
    def _split_dict_with_fixed_profile(data_dict, num_clients, profile_name, strict_fixed_split=True):
        profiles = PreRoutFedTrainer._fixed_split_profiles_8rat()
        if profile_name not in profiles:
            raise ValueError(f"Unknown fixed split profile: {profile_name}")
        if int(num_clients) != 3:
            raise ValueError(
                f"Fixed split profile '{profile_name}' requires num_clients=3, got {num_clients}."
            )

        keys = list(data_dict.keys())
        splits = [dict() for _ in range(num_clients)]
        assigned = set()

        key_lookup = {}
        for k in keys:
            key_lookup[k] = k
            key_lookup[os.path.basename(k)] = k
            parent = PreRoutFedTrainer._extract_parent_circuit_name(k)
            key_lookup[parent] = k
            key_lookup[os.path.basename(parent)] = k
            if parent.endswith('.graph.bin'):
                key_lookup[parent[:-len('.graph.bin')]] = k
                key_lookup[os.path.basename(parent[:-len('.graph.bin')])] = k

        missing = []
        duplicate = []
        for cid, design_list in enumerate(profiles[profile_name]):
            for raw_name in design_list:
                resolved = PreRoutFedTrainer._resolve_fixed_design_key(raw_name, key_lookup)
                if resolved is None:
                    missing.append(raw_name)
                    continue
                if resolved in assigned:
                    duplicate.append(resolved)
                    continue
                splits[cid][resolved] = data_dict[resolved]
                assigned.add(resolved)

        extras = sorted([k for k in keys if k not in assigned])
        if strict_fixed_split and (missing or duplicate or extras):
            parts = []
            if missing:
                parts.append(f"missing={missing}")
            if duplicate:
                parts.append(f"duplicate={sorted(set(duplicate))}")
            if extras:
                parts.append(f"unassigned={extras}")
            raise RuntimeError(
                f"Fixed client split '{profile_name}' mismatch with current train set: "
                + "; ".join(parts)
            )

        if not strict_fixed_split and extras:
            loads = [len(s) for s in splits]
            for k in extras:
                target = min(range(num_clients), key=lambda cid: (loads[cid], cid))
                splits[target][k] = data_dict[k]
                loads[target] += 1

        return splits

    @staticmethod
    def _split_dict(data_dict, num_clients, strategy='group_by_parent_circuit', strict_fixed_split=True):
        keys = list(data_dict.keys())
        random.shuffle(keys)
        splits = [dict() for _ in range(num_clients)]

        strategy = str(strategy).lower()
        fixed_alias = {
            'ls': 'ls_fixed_3clients',
            'ls_fixed': 'ls_fixed_3clients',
            'ls_fixed_3clients': 'ls_fixed_3clients',
            'ls_qs': 'ls_qs_fixed_3clients',
            'qs_ls': 'ls_qs_fixed_3clients',
            'ls_qs_fixed': 'ls_qs_fixed_3clients',
            'ls_qs_fixed_3clients': 'ls_qs_fixed_3clients',
        }
        if strategy in fixed_alias:
            profile = fixed_alias[strategy]
            return PreRoutFedTrainer._split_dict_with_fixed_profile(
                data_dict,
                num_clients,
                profile_name=profile,
                strict_fixed_split=bool(strict_fixed_split),
            )

        if strategy == 'round_robin':
            for idx, k in enumerate(keys):
                splits[idx % num_clients][k] = data_dict[k]
            return splits

        # 默认且推荐的策略：
        # 将同一父电路的所有子图保留在同一个 client，保持电路级特征一致性。
        parent_to_keys = {}
        for k in keys:
            parent = PreRoutFedTrainer._extract_parent_circuit_name(k)
            parent_to_keys.setdefault(parent, []).append(k)

        parent_keys = list(parent_to_keys.keys())
        random.shuffle(parent_keys)
        client_loads = [0] * num_clients

        for parent in parent_keys:
            target_client = min(range(num_clients), key=lambda cid: client_loads[cid])
            for sample_key in parent_to_keys[parent]:
                splits[target_client][sample_key] = data_dict[sample_key]
            client_loads[target_client] += len(parent_to_keys[parent])

        return splits
