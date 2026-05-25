# PreRoutGNN 完整模型的联邦客户端实现
# 文件定位：
# - 这是“单个联邦客户端”的核心实现。
# - 上游由 trainer.py 创建 client，并在每轮训练前下发全局模型状态。
# - 下游负责：
#   1) 用 PreRoutGNN 的 encoder + main_model 完成前向；
#   2) 计算多任务损失与评估指标；
#   3) 在本地训练时叠加 FedEDA / FedProx / SCAFFOLD 约束；
#   4) 把更新后的本地模型返回给 trainer.py 聚合。

import math
import copy
import csv
import json
import torch
from torch.nn import functional as F
from common import find_device, get_model_size

# 导入 PreRoutGNN 的指标函数。
# trainer.py 负责调度联邦流程，这里负责单客户端的具体数值计算。
import sys
import os
PREROUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'PreRoutGNN'))
if PREROUT_ROOT not in sys.path:
    sys.path.insert(0, PREROUT_ROOT)

from metric import calc_mse, calc_r2_torch, calc_r2_flatten, calc_mae


class PreRoutFullClient:
    """完整 PreRoutGNN 双模型结构的联邦客户端。"""
    
    def __init__(self, client_id, train_data, test_data, cfg, encoder, main_model, lr, encoder_lr=None):
        # -------------------------------------------------
        # 一、客户端基础状态与优化器初始化
        # -------------------------------------------------
        self.client_id = client_id
        self.train_data = train_data
        self.test_data = test_data
        self.cfg = cfg
        self.device = find_device()

        # 双模型结构：encoder + PreRoutGNN 主模型
        self.encoder = encoder.to(self.device)
        self.main_model = main_model.to(self.device)
        
        # 为主模型和可选 encoder 微调设置优化器
        if getattr(cfg, 'finetune_graph_autoencoder', False) and encoder_lr:
            self.optimizer = torch.optim.Adam([
                {'params': self.main_model.parameters(), 'lr': lr},
                {'params': self.encoder.parameters(), 'lr': encoder_lr}
            ])
        else:
            self.optimizer = torch.optim.Adam(self.main_model.parameters(), lr=lr)
        self._optimizer_param_ids = {
            id(p)
            for group in self.optimizer.param_groups
            for p in group.get('params', [])
        }

        # 可选学习率调度器：为了与原始实现保持一致，这里简化为指数衰减版本
        self.scheduler = None
        decay = getattr(cfg, 'lr_decay_rate', 1.0)
        gap = getattr(cfg, 'gap_update_lr', 0)
        if decay != 1.0:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=decay)
            self.scheduler_gap = max(1, gap) if gap else 1
        else:
            self.scheduler_gap = 0

        self.comm_cost = get_model_size(self.main_model) + get_model_size(self.encoder)
        self.training = False
        self.num_samples = len(self.train_data)

        # 多任务损失权重，用于平衡 AT / slew / netdelay / celldelay 的训练影响
        default_loss_weights = {
            'AT': 1.0,
            'slew': 1.0,
            'netdelay': 1.0,
            'celldelay': 1.0,
        }
        cfg_loss_weights = getattr(cfg, 'loss_weights', None)
        if isinstance(cfg_loss_weights, dict):
            default_loss_weights.update(cfg_loss_weights)
        self.loss_weights = default_loss_weights

        self.slew_loss_type = str(getattr(cfg, 'slew_loss_type', 'mse')).lower()
        self.slew_huber_beta = float(getattr(cfg, 'slew_huber_beta', 1.0))
        self.celldelay_loss_type = str(getattr(cfg, 'celldelay_loss_type', 'mse')).lower()
        self.celldelay_huber_beta = float(getattr(cfg, 'celldelay_huber_beta', 1.0))

        # FedEDA 相关配置：
        # trainer.py 会先调用 _fededa_init() 收集所有 client 的复杂度元数据，
        # 再把全局 CM_max / CM_min 广播回来。当前 client 在本地训练时使用这些
        # 信息，把电路复杂度映射成漂移正则强度。
        self.gamma_size = float(getattr(cfg, 'fededa_gamma_size', 0.0))
        self.gamma_p = float(getattr(cfg, 'fededa_gamma_p', 0.0))
        self.fededa_reg_scale = float(getattr(cfg, 'fededa_reg_scale', 1.0))
        self.fededa_alpha_clip_max = float(getattr(cfg, 'fededa_alpha_clip_max', 10.0))
        self.fededa_lambda_reduce = str(getattr(cfg, 'fededa_lambda_reduce', 'mean')).lower()
        self.fededa_drift_normalize = bool(getattr(cfg, 'fededa_drift_normalize', True))
        self.fededa_size_source = str(getattr(cfg, 'fededa_size_source', 'netlist_csv')).lower()
        self.fededa_size_csv_path = str(getattr(cfg, 'fededa_size_csv_path', '')).strip()
        self.fededa_size_csv_column = str(
            getattr(cfg, 'fededa_size_csv_column', 'size_cells_from_verilog_sky130')
        ).strip()
        self.fededa_cm_source = str(getattr(cfg, 'fededa_cm_source', 'precomputed_json')).lower()
        self.fededa_cm_json_path = str(getattr(cfg, 'fededa_cm_json_path', '')).strip()
        self._size_by_design = self._load_size_lookup_from_csv(
            self.fededa_size_csv_path,
            self.fededa_size_csv_column,
        ) if self.fededa_size_source in ('csv', 'netlist_csv') else {}
        self._cm_by_design = self._load_cm_lookup_from_json(
            self.fededa_cm_json_path
        ) if self.fededa_cm_source in ('json', 'precomputed_json') else {}
        self._size_source_reported = False
        self._cm_source_reported = False
        self._fededa_enabled = (self.gamma_size > 0 or self.gamma_p > 0)
        self.global_encoder_state = None  # 来自 server 的全局 encoder 状态（用于计算漂移）
        self.global_model_state = None
        self.circuit_metadata = {}        # {circuit_name: {size, p}}
        self.design_metadata = {}         # {design_name: {size, p}}
        self.cm_max = None                # server 广播的全局复杂度上界
        self.cm_min = None
        self._alpha_values = {}           # {circuit_name: {size, p}} 反向归一化后的 alpha

        # FL 算法选择：
        # - fedavg: 仅做参数平均
        # - fedprox: 额外加统一强度的模型漂移惩罚
        # - scaffold: 用控制变量修正梯度方向
        self.fl_algorithm = str(getattr(cfg, 'fl_algorithm', 'fedavg')).lower()
        if self.fl_algorithm not in ('fedavg', 'fedprox', 'scaffold'):
            self.fl_algorithm = 'fedavg'

        # FedProx：L = L_task + (mu/2) * ||w - w_global||^2
        self.fedprox_mu = float(getattr(cfg, 'fedprox_mu', 0.0))
        self.fedprox_drift_normalize = bool(getattr(cfg, 'fedprox_drift_normalize', True))
        self._fedprox_enabled = (self.fl_algorithm == 'fedprox' and self.fedprox_mu > 0.0)

        # SCAFFOLD 控制变量
        self._scaffold_enabled = (self.fl_algorithm == 'scaffold')
        self.scaffold_local_encoder_control = {}
        self.scaffold_local_model_control = {}
        self.scaffold_global_encoder_control = {}
        self.scaffold_global_model_control = {}
        if self._scaffold_enabled:
            self.init_scaffold_local_control()

    # -------------------------------------------------
    # 通用工具函数：权重、命名规整、状态迁移
    # -------------------------------------------------
    def _task_weight(self, task_name):
        try:
            return float(self.loss_weights.get(task_name, 1.0))
        except (TypeError, ValueError):
            return 1.0

    @staticmethod
    def _safe_rent_ratio_p(n, e):
        if n <= 1:
            return 0.5
        return min(1.0, max(0.0, math.log(e + 1) / math.log(n + 1)))

    @staticmethod
    def _fit_slope(xs, ys):
        if len(xs) < 2:
            return None
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        var_x = sum((x - x_mean) ** 2 for x in xs)
        if var_x <= 1e-12:
            return None
        cov_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        return cov_xy / var_x

    @staticmethod
    def _normalize_design_name(name):
        if name is None:
            return None
        base = os.path.basename(str(name).strip())
        marker = '.graph.bin-'
        if marker in base:
            base = base.split(marker, 1)[0] + '.graph.bin'
        if base.endswith('.graph.bin'):
            base = base[:-len('.graph.bin')]
        elif base.endswith('.v'):
            base = base[:-2]
        return base.lower().strip() if base else None

    @staticmethod
    def _to_cpu_state(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: PreRoutFullClient._to_cpu_state(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [PreRoutFullClient._to_cpu_state(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(PreRoutFullClient._to_cpu_state(v) for v in obj)
        return copy.deepcopy(obj)

    @staticmethod
    def _optimizer_state_to_device(optimizer, device):
        for state in optimizer.state.values():
            if not isinstance(state, dict):
                continue
            for k, v in list(state.items()):
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # -------------------------------------------------
    # FedEDA 元数据读取：优先读取离线预计算的 size / p（创新： client 端读取 size/p 预计算结果）
    # -------------------------------------------------
    def _load_size_lookup_from_csv(self, csv_path, size_column):
        if not csv_path:
            return {}
        if not os.path.isfile(csv_path):
            return {}

        lookup = {}
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                design_raw = row.get('design') or row.get('circuit') or row.get('name')
                key = self._normalize_design_name(design_raw)
                if not key:
                    continue
                raw = row.get(size_column)
                if raw is None:
                    continue
                try:
                    value = float(raw)
                except Exception:
                    continue
                if value > 0:
                    lookup[key] = value
        return lookup

    def _load_cm_lookup_from_json(self, json_path):
        if not json_path:
            return {}
        if not os.path.isfile(json_path):
            return {}

        try:
            with open(json_path, 'r') as f:
                payload = json.load(f)
        except Exception:
            return {}

        designs = payload.get('designs') if isinstance(payload, dict) else None
        if not isinstance(designs, dict):
            return {}

        lookup = {}
        for key_raw, rec in designs.items():
            key = self._normalize_design_name(key_raw)
            if not key or not isinstance(rec, dict):
                continue

            row = {}
            for field in ('size', 'p'):
                raw = rec.get(field)
                if raw is None:
                    continue
                try:
                    val = float(raw)
                except Exception:
                    continue
                if math.isfinite(val):
                    row[field] = val
            if row:
                lookup[key] = row
        return lookup

    def _resolve_cm_value(self, circuit_name, field):
        if not self._cm_by_design:
            return None, False
        key = self._normalize_design_name(circuit_name)
        if not key:
            return None, False
        row = self._cm_by_design.get(key)
        if not isinstance(row, dict):
            return None, False
        val = row.get(field)
        if val is None:
            return None, False
        try:
            out = float(val)
        except Exception:
            return None, False
        if not math.isfinite(out):
            return None, False
        return out, True

    def _resolve_size_value(self, circuit_name, fallback_nodes):
        size_from_cm, hit_cm = self._resolve_cm_value(circuit_name, 'size')
        if hit_cm and size_from_cm > 0:
            return float(size_from_cm), True
        if self.fededa_size_source in ('csv', 'netlist_csv') and self._size_by_design:
            key = self._normalize_design_name(circuit_name)
            if key in self._size_by_design:
                return float(self._size_by_design[key]), True
        return float(fallback_nodes), False

    def _get_edge_index(self, g):
        """在 CPU 上返回同构图/异构图统一格式的边索引 (n, u, v)。"""
        try:
            n = int(g.num_nodes())
        except Exception:
            n = int(sum(g.num_nodes(nt) for nt in g.ntypes))

        try:
            u, v = g.edges(form='uv')
            return n, u.detach().cpu().long(), v.detach().cpu().long()
        except Exception:
            us, vs = [], []
            etypes = getattr(g, 'canonical_etypes', None) or getattr(g, 'etypes', [])
            for et in etypes:
                try:
                    uu, vv = g.edges(etype=et, form='uv')
                except Exception:
                    uu, vv = g.edges(etype=et)
                us.append(uu.detach().cpu().long())
                vs.append(vv.detach().cpu().long())
            if us:
                return n, torch.cat(us, dim=0), torch.cat(vs, dim=0)
            empty = torch.empty(0, dtype=torch.long)
            return n, empty, empty

    def _estimate_p_topo_cut_fit(self, n, u, v, topo):
        """近似估计 Rent 的 p。

        方法是用拓扑层级前缀划分近似不同尺度的分块，
        再拟合 log(切边终端数) 对 log(分块规模) 的斜率。
        """
        if n <= 2 or u.numel() == 0 or not topo:
            return None

        level_id = torch.full((n,), -1, dtype=torch.long)
        for lid, nodes in enumerate(topo):
            idx = nodes.detach().cpu().long()
            idx = idx[(idx >= 0) & (idx < n)]
            level_id[idx] = lid

        max_level = int(level_id.max().item()) if level_id.numel() > 0 else -1
        if max_level <= 0:
            return None

        cut_fracs = getattr(self.cfg, 'fededa_p_cut_fracs', [0.2, 0.4, 0.6, 0.8])
        xs, ys = [], []
        for frac in cut_fracs:
            try:
                frac = float(frac)
            except Exception:
                continue
            if frac <= 0.0 or frac >= 1.0:
                continue

            threshold = int(max_level * frac)
            mask = level_id <= threshold
            m = int(mask.sum().item())
            if m <= 1 or m >= n:
                continue

            cut = int(torch.logical_xor(mask[u], mask[v]).sum().item())
            if cut <= 0:
                continue

            xs.append(math.log(float(m)))
            ys.append(math.log(float(cut)))

        slope = self._fit_slope(xs, ys)
        if slope is None:
            return None
        return min(1.0, max(0.0, float(slope)))

    # -------------------------------------------------
    # 电路复杂度元数据与自适应漂移正则(核心)
    # -------------------------------------------------
    # 这一段和 scripts/compute_fededa_cm_from_verilog.py 是直接关联的：
    # - 预计算脚本负责离线生成 design -> {size, p}；
    # - 本文件优先读取这些预计算结果；

    def compute_circuit_metadata(self):#创新：client 端统计自己持有电路的复杂度元数据
        """计算每个样本的复杂度元数据 CM(size, p)，优先使用离线预计算结果。

        优先级：
        1) 预计算 JSON 中的 size / p；
        2) size CSV 中的 size；
        3) 图上的 p 近似估计作为回退方案。
        """
        metadata = {}
        design_metadata = {}
        per_design_samples = {}
        per_design_size_candidates = {}
        per_design_p_candidates = {}

        p_mode = str(getattr(self.cfg, 'fededa_p_estimator', 'rent_topo_fit')).lower()

        size_hit = 0
        size_miss = 0
        p_json_hit = 0
        p_fallback = 0

        # 遍历本 client 的全部训练电路，得到每个电路的复杂度元数据。
        for name, (g, ts) in self.train_data.items():
            design_key = self._normalize_design_name(name) or str(name)
            per_design_samples.setdefault(design_key, []).append(name)

            n, u, v = self._get_edge_index(g)
            e = int(u.numel())

            size_val, hit_size = self._resolve_size_value(name, n)
            if hit_size:
                size_hit += 1
            else:
                size_miss += 1
            per_design_size_candidates.setdefault(design_key, []).append(float(size_val))

            # p 的优先级：
            # 1) 预计算 JSON；
            # 2) 图上近似 Rent 拟合；
            # 3) 边节点比率近似。
            p_est, hit_p_json = self._resolve_cm_value(name, 'p')
            if hit_p_json:
                p_json_hit += 1
            else:
                if p_mode in ('rent_topo_fit', 'topo_fit'):
                    p_est = self._estimate_p_topo_cut_fit(
                        n, u, v, ts.get('topo') if isinstance(ts, dict) else None
                    )
                elif p_mode in ('edge_node_ratio', 'ratio'):
                    p_est = None

                if p_est is None:
                    p_est = self._safe_rent_ratio_p(n, e)
                p_fallback += 1
            per_design_p_candidates.setdefault(design_key, []).append(float(p_est))

        for design_key, sample_names in per_design_samples.items():
            size_candidates = per_design_size_candidates.get(design_key, [])
            p_candidates = per_design_p_candidates.get(design_key, [])

            size_design = max(size_candidates) if size_candidates else 0.0
            p_design = sum(p_candidates) / len(p_candidates) if p_candidates else 0.0

            design_metadata[design_key] = {
                'size': float(size_design),
                'p': float(p_design),
            }

            for sample_name in sample_names:
                metadata[sample_name] = {
                    'size': float(size_design),
                    'p': float(p_design),
                }

        if not self._size_source_reported and self.fededa_size_source in ('csv', 'netlist_csv'):
            if self._size_by_design:
                print(
                    f"[FedEDA][Client {self.client_id}] size source={self.fededa_size_source}, "
                    f"lookup_hit={size_hit}, fallback_to_num_nodes={size_miss}."
                )
            elif self._cm_by_design:
                print(
                    f"[FedEDA][Client {self.client_id}] size source=precomputed_json, "
                    f"lookup_hit={size_hit}, fallback_to_num_nodes={size_miss}."
                )
            else:
                print(
                    f"[FedEDA][Client {self.client_id}] size lookup unavailable, "
                    "fallback to g.num_nodes() for all circuits."
                )
            self._size_source_reported = True

        if not self._cm_source_reported and self.fededa_cm_source in ('json', 'precomputed_json'):
            if self._cm_by_design:
                print(
                    f"[FedEDA][Client {self.client_id}] p source={self.fededa_cm_source}, "
                    f"p_json_hit={p_json_hit}, p_fallback={p_fallback}."
                )
            else:
                print(
                    f"[FedEDA][Client {self.client_id}] p json unavailable, "
                    "fallback to graph-based estimator."
                )
            self._cm_source_reported = True

        self.circuit_metadata = metadata
        self.design_metadata = design_metadata
        return metadata

    def get_circuit_metadata_summary(self):
        """返回聚合后的 CM {size, p}，供 server 计算全局上下界。"""
        if not self.circuit_metadata:
            self.compute_circuit_metadata()
        vals = list(self.design_metadata.values()) if self.design_metadata else list(self.circuit_metadata.values())
        return {
            'size':  sum(v['size'] for v in vals),
            'p':     sum(v['p'] for v in vals) / max(1, len(vals)),
        }

    def get_circuit_metadata_values(self):
        """返回逐电路的 CM 值，供 server 计算全局电路级上下界。"""
        if not self.circuit_metadata:
            self.compute_circuit_metadata()
        return list(self.design_metadata.values()) if self.design_metadata else list(self.circuit_metadata.values())

    def set_cm_bounds(self, cm_max, cm_min): #创新： client 端把 size/p 变成 alpha，再加到本地正则
        """接收 server 的全局复杂度上下界，并预计算每个电路的 alpha。

        FedEDA 使用反向 min-max 归一化：
            alpha = (CM_max - CM_min) / (CM - CM_min)
        规模更小、结构更简单的电路会得到更大的 alpha，
        从而在本地训练时受到更强的漂移惩罚。
        """
        self.cm_max = cm_max
        self.cm_min = cm_min
        if not self.circuit_metadata:
            self.compute_circuit_metadata()

        def _inv_minmax(val, vmax, vmin):
            if abs(float(vmax) - float(vmin)) < 1e-8:
                return 1.0  # 如果所有电路复杂度相同，则统一赋权
            denom = max(1e-8, float(val) - float(vmin))
            alpha = (float(vmax) - float(vmin)) / denom
            if self.fededa_alpha_clip_max > 0:
                alpha = min(alpha, self.fededa_alpha_clip_max)
            return alpha

        self._alpha_values = {}
        for name, cm in self.circuit_metadata.items():
            self._alpha_values[name] = {
                'size':  _inv_minmax(cm['size'],  cm_max['size'],  cm_min['size']),
                'p':     _inv_minmax(cm['p'],     cm_max['p'],     cm_min['p']),
            }

    # -------------------------------------------------
    # 联邦算法状态同步：接收来自 trainer.py 的全局模型 / 控制变量
    # -------------------------------------------------
    def set_global_state(self, encoder_state, model_state):
        """保存聚合后的全局状态 w_t，供 FedEDA/FedProx 计算漂移项。"""
        self.global_encoder_state = copy.deepcopy(encoder_state) if encoder_state is not None else None
        self.global_model_state = copy.deepcopy(model_state) if model_state is not None else None

    def init_scaffold_local_control(self):
        """将本地 SCAFFOLD 控制变量 c_i 初始化为全零。"""
        self.scaffold_local_encoder_control = {}
        self.scaffold_local_model_control = {}

        for name, param in self.encoder.named_parameters():
            if param.requires_grad and id(param) in self._optimizer_param_ids:
                self.scaffold_local_encoder_control[name] = torch.zeros_like(
                    param.detach().cpu().float()
                )
        for name, param in self.main_model.named_parameters():
            if param.requires_grad and id(param) in self._optimizer_param_ids:
                self.scaffold_local_model_control[name] = torch.zeros_like(
                    param.detach().cpu().float()
                )

    def set_scaffold_global_control(self, encoder_control, model_control):
        """接收 server 端的 SCAFFOLD 控制变量 c。"""
        if not self._scaffold_enabled:
            return

        self.scaffold_global_encoder_control = {}
        self.scaffold_global_model_control = {}

        for name, param in self.encoder.named_parameters():
            if not (param.requires_grad and id(param) in self._optimizer_param_ids):
                continue
            src = encoder_control.get(name) if isinstance(encoder_control, dict) else None
            if src is None:
                src = torch.zeros_like(param.detach().cpu().float())
            self.scaffold_global_encoder_control[name] = src.detach().cpu().float().clone()

        for name, param in self.main_model.named_parameters():
            if not (param.requires_grad and id(param) in self._optimizer_param_ids):
                continue
            src = model_control.get(name) if isinstance(model_control, dict) else None
            if src is None:
                src = torch.zeros_like(param.detach().cpu().float())
            self.scaffold_global_model_control[name] = src.detach().cpu().float().clone()

    def _compute_model_drift(self, normalize=True):
        """计算 ||w_global - w_local||²，对本地参数保持可微。"""
        if self.global_model_state is None:
            return torch.zeros([], device=self.device)
        drift = torch.zeros([], device=self.device)
        param_count = 0
        for name, param in self.main_model.named_parameters():
            if name in self.global_model_state and param.requires_grad and id(param) in self._optimizer_param_ids:
                g_w = self.global_model_state[name].to(self.device).detach().float()
                drift = drift + ((param.float() - g_w) ** 2).sum()
                param_count += param.numel()
        if (self.global_encoder_state is not None
                and getattr(self.cfg, 'finetune_graph_autoencoder', False)):
            for name, param in self.encoder.named_parameters():
                if name in self.global_encoder_state and param.requires_grad and id(param) in self._optimizer_param_ids:
                    g_w = self.global_encoder_state[name].to(self.device).detach().float()
                    drift = drift + ((param.float() - g_w) ** 2).sum()
                    param_count += param.numel()
        if normalize and param_count > 0:
            drift = drift / float(param_count)
        return drift

    def _apply_scaffold_grad_correction(self):
        """应用 SCAFFOLD 梯度修正：grad <- grad + (c - c_i)。"""
        if not self._scaffold_enabled:
            return

        for name, param in self.main_model.named_parameters():
            if param.grad is None:
                continue
            c_global = self.scaffold_global_model_control.get(name)
            c_local = self.scaffold_local_model_control.get(name)
            if c_global is None or c_local is None:
                continue
            param.grad = param.grad + (c_global.to(self.device) - c_local.to(self.device))

        for name, param in self.encoder.named_parameters():
            if param.grad is None:
                continue
            c_global = self.scaffold_global_encoder_control.get(name)
            c_local = self.scaffold_local_encoder_control.get(name)
            if c_global is None or c_local is None:
                continue
            param.grad = param.grad + (c_global.to(self.device) - c_local.to(self.device))

    def _update_scaffold_local_control(self, local_steps):
        """更新客户端控制变量 c_i，并返回给 server 用于更新 c 的增量。"""
        if (not self._scaffold_enabled) or self.global_model_state is None or local_steps <= 0:
            return None

        base_lr = float(self.optimizer.param_groups[0].get('lr', getattr(self.cfg, 'lr', 1e-4)))
        denom = max(1e-12, base_lr * float(local_steps))

        delta_encoder = {}
        delta_model = {}

        for name, param in self.main_model.named_parameters():
            if not (param.requires_grad and id(param) in self._optimizer_param_ids):
                continue
            if name not in self.global_model_state:
                continue

            c_i_old = self.scaffold_local_model_control.get(name)
            if c_i_old is None:
                c_i_old = torch.zeros_like(param.detach().cpu().float())
                self.scaffold_local_model_control[name] = c_i_old

            c_global = self.scaffold_global_model_control.get(name)
            if c_global is None:
                c_global = torch.zeros_like(c_i_old)

            w_t = self.global_model_state[name].detach().cpu().float()
            w_i = param.detach().cpu().float()
            c_i_new = c_i_old - c_global + (w_t - w_i) / denom
            delta_model[name] = (c_i_new - c_i_old).clone()
            self.scaffold_local_model_control[name] = c_i_new

        if self.global_encoder_state is not None:
            for name, param in self.encoder.named_parameters():
                if not (param.requires_grad and id(param) in self._optimizer_param_ids):
                    continue
                if name not in self.global_encoder_state:
                    continue

                c_i_old = self.scaffold_local_encoder_control.get(name)
                if c_i_old is None:
                    c_i_old = torch.zeros_like(param.detach().cpu().float())
                    self.scaffold_local_encoder_control[name] = c_i_old

                c_global = self.scaffold_global_encoder_control.get(name)
                if c_global is None:
                    c_global = torch.zeros_like(c_i_old)

                w_t = self.global_encoder_state[name].detach().cpu().float()
                w_i = param.detach().cpu().float()
                c_i_new = c_i_old - c_global + (w_t - w_i) / denom
                delta_encoder[name] = (c_i_new - c_i_old).clone()
                self.scaffold_local_encoder_control[name] = c_i_new

        return {'encoder': delta_encoder, 'model': delta_model}

    # -------------------------------------------------
    # 模型状态与训练状态导出/恢复
    # -------------------------------------------------
    def load_state(self, encoder_state, model_state):
        """加载 encoder 和主模型两套参数。"""
        self.encoder.load_state_dict(encoder_state)
        self.main_model.load_state_dict(model_state)

    def get_state(self):
        """返回 encoder 和主模型两套参数。"""
        return {
            'encoder': self.encoder.state_dict(),
            'model': self.main_model.state_dict()
        }

    def export_training_state(self):
        """导出完整本地训练状态，供 checkpoint 保存与恢复。"""
        return {
            'state': self._to_cpu_state(self.get_state()),
            'optimizer': self._to_cpu_state(self.optimizer.state_dict()),
            'scheduler': self._to_cpu_state(self.scheduler.state_dict()) if self.scheduler is not None else None,
            'scaffold_local_encoder_control': self._to_cpu_state(self.scaffold_local_encoder_control),
            'scaffold_local_model_control': self._to_cpu_state(self.scaffold_local_model_control),
            'scaffold_global_encoder_control': self._to_cpu_state(self.scaffold_global_encoder_control),
            'scaffold_global_model_control': self._to_cpu_state(self.scaffold_global_model_control),
            'global_encoder_state': self._to_cpu_state(self.global_encoder_state),
            'global_model_state': self._to_cpu_state(self.global_model_state),
            'circuit_metadata': copy.deepcopy(self.circuit_metadata),
            'design_metadata': copy.deepcopy(self.design_metadata),
            'cm_max': copy.deepcopy(self.cm_max),
            'cm_min': copy.deepcopy(self.cm_min),
            'alpha_values': copy.deepcopy(self._alpha_values),
        }

    def load_training_state(self, payload):
        """从 checkpoint 载荷中恢复完整本地训练状态。"""
        if not isinstance(payload, dict):
            return

        state = payload.get('state')
        if isinstance(state, dict):
            enc = state.get('encoder')
            mdl = state.get('model')
            if isinstance(enc, dict) and isinstance(mdl, dict):
                self.load_state(enc, mdl)

        opt_state = payload.get('optimizer')
        if isinstance(opt_state, dict):
            self.optimizer.load_state_dict(opt_state)
            self._optimizer_state_to_device(self.optimizer, self.device)

        sch_state = payload.get('scheduler')
        if self.scheduler is not None and isinstance(sch_state, dict):
            self.scheduler.load_state_dict(sch_state)

        self.scaffold_local_encoder_control = self._to_cpu_state(
            payload.get('scaffold_local_encoder_control', {})
        ) or {}
        self.scaffold_local_model_control = self._to_cpu_state(
            payload.get('scaffold_local_model_control', {})
        ) or {}
        self.scaffold_global_encoder_control = self._to_cpu_state(
            payload.get('scaffold_global_encoder_control', {})
        ) or {}
        self.scaffold_global_model_control = self._to_cpu_state(
            payload.get('scaffold_global_model_control', {})
        ) or {}
        self.global_encoder_state = self._to_cpu_state(payload.get('global_encoder_state'))
        self.global_model_state = self._to_cpu_state(payload.get('global_model_state'))

        self.circuit_metadata = copy.deepcopy(payload.get('circuit_metadata', {}))
        self.design_metadata = copy.deepcopy(payload.get('design_metadata', {}))
        self.cm_max = copy.deepcopy(payload.get('cm_max'))
        self.cm_min = copy.deepcopy(payload.get('cm_min'))
        self._alpha_values = copy.deepcopy(payload.get('alpha_values', {}))

    # client 前向主链（核心）
    def _forward_with_encoder(self, g, ts):
        """
        执行 encoder + 主模型的完整前向。

        返回：
        - netdelay_pred, celldelay_pred, AT_slew_pred
        - 各任务真值 targets
        - 已移动到当前设备上的图 g
        """
        g = g.to(self.device)
        
        # 第 1 步：在 homo 图上运行 encoder，提取节点潜在表示 latent
        homo_graph = ts['homo']
        homo_graph = homo_graph.to(self.device)
        nf_homo = homo_graph.ndata['nf']
        
        # 将 ts 字典递归移动到设备上，兼容其中的嵌套张量/列表/字典
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
        
        ts_device = to_device_recursive(ts, self.device)
        
        with torch.set_grad_enabled(self.training):
            latent = self.encoder(homo_graph, nf_homo)
            global_embedding = latent.mean(dim=0).expand_as(latent)
            
            # 第 2 步：将节点级表示 latent 和图级表示 global_embedding
            # 将 original_nf + latent + global_embedding 拼接为新的节点特征，供主模型做时序预测。
            original_nf = g.ndata['nf']
            g.ndata['nf'] = torch.cat([original_nf, latent, global_embedding], dim=1)
            
            # 第 3 步：送入主模型做时序预测
            use_groundtruth = getattr(self.cfg, 'groundtruth', True) if self.training else False
            netdelay_pred, celldelay_pred, AT_slew_pred = self.main_model(
                g,
                ts_device,
                groundtruth=use_groundtruth
            )
            
            # 第 4 步：提取各任务的真实标签
            AT_truth = g.ndata['n_atslew'][:, 0:4]
            slew_truth = g.ndata['n_atslew'][:, 4:8] if getattr(self.cfg, 'predict_slew', True) else None
            netdelay_truth = g.ndata['n_net_delays_log'] if getattr(self.cfg, 'predict_netdelay', True) else None
            celldelay_truth = g.edges['cell_out'].data['e_cell_delays'] if getattr(self.cfg, 'predict_celldelay', True) else None
            
            # 恢复原始节点特征，避免污染后续流程
            g.ndata['nf'] = original_nf
            
        return netdelay_pred, celldelay_pred, AT_slew_pred, (AT_truth, slew_truth, netdelay_truth, celldelay_truth), g

    # -------------------------------------------------
    # 多任务损失与指标
    # -------------------------------------------------
    def _compute_loss(self, g, ts):
        """计算 AT、Slew、NetDelay、CellDelay 的多任务损失。"""
        netdelay_pred, celldelay_pred, AT_slew_pred, (AT_truth, slew_truth, netdelay_truth, celldelay_truth), g = \
            self._forward_with_encoder(g, ts)
        
        AT_pred = AT_slew_pred[:, 0:4]
        slew_pred = AT_slew_pred[:, 4:8] if getattr(self.cfg, 'predict_slew', True) else None
        
        # 计算各任务损失；如果图上存在 valid mask，则只对有效节点计损失
        loss_AT = F.mse_loss(AT_pred, AT_truth, reduction='none').mean(dim=1)
        if g.ndata.get('valid') is not None:
            loss_AT = loss_AT * g.ndata['valid']
        losses = {'AT': loss_AT.mean().item()}
        total_loss = self._task_weight('AT') * loss_AT.mean()

        if slew_pred is not None and slew_truth is not None:
            if self.slew_loss_type in ('smooth_l1', 'huber'):
                loss_slew = F.smooth_l1_loss(
                    slew_pred,
                    slew_truth,
                    beta=self.slew_huber_beta,
                    reduction='none',
                ).mean(dim=1)
            else:
                loss_slew = F.mse_loss(slew_pred, slew_truth, reduction='none').mean(dim=1)
            if g.ndata.get('valid') is not None:
                loss_slew = loss_slew * g.ndata['valid']
            losses['slew'] = loss_slew.mean().item()
            total_loss = total_loss + self._task_weight('slew') * loss_slew.mean()

        if netdelay_pred is not None and netdelay_truth is not None:
            loss_netdelay = F.mse_loss(netdelay_pred, netdelay_truth, reduction='none').mean(dim=1)
            if g.ndata.get('valid') is not None:
                loss_netdelay = loss_netdelay * g.ndata['valid']
            losses['netdelay'] = loss_netdelay.mean().item()
            total_loss = total_loss + self._task_weight('netdelay') * loss_netdelay.mean()

        if celldelay_pred is not None and celldelay_truth is not None:
            if self.celldelay_loss_type in ('smooth_l1', 'huber'):
                loss_celldelay = F.smooth_l1_loss(
                    celldelay_pred,
                    celldelay_truth,
                    beta=self.celldelay_huber_beta,
                )
            else:
                loss_celldelay = F.mse_loss(celldelay_pred, celldelay_truth)
            losses['celldelay'] = loss_celldelay.item()
            total_loss = total_loss + self._task_weight('celldelay') * loss_celldelay
        
        return total_loss, losses

    def _compute_metrics(self, g, ts):
        """用 PreRoutGNN 的指标函数计算各任务的 MSE、MAE、R²。"""
        netdelay_pred, celldelay_pred, AT_slew_pred, (AT_truth, slew_truth, netdelay_truth, celldelay_truth), g = \
            self._forward_with_encoder(g, ts)
        
        AT_pred = AT_slew_pred[:, 0:4]
        slew_pred = AT_slew_pred[:, 4:8] if getattr(self.cfg, 'predict_slew', True) else None
        
        metrics = {}
        
        mse_AT = calc_mse(AT_truth, AT_pred).item()
        mae_AT = calc_mae(AT_truth, AT_pred).item()
        r2_AT = calc_r2_torch(AT_truth, AT_pred).item()
        metrics.update({'mse-AT': mse_AT, 'mae-AT': mae_AT, 'r2-AT': r2_AT})
        if slew_pred is not None and slew_truth is not None:
            mse_slew = calc_mse(slew_truth, slew_pred).item()
            mae_slew = calc_mae(slew_truth, slew_pred).item()
            r2_slew = calc_r2_torch(slew_truth, slew_pred).item()
            metrics.update({'mse-slew': mse_slew, 'mae-slew': mae_slew, 'r2-slew': r2_slew})
        
        # NetDelay 指标
        if netdelay_pred is not None and netdelay_truth is not None:
            mse_netdelay = calc_mse(netdelay_truth, netdelay_pred).item()
            mae_netdelay = calc_mae(netdelay_truth, netdelay_pred).item()
            r2_netdelay = calc_r2_torch(netdelay_truth, netdelay_pred).item()
            metrics.update({'mse-netdelay': mse_netdelay, 'mae-netdelay': mae_netdelay, 'r2-netdelay': r2_netdelay})
        
        # CellDelay 指标
        if celldelay_pred is not None and celldelay_truth is not None:
            mse_celldelay = calc_mse(celldelay_truth, celldelay_pred).item()
            mae_celldelay = calc_mae(celldelay_truth, celldelay_pred).item()
            r2_celldelay = calc_r2_torch(celldelay_truth, celldelay_pred).item()
            metrics.update({'mse-celldelay': mse_celldelay, 'mae-celldelay': mae_celldelay, 'r2-celldelay': r2_celldelay})
        
        # 总体指标：对各任务指标做平均
        mse_overall = sum([v for k, v in metrics.items() if k.startswith('mse-')]) / len([k for k in metrics if k.startswith('mse-')])
        mae_overall = sum([v for k, v in metrics.items() if k.startswith('mae-')]) / len([k for k in metrics if k.startswith('mae-')])
        r2_overall = sum([v for k, v in metrics.items() if k.startswith('r2-')]) / len([k for k in metrics if k.startswith('r2-')])
        metrics.update({'mse': mse_overall, 'mae': mae_overall, 'r2': r2_overall})
        
        return metrics

    # -------------------------------------------------
    # 本地训练：任务损失 + 联邦正则 + 参数更新
    # -------------------------------------------------
    # 这是 trainer.py 每轮调用 client.train_local() 时真正发生梯度更新的地方。
    def train_epoch(self):
        if len(self.train_data) == 0:
            return 0.0, {}, {'task_loss': 0.0, 'reg_loss': 0.0, 'local_steps': 0}

        self.encoder.train()
        self.main_model.train()
        self.training = True
        total_loss = 0.0
        task_loss_total = 0.0
        reg_loss_total = 0.0
        local_steps = 0
        losses_sum = {}

        for circuit_name, (g, ts) in self.train_data.items():
            self.optimizer.zero_grad()
            loss, losses = self._compute_loss(g, ts)
            task_loss_total += loss.item()
            total_step_loss = loss

            # FedEDA：按电路复杂度自适应调节漂移正则强度。
            # 这里使用的 alpha 值来自 trainer.py -> _fededa_init() 广播的全局范围。
            if self._fededa_enabled and self.global_model_state is not None and self._alpha_values:
                alpha = self._alpha_values.get(circuit_name, {})
                lambda_reg = (
                    alpha.get('size', 1.0) * self.gamma_size
                    + alpha.get('p', 1.0) * self.gamma_p
                )
                if self.fededa_lambda_reduce == 'mean' and len(self.train_data) > 0:
                    lambda_reg /= float(len(self.train_data))
                if self.fededa_reg_scale > 0:
                    lambda_reg *= self.fededa_reg_scale

                if lambda_reg > 0:
                    drift = self._compute_model_drift(normalize=self.fededa_drift_normalize)
                    if drift.item() > 0:
                        reg = lambda_reg * drift
                        reg_loss_total += reg.item()
                        total_step_loss = total_step_loss + reg

            # FedProx：与 FedEDA 相比，它对所有电路施加统一强度的漂移惩罚。
            if self._fedprox_enabled and self.global_model_state is not None:
                prox_drift = self._compute_model_drift(normalize=self.fedprox_drift_normalize)
                if prox_drift.item() > 0:
                    prox_reg = 0.5 * self.fedprox_mu * prox_drift
                    reg_loss_total += prox_reg.item()
                    total_step_loss = total_step_loss + prox_reg

            total_loss += total_step_loss.item()
            total_step_loss.backward()

            # SCAFFOLD：在反向传播后、参数更新前，对梯度做控制变量修正。
            if self._scaffold_enabled:
                self._apply_scaffold_grad_correction()

            if getattr(self.cfg, 'max_gradient_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.main_model.parameters()),
                    getattr(self.cfg, 'max_gradient_norm', 1000.0)
                )

            self.optimizer.step()
            local_steps += 1

            del loss, total_step_loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            for k, v in losses.items():
                losses_sum[k] = losses_sum.get(k, 0.0) + v

        if self.scheduler and getattr(self.cfg, 'current_epoch', 1) % self.scheduler_gap == 0:
            self.scheduler.step()

        num = len(self.train_data)
        avg_losses = {k: v / num for k, v in losses_sum.items()}
        train_parts = {
            'task_loss': task_loss_total / num,
            'reg_loss': reg_loss_total / num,
            'local_steps': local_steps,
        }
        return total_loss / num, avg_losses, train_parts

    # -------------------------------------------------
    # 本地评估与 trainer.py 的交互出口
    # -------------------------------------------------
    @torch.no_grad()
    def test_epoch(self):
        if len(self.test_data) == 0:
            return 0.0, {}, {}

        self.encoder.eval()
        self.main_model.eval()
        self.training = False
        total_loss = 0.0
        losses_sum = {}
        metrics_sum = {}

        for circuit_name, (g, ts) in self.test_data.items():
            loss, losses = self._compute_loss(g, ts)
            total_loss += loss.item()
            
            for k, v in losses.items():
                losses_sum[k] = losses_sum.get(k, 0.0) + v
            
            # 计算评估指标
            metrics = self._compute_metrics(g, ts)
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v

        num = len(self.test_data)
        avg_losses = {k: v / num for k, v in losses_sum.items()}
        avg_metrics = {k: v / num for k, v in metrics_sum.items()}
        return total_loss / num, avg_losses, avg_metrics

    def train_local(self, local_epochs):
        # trainer.py 在每轮联邦训练时调用这里。
        # 返回内容包括：
        # - 当前 client 的双模型参数 state
        # - 本地训练/验证损失
        # - 多任务评估指标
        # - 样本数 num_samples（供 FedAvg 加权）
        # - scaffold_delta_control（供 SCAFFOLD 更新 server control）
        local_train_loss = 0.0
        local_train_task_loss = 0.0
        local_train_reg_loss = 0.0
        local_test_loss = 0.0
        final_metrics = {}
        final_losses = {}
        total_local_steps = 0

        for e in range(local_epochs):
            self.cfg.current_epoch = int(getattr(self.cfg, 'global_epoch', e + 1))
            tl, train_losses, train_parts = self.train_epoch()
            vl, test_losses, metrics = self.test_epoch()
            local_train_loss += tl / local_epochs
            local_train_task_loss += train_parts.get('task_loss', tl) / local_epochs
            local_train_reg_loss += train_parts.get('reg_loss', 0.0) / local_epochs
            local_test_loss += vl / local_epochs
            total_local_steps += int(train_parts.get('local_steps', 0))
            final_metrics = metrics
            final_losses = test_losses

        scaffold_delta_control = None
        if self._scaffold_enabled:
            scaffold_delta_control = self._update_scaffold_local_control(total_local_steps)

        return {
            'id': self.client_id,
            'state': self.get_state(),
            'train_loss': local_train_loss,
            'train_task_loss': local_train_task_loss,
            'train_reg_loss': local_train_reg_loss,
            'train_acc': 0.0,
            'test_loss': local_test_loss,
            'test_acc': 0.0,
            'comm_cost': self.comm_cost,
            'metrics': final_metrics,
            'losses': final_losses,
            'num_samples': self.num_samples,
            'local_steps': total_local_steps,
            'scaffold_delta_control': scaffold_delta_control,
        }
