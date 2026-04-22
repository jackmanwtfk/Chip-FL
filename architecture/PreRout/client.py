import torch
from torch.nn import functional as F

from common import find_device, get_param_size, get_model_size


class GraphAutoEncoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, g):
        nf = g.ndata['nf']
        latent = self.encoder(g, nf)
        rec = self.decoder(g, latent)
        return latent, rec


class PreRoutClient:
    def __init__(self, client_id, train_data, test_data, cfg, model_builder, lr):
        self.client_id = client_id
        self.train_data = train_data
        self.test_data = test_data
        self.cfg = cfg
        self.device = find_device()

        self.model = model_builder().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.comm_cost = get_model_size(self.model)

    def load_state(self, state_dict):
        self.model.load_state_dict(state_dict)

    def get_state(self):
        return self.model.state_dict()

    def _compute_loss(self, g):
        g = g.to(self.device)
        nf = g.ndata['nf']
        latent, rec = self.model(g)

        loss_reconstruction = F.mse_loss(rec, nf[:, 0:10], reduction="none").mean(dim=1) * g.ndata['valid']
        loss_reconstruction = loss_reconstruction.mean()
        loss = loss_reconstruction

        loss_KL_val = 0.0
        if getattr(self.cfg, 'weight_KL_divergence', 0.0) > 0:
            latent_var = latent.var(dim=0)
            latent_mean = latent.mean(dim=0)
            loss_KL = latent_var + latent_mean**2 - 1 - torch.log(latent_var)
            loss_KL = loss_KL.mean()
            loss += self.cfg.weight_KL_divergence * loss_KL
            loss_KL_val = loss_KL.item()

        return loss, loss_reconstruction.item(), loss_KL_val

    def _compute_metrics(self, g):
        """Compute reconstruction metrics: MSE, MAE, R2 for each task (AT, Slew, NetDelay, CellDelay)"""
        g = g.to(self.device)
        nf = g.ndata['nf']
        target = nf[:, 0:10]
        valid_mask = g.ndata['valid'].bool()
        
        latent, rec = self.model(g)
        
        # Apply valid mask
        target_valid = target[valid_mask]
        rec_valid = rec[valid_mask]
        
        # Task indices in node features (first 10 dims):
        # 0-3: AT (Arrival Time)
        # 4-7: Slew (Slew Rate)
        # 8-9: NetDelay, CellDelay (combined)
        
        metrics = {}
        task_ranges = {
            'AT': (0, 4),
            'slew': (4, 8),
            'netdelay': (8, 9),
            'celldelay': (9, 10),
        }
        
        # Overall metrics
        mse_overall = F.mse_loss(rec_valid, target_valid).item()
        mae_overall = F.l1_loss(rec_valid, target_valid).item()
        ss_res_overall = ((target_valid - rec_valid) ** 2).sum()
        ss_tot_overall = ((target_valid - target_valid.mean(dim=0, keepdim=True)) ** 2).sum()
        r2_overall = (1 - ss_res_overall / (ss_tot_overall + 1e-8)).item()
        
        metrics['mse'] = mse_overall
        metrics['mae'] = mae_overall
        metrics['r2'] = r2_overall
        
        # Per-task metrics
        for task_name, (start, end) in task_ranges.items():
            target_task = target_valid[:, start:end]
            rec_task = rec_valid[:, start:end]
            
            mse_task = F.mse_loss(rec_task, target_task).item()
            mae_task = F.l1_loss(rec_task, target_task).item()
            
            ss_res = ((target_task - rec_task) ** 2).sum()
            ss_tot = ((target_task - target_task.mean(dim=0, keepdim=True)) ** 2).sum()
            r2_task = (1 - ss_res / (ss_tot + 1e-8)).item()
            
            metrics[f'mse-{task_name}'] = mse_task
            metrics[f'mae-{task_name}'] = mae_task
            metrics[f'r2-{task_name}'] = r2_task
        
        return metrics

    def train_epoch(self):
        if len(self.train_data) == 0:
            return 0.0, 0.0

        self.model.train()
        total_loss = 0.0
        total_rec = 0.0

        for _, (g, _) in self.train_data.items():
            self.optimizer.zero_grad()
            loss, loss_rec, _ = self._compute_loss(g)
            loss.backward()

            if getattr(self.cfg, 'max_gradient_norm', -1) and self.cfg.max_gradient_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_gradient_norm)

            self.optimizer.step()
            total_loss += loss.item()
            total_rec += loss_rec

        num = len(self.train_data)
        return total_loss / num, total_rec / num

    @torch.no_grad()
    def test_epoch(self):
        if len(self.test_data) == 0:
            return 0.0, 0.0, {}

        self.model.eval()
        total_loss = 0.0
        total_rec = 0.0
        metrics_sum = {'mse': 0.0, 'mae': 0.0, 'r2': 0.0}

        for _, (g, _) in self.test_data.items():
            loss, loss_rec, _ = self._compute_loss(g)
            total_loss += loss.item()
            total_rec += loss_rec
            
            # Compute metrics
            metrics = self._compute_metrics(g)
            for k in metrics_sum:
                metrics_sum[k] += metrics[k]

        num = len(self.test_data)
        avg_metrics = {k: v / num for k, v in metrics_sum.items()}
        return total_loss / num, total_rec / num, avg_metrics

    def train_local(self, local_epochs):
        local_train_loss = 0.0
        local_test_loss = 0.0
        final_metrics = {}

        for _ in range(local_epochs):
            tl, _ = self.train_epoch()
            vl, _, metrics = self.test_epoch()
            local_train_loss += tl / local_epochs
            local_test_loss += vl / local_epochs
            final_metrics = metrics  # Keep last epoch metrics

        return {
            'id': self.client_id,
            'state': self.get_state(),
            'train_loss': local_train_loss,
            'train_acc': 0.0,
            'test_loss': local_test_loss,
            'test_acc': 0.0,
            'comm_cost': self.comm_cost,
            'metrics': final_metrics,  # R2, MSE, MAE
        }
