from copy import deepcopy

from architecture.PreRout.trainer import PreRoutFedTrainer, DEFAULT_PREROUT_CFG


class FedEDATrainer(PreRoutFedTrainer):
    def __init__(
        self,
        num_clients=3,
        local_epochs=1,
        epochs=5,
        lr=5e-4,
        num_select=0,
        parallel=False,
        prerout=None,
        fededa=None,
        **kwargs,
    ):
        merged_cfg = deepcopy(DEFAULT_PREROUT_CFG)

        if isinstance(prerout, dict):
            merged_cfg.update(prerout)
        if isinstance(fededa, dict):
            merged_cfg.update(fededa)

        merged_cfg.setdefault('fededa_gamma_size', 0.1)
        merged_cfg.setdefault('fededa_gamma_sigma', 0.001)
        merged_cfg.setdefault('fededa_gamma_p', 0.001)
        merged_cfg.setdefault('fedavg_aggregation_mode', 'equal')
        merged_cfg.setdefault('client_data_split_strategy', 'group_by_parent_circuit')
        merged_cfg.setdefault('fededa_p_estimator', 'rent_topo_fit')
        merged_cfg.setdefault('fededa_alpha_clip_max', 10.0)
        merged_cfg.setdefault('fededa_lambda_reduce', 'mean')
        merged_cfg.setdefault('fededa_drift_normalize', True)
        merged_cfg.setdefault('fededa_task_profile', 'prerout_timing')
        merged_cfg.setdefault('paper_label_key_drv', 'n_drv')
        merged_cfg.setdefault('paper_label_key_rc', 'n_rc')
        merged_cfg.setdefault('paper_label_key_wl', 'n_wl')
        merged_cfg.setdefault('paper_tasks', {
            'drv': {'label_key': 'n_drv', 'task_type': 'classification'},
            'rc': {'label_key': 'n_rc', 'task_type': 'regression'},
            'wl': {'label_key': 'n_wl', 'task_type': 'regression'},
        })

        super().__init__(
            num_clients=num_clients,
            local_epochs=local_epochs,
            epochs=epochs,
            lr=lr,
            num_select=num_select,
            parallel=parallel,
            prerout=merged_cfg,
            **kwargs,
        )

    def pretrain(self):
        super().pretrain()
        self._validate_task_profile()

    def _validate_task_profile(self):
        profile = str(getattr(self.clients[0].cfg, 'fededa_task_profile', 'prerout_timing')).lower() if self.clients else 'prerout_timing'
        if profile != 'paper':
            return

        if not self.clients:
            raise RuntimeError('FedEDA paper profile validation failed: no client is initialized.')

        sample_client = self.clients[0]
        if not sample_client.train_data:
            raise RuntimeError('FedEDA paper profile validation failed: empty training data.')

        sample_graph = next(iter(sample_client.train_data.values()))[0]
        ndata_keys = set(sample_graph.ndata.keys())

        tasks_cfg = getattr(sample_client.cfg, 'paper_tasks', None)
        if isinstance(tasks_cfg, dict):
            key_drv = str(tasks_cfg.get('drv', {}).get('label_key', 'n_drv'))
            key_rc = str(tasks_cfg.get('rc', {}).get('label_key', 'n_rc'))
            key_wl = str(tasks_cfg.get('wl', {}).get('label_key', 'n_wl'))
        else:
            key_drv = str(getattr(sample_client.cfg, 'paper_label_key_drv', 'n_drv'))
            key_rc = str(getattr(sample_client.cfg, 'paper_label_key_rc', 'n_rc'))
            key_wl = str(getattr(sample_client.cfg, 'paper_label_key_wl', 'n_wl'))

        missing = [k for k in (key_drv, key_rc, key_wl) if k not in ndata_keys]
        if missing:
            raise RuntimeError(
                'FedEDA paper profile requires DRV/RC/WL labels in graph ndata, '
                f'but missing keys: {missing}. '
                f'Current available ndata keys: {sorted(ndata_keys)}. '
                'This dataset is not the FedEDA-paper task dataset.'
            )
