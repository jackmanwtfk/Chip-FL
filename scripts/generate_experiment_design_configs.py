#!/usr/bin/env python3
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# Keep generator self-contained within LS+QS experiment-design ecosystem.
BASE_CFG = ROOT / "configs" / "exp_design_ls+qs" / "EXP1_fedavg_lsqs.json"
OUT_DIR = ROOT / "configs" / "exp_design_ls+qs"
MANIFEST = OUT_DIR / "manifest.json"


def deep_set(d, path, value):
    cur = d
    for key in path[:-1]:
        cur = cur[key]
    cur[path[-1]] = value


def build_case(base_cfg, case):
    cfg = copy.deepcopy(base_cfg)
    train = cfg["train"]
    pr = train["prerout"]

    # Unified baseline for this experimental suite
    train["num_clients"] = 3
    pr["client_data_split_strategy"] = "ls_qs_fixed_3clients"
    pr["client_fixed_split_strict"] = True
    pr["resume_training"] = False
    pr["resume_checkpoint_path"] = ""
    pr["checkpoint_save_every"] = 10
    pr["checkpoint_keep_last"] = 2
    pr["fedavg_aggregation_mode"] = "equal"
    pr["fl_algorithm"] = "fedavg"
    pr["fedprox_mu"] = 0.0
    pr["fededa_gamma_size"] = 0.0
    pr["fededa_gamma_p"] = 0.0
    pr["fededa_drift_normalize"] = True
    loss_weights = dict(pr.get("loss_weights", {}))
    loss_weights["slew"] = 1.5
    pr["loss_weights"] = loss_weights
    pr["slew_loss_type"] = "huber"
    pr["slew_huber_beta"] = 0.2
    pr["celldelay_loss_type"] = "huber"
    pr["celldelay_huber_beta"] = 0.2
    pr["sub_graph_size"] = 50000

    name = case["case"]
    if name == "EXP1_local_lsqs":
        pr["fl_algorithm"] = "local"
    elif name == "EXP1_fedavg_lsqs":
        pr["fl_algorithm"] = "fedavg"
    elif name == "EXP1_fedprox_lsqs":
        pr["fl_algorithm"] = "fedprox"
        pr["fedprox_mu"] = 0.001
    elif name == "EXP1_fededa_lsqs":
        pr["fl_algorithm"] = "fedavg"
        pr["fededa_gamma_size"] = 0.1
        pr["fededa_gamma_p"] = 0.001
    elif name == "EXP2_fededa_size":
        pr["fededa_gamma_size"] = 0.1
    elif name == "EXP2_fededa_p":
        pr["fededa_gamma_p"] = 0.001
    elif name == "EXP3_fededa_size_p_nopart":
        pr["fededa_gamma_size"] = 0.1
        pr["fededa_gamma_p"] = 0.001
        pr["sub_graph_size"] = 0
    elif name == "EXP3_fededa_size_p_part":
        pr["fededa_gamma_size"] = 0.1
        pr["fededa_gamma_p"] = 0.001
        pr["sub_graph_size"] = 50000
    else:
        raise ValueError(f"Unknown case: {name}")

    return cfg


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_cfg = json.loads(BASE_CFG.read_text(encoding="utf-8"))

    cases = [
        {
            "case": "EXP1_local_lsqs",
            "group": "exp1",
            "desc": "Local learning averaged across 3 clients under LS+QS split.",
        },
        {
            "case": "EXP1_fedavg_lsqs",
            "group": "exp1",
            "desc": "FedAvg baseline under LS+QS split.",
        },
        {
            "case": "EXP1_fedprox_lsqs",
            "group": "exp1",
            "desc": "FedProx(mu=0.001) under LS+QS split.",
        },
        {
            "case": "EXP1_fededa_lsqs",
            "group": "exp1",
            "desc": "FedEDA(size+p) under LS+QS split.",
        },
        {
            "case": "EXP2_fededa_size",
            "group": "exp2",
            "desc": "Metadata ablation: size only.",
        },
        {
            "case": "EXP2_fededa_p",
            "group": "exp2",
            "desc": "Metadata ablation: p only.",
        },
        {
            "case": "EXP3_fededa_size_p_nopart",
            "group": "exp3",
            "desc": "Partition ablation: no partition (sub_graph_size=0) + FedEDA(size+p).",
        },
        {
            "case": "EXP3_fededa_size_p_part",
            "group": "exp3",
            "desc": "Partition ablation: partition (sub_graph_size=50000) + FedEDA(size+p).",
        },
    ]

    manifest = []
    for item in cases:
        cfg = build_case(base_cfg, item)
        cfg_rel = Path("configs") / "exp_design_ls+qs" / f"{item['case']}.json"
        cfg_path = ROOT / cfg_rel
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest.append(
            {
                "case": item["case"],
                "group": item["group"],
                "desc": item["desc"],
                "config": str(cfg_rel).replace("\\", "/"),
            }
        )

    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated {len(manifest)} configs under: {OUT_DIR}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
