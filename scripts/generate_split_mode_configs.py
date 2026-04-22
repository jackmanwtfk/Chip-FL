#!/usr/bin/env python3
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CFG = ROOT / "configs" / "exp_design" / "EXP1_fedavg_lsqs.json"
OUT_DIR = ROOT / "configs" / "split_modes"
MANIFEST = OUT_DIR / "manifest.json"


def build_cfg(base_cfg, strategy):
    cfg = copy.deepcopy(base_cfg)
    train = cfg["train"]
    pr = train["prerout"]

    train["num_clients"] = 3
    pr["client_data_split_strategy"] = strategy
    pr["client_fixed_split_strict"] = True
    pr["resume_training"] = False
    pr["resume_checkpoint_path"] = ""
    pr["checkpoint_save_every"] = 10
    pr["checkpoint_keep_last"] = 2
    return cfg


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_cfg = json.loads(BASE_CFG.read_text(encoding="utf-8"))

    cases = [
        {
            "case": "LS_fixed_3clients",
            "desc": "LS split (5/5/5 designs across 3 clients, no overlap).",
            "strategy": "ls_fixed_3clients",
            "config": "LS_fixed_3clients.json",
        },
        {
            "case": "LS_QS_fixed_3clients",
            "desc": "LS+QS split (3/5/7 designs across 3 clients, no overlap).",
            "strategy": "ls_qs_fixed_3clients",
            "config": "LS_QS_fixed_3clients.json",
        },
    ]

    manifest = []
    for item in cases:
        cfg = build_cfg(base_cfg, item["strategy"])
        cfg_path = OUT_DIR / item["config"]
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest.append(
            {
                "case": item["case"],
                "desc": item["desc"],
                "strategy": item["strategy"],
                "config": f"configs/split_modes/{item['config']}",
            }
        )

    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated {len(manifest)} split-mode configs under: {OUT_DIR}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
