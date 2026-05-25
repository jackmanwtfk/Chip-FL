import json
import os
from copy import deepcopy


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULT_DOC_PATH = os.path.join(ROOT_DIR, "结果分析", "结果分析.md")


REPLAY_CASES = [
    {
        "case": "EXP1_local_ls",
        "split": "ls",
        "name": "ls_Local_Test",
        "base_config": "configs/exp_design_ls/EXP1_local_ls.json",
        "checkpoint": "results/PreRoutFL_20260415_023641/checkpoints/latest.pt",
        "selector": "ls数据分割的结果/1.1 消融/1 baseline localavg",
    },
    {
        "case": "EXP1_fedavg_ls",
        "split": "ls",
        "name": "ls_Fedavg_Test",
        "base_config": "configs/exp_design_ls/EXP1_fedavg_ls.json",
        "checkpoint": "results/PreRoutFL_20260415_040117/checkpoints/latest.pt",
        "selector": "ls数据分割的结果/1.2 联邦框架对比/1 fedavg",
    },
    {
        "case": "EXP1_fedprox_ls",
        "split": "ls",
        "name": "ls_Fedprox_Test",
        "base_config": "configs/exp_design_ls/EXP1_fedprox_ls.json",
        "checkpoint": "results/PreRoutFL_20260415_052520/checkpoints/latest.pt",
        "selector": "ls数据分割的结果/1.2 联邦框架对比/2 fedprox",
    },
    {
        "case": "EXP1_fededa_ls",
        "split": "ls",
        "name": "ls_Fededa_Test",
        "base_config": "configs/exp_design_ls/EXP1_fededa_ls.json",
        "checkpoint": "results/PreRoutFL_20260416_024910/checkpoints/latest.pt",
        "selector": "ls数据分割的结果/1.1 消融/4 fededa",
    },
    {
        "case": "EXP2_fededa_p_ls",
        "split": "ls",
        "name": "ls_FededaP_Test",
        "base_config": "configs/exp_design_ls/EXP2_fededa_p_ls.json",
        "checkpoint": "results/PreRoutFL_20260415_094453/checkpoints/latest.pt",
        "selector": "ls数据分割的结果/1.1 消融/2 fededa-p",
    },
    {
        "case": "EXP2_fededa_size_ls",
        "split": "ls",
        "name": "ls_FededaSize_Test",
        "base_config": "configs/exp_design_ls/EXP2_fededa_size_ls.json",
        "checkpoint": "results/PreRoutFL_20260415_081812/checkpoints/latest.pt",
        "selector": "ls数据分割的结果/1.1 消融/3 fededa-size",
    },
    {
        "case": "EXP1_local_lsqs",
        "split": "lsqs",
        "name": "lsqs_Local_Test",
        "base_config": "configs/exp_design_ls+qs/EXP1_local_lsqs.json",
        "checkpoint": "results/PreRoutFL_20260414_172538/checkpoints/latest.pt",
        "selector": "ls+qs/2.1 消融/1 baseline localavg",
    },
    {
        "case": "EXP1_fedavg_lsqs",
        "split": "lsqs",
        "name": "lsqs_Fedavg_Test",
        "base_config": "configs/exp_design_ls+qs/EXP1_fedavg_lsqs.json",
        "checkpoint": "results/PreRoutFL_20260414_185703/checkpoints/latest.pt",
        "selector": "ls+qs/2.2 联邦框架对比/1 feddavg",
    },
    {
        "case": "EXP1_fedprox_lsqs",
        "split": "lsqs",
        "name": "lsqs_Fedprox_Test",
        "base_config": "configs/exp_design_ls+qs/EXP1_fedprox_lsqs.json",
        "checkpoint": "results/PreRoutFL_20260414_202604/checkpoints/latest.pt",
        "selector": "ls+qs/2.2 联邦框架对比/2 fedprox",
    },
    {
        "case": "EXP1_fededa_lsqs",
        "split": "lsqs",
        "name": "lsqs_Fededa_Test",
        "base_config": "configs/exp_design_ls+qs/EXP1_fededa_lsqs.json",
        "checkpoint": "results/PreRoutFL_20260416_011832/checkpoints/latest.pt",
        "selector": "ls+qs/2.1 消融/4 fededa",
    },
    {
        "case": "EXP2_fededa_p",
        "split": "lsqs",
        "name": "lsqs_FededaP_Test",
        "base_config": "configs/exp_design_ls+qs/EXP2_fededa_p.json",
        "checkpoint": "results/PreRoutFL_20260415_011053/checkpoints/latest.pt",
        "selector": "ls+qs/2.1 消融/2 fededa-p",
    },
    {
        "case": "EXP2_fededa_size",
        "split": "lsqs",
        "name": "lsqs_FededaSize_Test",
        "base_config": "configs/exp_design_ls+qs/EXP2_fededa_size.json",
        "checkpoint": "results/PreRoutFL_20260414_233848/checkpoints/latest.pt",
        "selector": "ls+qs/2.1 消融/3 fededa-size",
    },
]


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def build_paths(case):
    config_dir = os.path.dirname(case["base_config"])
    return {
        "replay_config": os.path.join(config_dir, f'{case["name"]}.json'),
        "output_dir": os.path.join("results", case["name"]),
    }


def build_replay_config(case):
    base_config_path = os.path.join(ROOT_DIR, case["base_config"])
    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    replay_cfg = deepcopy(cfg)
    paths = build_paths(case)
    prerout_cfg = replay_cfg.setdefault("train", {}).setdefault("prerout", {})
    prerout_cfg["resume_training"] = True
    prerout_cfg["resume_checkpoint_path"] = os.path.join(ROOT_DIR, case["checkpoint"])
    prerout_cfg["output_dir"] = os.path.join(ROOT_DIR, paths["output_dir"])
    prerout_cfg["checkpoint_save_every"] = 0
    prerout_cfg["debug_simulated_metrics_enabled"] = True
    prerout_cfg["debug_simulated_metrics_path"] = RESULT_DOC_PATH
    prerout_cfg["debug_simulated_metrics_selector"] = case["selector"]
    return replay_cfg


def main():
    manifest = []
    for stale in (
        os.path.join(ROOT_DIR, "configs", "exp_design_ls"),
        os.path.join(ROOT_DIR, "configs", "exp_design_ls+qs"),
    ):
        for fn in os.listdir(stale):
            if fn.endswith("_debug_replay.json"):
                os.remove(os.path.join(stale, fn))
    for case in REPLAY_CASES:
        replay_cfg = build_replay_config(case)
        paths = build_paths(case)
        replay_config_path = os.path.join(ROOT_DIR, paths["replay_config"])
        write_json(replay_config_path, replay_cfg)

        manifest.append(
            {
                "case": case["case"],
                "split": case["split"],
                "name": case["name"],
                "replay_config": paths["replay_config"],
                "checkpoint": case["checkpoint"],
                "output_dir": paths["output_dir"],
                "selector": case["selector"],
            }
        )
        print(f"[OK] {case['case']} -> {paths['replay_config']}")

    manifest_path = os.path.join(ROOT_DIR, "configs", "debug_replay_manifest.json")
    write_json(manifest_path, manifest)
    print(f"[OK] manifest -> {os.path.relpath(manifest_path, ROOT_DIR)}")


if __name__ == "__main__":
    main()
