#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Defaults for this exact experiment
BASE_CFG="configs/exp_design_ls/EXP1_fededa_ls.json"
RESUME_CFG="configs/exp_design_ls/EXP1_fededa_ls_resume300.auto.json"
CKPT_PATH="/root/autodl-tmp/Fededa-PreRout/results/PreRoutFL_20260416_024910/checkpoints/epoch_0300.pt"
LOG_FILE="/root/autodl-tmp/Fededa-PreRout/results/fededa_pair_logs/20260416_011830/EXP1_fededa_ls.log"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/PreRoutFed/bin/python}"
RUN_MODE="fg"  # fg | bg

usage() {
  cat <<'EOF'
Usage:
  bash scripts/resume_exp1_fededa_ls_from300.sh [options]

Options:
  --ckpt <path>      Checkpoint path (default: epoch_0300.pt for this run)
  --log <path>       Log file path (default: EXP1_fededa_ls.log)
  --python <path>    Python executable (default: PreRoutFed env python)
  --fg               Run in foreground (default)
  --bg               Run in background and append log continuously
  -h, --help         Show help

Examples:
  bash scripts/resume_exp1_fededa_ls_from300.sh
  bash scripts/resume_exp1_fededa_ls_from300.sh --bg
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)
      CKPT_PATH="$2"
      shift 2
      ;;
    --log)
      LOG_FILE="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --fg)
      RUN_MODE="fg"
      shift
      ;;
    --bg)
      RUN_MODE="bg"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$BASE_CFG" ]]; then
  echo "[ERROR] Base config not found: $BASE_CFG"
  exit 1
fi

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "[ERROR] Checkpoint not found: $CKPT_PATH"
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python executable not found or not executable: $PYTHON_BIN"
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"

# Generate a resume config from the base config.
python3 - "$BASE_CFG" "$RESUME_CFG" "$CKPT_PATH" <<'PY'
import json
import sys

base_cfg, out_cfg, ckpt_path = sys.argv[1], sys.argv[2], sys.argv[3]

with open(base_cfg, "r", encoding="utf-8") as f:
    cfg = json.load(f)

pr = cfg.setdefault("train", {}).setdefault("prerout", {})
pr["resume_training"] = True
pr["resume_checkpoint_path"] = ckpt_path

with open(out_cfg, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY

CMD=("$PYTHON_BIN" -u main.py "$RESUME_CFG")

{
  echo ""
  echo "===== RESUME SCRIPT START $(date -u '+%Y-%m-%d %H:%M:%S UTC') ====="
  echo "ckpt=$CKPT_PATH"
  echo "cfg=$RESUME_CFG"
  echo "python=$PYTHON_BIN"
  echo "mode=$RUN_MODE"
} >> "$LOG_FILE"

if [[ "$RUN_MODE" == "bg" ]]; then
  # Keep a PTY in background mode to preserve stable flushing behavior.
  nohup script -q -f -a "$LOG_FILE" -c "${CMD[*]}" /dev/null \
    > "${LOG_FILE}.launcher.out" 2>&1 &
  PID=$!
  echo "$PID" > /tmp/fededa_exp1_resume300.pid
  echo "[OK] Started in background. script PID=$PID"
  echo "[INFO] log=$LOG_FILE"
  echo "[INFO] pid_file=/tmp/fededa_exp1_resume300.pid"
else
  "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
fi
