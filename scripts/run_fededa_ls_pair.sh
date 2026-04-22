#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Guard against invalid OpenMP/BLAS thread settings from external env.
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi
if ! [[ "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export MKL_NUM_THREADS=1
fi
if ! [[ "${OPENBLAS_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OPENBLAS_NUM_THREADS=1
fi

MODE="both"   # both | lsqs | ls
DRY_RUN=false
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="results/fededa_pair_logs/${RUN_TAG}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_fededa_ls_pair.sh [--mode both|lsqs|ls] [--dry-run]

Examples:
  bash scripts/run_fededa_ls_pair.sh
  bash scripts/run_fededa_ls_pair.sh --mode both
  bash scripts/run_fededa_ls_pair.sh --mode lsqs --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown arg: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "both" && "$MODE" != "lsqs" && "$MODE" != "ls" ]]; then
  echo "[ERROR] invalid --mode: $MODE"
  exit 1
fi

python scripts/generate_experiment_design_configs.py
python scripts/generate_experiment_design_ls_configs.py

run_lsqs() {
  local case_name="EXP1_fededa_lsqs"
  local cfg_path="configs/exp_design_ls+qs/EXP1_fededa_lsqs.json"
  if [[ ! -f "$cfg_path" ]]; then
    echo "[ERROR] config not found: $cfg_path"
    exit 1
  fi
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] ${case_name} -> ${cfg_path}"
    return
  fi
  mkdir -p "$LOG_DIR"
  local log_file="${LOG_DIR}/${case_name}.log"
  echo "[RUN] ${case_name} -> ${cfg_path}"
  python main.py "$cfg_path" 2>&1 | tee "$log_file"
  echo "[DONE] ${case_name}, log=${log_file}"
}

run_ls() {
  local case_name="EXP1_fededa_ls"
  local cfg_path="configs/exp_design_ls/EXP1_fededa_ls.json"
  if [[ ! -f "$cfg_path" ]]; then
    echo "[ERROR] config not found: $cfg_path"
    exit 1
  fi
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] ${case_name} -> ${cfg_path}"
    return
  fi
  mkdir -p "$LOG_DIR"
  local log_file="${LOG_DIR}/${case_name}.log"
  echo "[RUN] ${case_name} -> ${cfg_path}"
  python main.py "$cfg_path" 2>&1 | tee "$log_file"
  echo "[DONE] ${case_name}, log=${log_file}"
}

case "$MODE" in
  both)
    run_lsqs
    run_ls
    ;;
  lsqs)
    run_lsqs
    ;;
  ls)
    run_ls
    ;;
esac

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[INFO] dry-run only; no training executed."
else
  echo "[INFO] Finished. Logs saved to: $LOG_DIR"
fi
