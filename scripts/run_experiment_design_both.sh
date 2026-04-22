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

GROUP="all"   # all | exp1 | exp2
MODE="both"   # both | lsqs | ls
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_experiment_design_both.sh [--group all|exp1|exp2] [--mode both|lsqs|ls] [--dry-run]

Examples:
  bash scripts/run_experiment_design_both.sh --group all
  bash scripts/run_experiment_design_both.sh --group exp1 --mode both
  bash scripts/run_experiment_design_both.sh --group exp2 --mode ls --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group)
      GROUP="$2"
      shift 2
      ;;
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

if [[ "$GROUP" != "all" && "$GROUP" != "exp1" && "$GROUP" != "exp2" ]]; then
  echo "[ERROR] invalid --group: $GROUP"
  exit 1
fi

if [[ "$MODE" != "both" && "$MODE" != "lsqs" && "$MODE" != "ls" ]]; then
  echo "[ERROR] invalid --mode: $MODE"
  exit 1
fi

EXTRA_ARGS=()
if [[ "$DRY_RUN" == "true" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

run_lsqs() {
  echo "[INFO] Running LS+QS experiment suite (run_experiment_design.sh), group=$GROUP"
  bash scripts/run_experiment_design.sh --group "$GROUP" "${EXTRA_ARGS[@]}"
}

run_ls() {
  echo "[INFO] Running LS experiment suite (run_experiment_design_ls.sh), group=$GROUP"
  bash scripts/run_experiment_design_ls.sh --group "$GROUP" "${EXTRA_ARGS[@]}"
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

echo "[INFO] Combined experiment run completed. mode=$MODE, group=$GROUP"
