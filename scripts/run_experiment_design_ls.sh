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
DRY_RUN=false
RUN_TAG="$(date +%Y%m%d_%H%M%S)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_experiment_design_ls.sh [--group all|exp1|exp2] [--dry-run]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group)
      GROUP="$2"
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

python scripts/generate_experiment_design_ls_configs.py

MANIFEST="configs/exp_design_ls/manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
  echo "[ERROR] manifest not found: $MANIFEST"
  exit 1
fi

mapfile -t CASE_LINES < <(python - "$GROUP" <<'PY'
import json
import sys
from pathlib import Path

group = sys.argv[1]
items = json.loads(Path("configs/exp_design_ls/manifest.json").read_text(encoding="utf-8"))
for it in items:
    if group == "all":
        if it.get("group") not in ("exp1", "exp2"):
            continue
    elif it.get("group") != group:
        continue
    print(f"{it['case']} {it['config']}")
PY
)

if [[ ${#CASE_LINES[@]} -eq 0 ]]; then
  echo "[ERROR] no cases selected for group=$GROUP"
  exit 1
fi

echo "[INFO] Selected LS group=$GROUP, cases:"
printf '%s\n' "${CASE_LINES[@]}"

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[INFO] dry-run only; no training executed."
  exit 0
fi

LOG_DIR="results/exp_design_ls_logs/${RUN_TAG}_${GROUP}"
mkdir -p "$LOG_DIR"

for line in "${CASE_LINES[@]}"; do
  case_name="${line%% *}"
  cfg_path="${line#* }"
  log_file="${LOG_DIR}/${case_name}.log"
  echo "[RUN] ${case_name} -> ${cfg_path}"
  python main.py "$cfg_path" 2>&1 | tee "$log_file"
  echo "[DONE] ${case_name}, log=${log_file}"
done

echo "[INFO] LS experiment-design runs completed."
