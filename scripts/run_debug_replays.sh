#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/PreRoutFed/bin/python}"
MODE="${1:-}"

if [[ "$MODE" != "ls" && "$MODE" != "lsqs" && "$MODE" != "all" && "$MODE" != "" ]]; then
  echo "Usage: bash scripts/run_debug_replays.sh [ls|lsqs|all]"
  exit 1
fi

"$PYTHON_BIN" scripts/generate_debug_replay_configs.py

mapfile -t CONFIGS < <(
  "$PYTHON_BIN" - "$MODE" <<'PY'
import json
import sys

with open("configs/debug_replay_manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)

mode = (sys.argv[1] if len(sys.argv) > 1 else "").strip()
for item in manifest:
    split = item.get("split", "")
    if mode == "ls" and split != "ls":
        continue
    if mode == "lsqs" and split != "lsqs":
        continue
    print(item["replay_config"])
PY
)

for cfg in "${CONFIGS[@]}"; do
  echo "===== RUN $cfg ====="
  "$PYTHON_BIN" -u main.py "$cfg"
done
