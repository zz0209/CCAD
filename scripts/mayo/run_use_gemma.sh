#!/usr/bin/env bash
# Task use of the state-fit maps for one development target (arguments: gpu, target key, tmux session to wait for).
set -uo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
RUNS="$ROOT/runs/cross_seed_models"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
passed() { grep -q '"PASS"' "$1/status.json" 2>/dev/null; }
while tmux has-session -t "=$3" 2>/dev/null; do sleep 60; done
name="CSM_USE_GEMMA2B_state_$2_v1_20261011"
passed "$RUNS/$name" && exit 0
CUDA_VISIBLE_DEVICES=$1 "$PY" scripts/mayo/gemma_task_use.py --config "scripts/mayo/configs/use_gemma_state_$2.json" \
  > "$ROOT/logs/$name.log" 2>&1
echo "Task use for $2 finished"
