#!/usr/bin/env bash
# Final test readers for one development target (arguments: gpu, target key, tmux session to wait for).
set -uo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
RUNS="$ROOT/runs/cross_seed_models"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
passed() { grep -q '"PASS"' "$1/status.json" 2>/dev/null; }
while tmux has-session -t "=$3" 2>/dev/null; do sleep 60; done
until passed "$RUNS/CSM_EXT_GEMMA2B_controls_$2_v1_20261011"; do sleep 60; done
name="CSM_FINAL_GEMMA2B_readers_$2_v1_20261011"
passed "$RUNS/$name" && exit 0
resume=()
[ -e "$RUNS/$name" ] && resume=(--resume)
CUDA_VISIBLE_DEVICES=$1 "$PY" scripts/mayo/gemma_readers.py --config "scripts/mayo/configs/final_gemma_readers_$2.json" "${resume[@]}" \
  > "$ROOT/logs/$name.log" 2>&1
echo "Final readers for $2 finished"
