#!/usr/bin/env bash
# Qwen3-1.7B fixed target-only correspondence (E9). Each argument is gpu:source:target, run side by side.
# Usage: run_e9_qwen.sh 1:2:1 0:1:2
set -euo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
export PYTHONPATH="$CODE/src:$CODE/scripts" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
for spec in "$@"; do
  IFS=: read -r gpu source target <<< "$spec"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/train_grammar_member_program.py \
    --config "scripts/mayo/configs/e9_qwen17_fixed_s${source}_t${target}.json" --target-seed "$target" \
    --source-seed "$source" > "$ROOT/logs/e9_qwen_s${source}_t${target}.log" 2>&1 &
done
wait
echo "E9 Qwen directions finished"
