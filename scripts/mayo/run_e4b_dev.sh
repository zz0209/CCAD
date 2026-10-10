#!/usr/bin/env bash
# Development learning-rate selection for the Qwen3-1.7B study: four runs, two per GPU.
set -euo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
export PYTHONPATH="$CODE/src:$CODE/scripts" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
dev() {
  local gpu=$1 lr=$2 source=$3 target=$4
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/train_grammar_member_program.py \
    --config "scripts/mayo/configs/e4b_qwen17_dev_lr${lr}_s${source}_t${target}.json" --target-seed "$target" --source-seed "$source" \
    > "$ROOT/logs/e4b_dev_lr${lr}_s${source}_t${target}.log" 2>&1
}
dev 0 1e-5 1 2 &
dev 0 3e-5 1 2 &
dev 1 1e-5 2 1 &
dev 1 3e-5 2 1 &
wait
echo "E4b development runs finished"
