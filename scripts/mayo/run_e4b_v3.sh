#!/usr/bin/env bash
# Qwen3-1.7B grammar transfer at the development-chosen learning rate, confirmed on a further unused BLiMP panel.
set -euo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
export PYTHONPATH="$CODE/src:$CODE/scripts" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
run_direction() {
  local gpu=$1 source=$2 target=$3
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/train_grammar_member_program.py \
    --config "scripts/mayo/configs/e4b_qwen17_s${source}_t${target}_v3.json" --target-seed "$target" --source-seed "$source"
}
run_direction "${GPU_A:-0}" 1 2 > "$ROOT/logs/e4b_v3_s1_t2.log" 2>&1 &
run_direction "${GPU_B:-1}" 2 1 > "$ROOT/logs/e4b_v3_s2_t1.log" 2>&1 &
wait
echo "E4b v3 directions finished"
