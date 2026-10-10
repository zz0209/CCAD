#!/usr/bin/env bash
# Qwen3-1.7B public-SAE grammar study: preparation, label-free matching, then both transfer directions in parallel.
set -euo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
BULK="$ROOT/runs/naacl_sprint_20261010"
export PYTHONPATH="$CODE/src:$CODE/scripts" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
mkdir -p "$BULK"
if [ ! -f "$BULK/qwen17_prepare_l14_v2/PREPARE.json" ]; then
  CUDA_VISIBLE_DEVICES=${PREPARE_GPU:-0} "$PY" scripts/naacl_qwen_prepare.py --config scripts/mayo/configs/e4b_qwen17_prepare.json
fi
for s in 1 2; do
  if [ ! -f "$BULK/qwen17_matching_maps_l14_v2_s$s/MATCHING_MAPS.json" ]; then
    CUDA_VISIBLE_DEVICES=${PREPARE_GPU:-0} "$PY" scripts/naacl_grammar_matching_maps.py --config "scripts/mayo/configs/e4b_qwen17_maps_s$s.json"
  fi
done
run_direction() {
  local gpu=$1 source=$2 target=$3
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/train_grammar_member_program.py \
    --config "scripts/mayo/configs/e4b_qwen17_s${source}_t${target}.json" --target-seed "$target" --source-seed "$source"
}
run_direction 0 1 2 > "$ROOT/logs/e4b_s1_t2.log" 2>&1 &
run_direction 1 2 1 > "$ROOT/logs/e4b_s2_t1.log" 2>&1 &
wait
echo "E4b directions finished"
