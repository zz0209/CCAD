#!/usr/bin/env bash
# Part-level SHIFT in Gemma-2-2B with the saved correspondences, one target per GPU.
set -euo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/mayo/gemma_part_use.py --config scripts/mayo/configs/e4c_gemma_partuse_w65k.json > "$ROOT/logs/e4c_partuse_w65k.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/mayo/gemma_part_use.py --config scripts/mayo/configs/e4c_gemma_partuse_l0low16k.json > "$ROOT/logs/e4c_partuse_l0low16k.log" 2>&1 &
wait
echo "E4c part-level use finished"
