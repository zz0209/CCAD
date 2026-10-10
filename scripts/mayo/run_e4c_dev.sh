#!/usr/bin/env bash
# Gemma SHIFT transfer development: smoke test, then dev-split choice runs for both target dictionaries in parallel.
set -euo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/mayo/gemma_shift_transfer.py --config scripts/mayo/configs/e4c_gemma_smoke.json > "$ROOT/logs/e4c_smoke_v2.log" 2>&1
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/mayo/gemma_shift_transfer.py --config scripts/mayo/configs/e4c_gemma_dev_w65k.json > "$ROOT/logs/e4c_dev_w65k.log" 2>&1 &
until [ -f "$ROOT/assets/GEMMA_SCOPE_LOWEST_16k_MANIFEST.json" ]; do sleep 30; done
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/mayo/gemma_shift_transfer.py --config scripts/mayo/configs/e4c_gemma_dev_l0low16k.json > "$ROOT/logs/e4c_dev_l0low16k.log" 2>&1 &
wait
echo "E4c development runs finished"
