#!/usr/bin/env bash
# Dictionary change of every E10 target, computed once all grid dictionaries are downloaded.
set -euo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
for m in 16k_next_lower_l0 16k_next_higher_l0 65k_lowest_l0 16k_highest_l0 65k_highest_l0; do
  until [ -f "$ROOT/assets/GEMMA_SCOPE_${m}_MANIFEST.json" ]; do sleep 60; done
done
CUDA_VISIBLE_DEVICES=${GPU:-1} "$PY" scripts/mayo/gemma_readback.py --config scripts/mayo/configs/e10_gemma_readback.json \
  ${RESUME:+--resume} >> "$ROOT/logs/e10_gemma_readback.log" 2>&1
echo "E10 read-back finished"
