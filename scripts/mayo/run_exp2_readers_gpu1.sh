#!/usr/bin/env bash
# The 16k EXP-2 held-out readers on GPU 1 (argument: gpu), started once both 16k reader-complete training runs have
# passed, in parallel with the 65k readers on GPU 0.
set -uo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
RUNS="$ROOT/runs/cross_seed_models"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
passed() { grep -q '"PASS"' "$1/status.json" 2>/dev/null; }
until passed "$RUNS/CSM_EXP2_GEMMA2B_state_l0low16k_ledoit_wolf_v1_20261011" && \
      passed "$RUNS/CSM_EXP2_GEMMA2B_state_l0low16k_standardized_v1_20261011"; do sleep 60; done
name=CSM_EXP2_GEMMA2B_readers_l0low16k_v2_20261011
resume=()
[ -e "$RUNS/$name" ] && resume=(--resume)
passed "$RUNS/$name" || CUDA_VISIBLE_DEVICES=$1 "$PY" scripts/mayo/gemma_readers.py \
  --config scripts/mayo/configs/exp2_gemma_readers_l0low16k_v2.json "${resume[@]}" > "$ROOT/logs/$name.log" 2>&1
echo "16k EXP-2 readers finished"
