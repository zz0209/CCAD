#!/usr/bin/env bash
# READER_COVERAGE EXP-1 on one GPU (argument: gpu): a smoke run, then the held-out reader evaluation of both Gemma
# targets in order. A failed or interrupted run resumes from its partial store.
set -uo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
RUNS="$ROOT/runs/cross_seed_models"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
passed() { grep -q '"PASS"' "$1/status.json" 2>/dev/null; }
readers() {
  passed "$RUNS/$1" && return 0
  local resume=()
  [ -e "$RUNS/$1" ] && resume=(--resume)
  CUDA_VISIBLE_DEVICES=$2 "$PY" scripts/mayo/gemma_readers.py --config "scripts/mayo/configs/$3.json" "${resume[@]}" \
    > "$ROOT/logs/$1.log" 2>&1
}
readers CSM_EXP1_GEMMA2B_readers_w65k_v1_20261011_SMOKE "$1" exp1_gemma_readers_w65k_smoke
passed "$RUNS/CSM_EXP1_GEMMA2B_readers_w65k_v1_20261011_SMOKE" || exit 1
readers CSM_EXP1_GEMMA2B_readers_w65k_v1_20261011 "$1" exp1_gemma_readers_w65k
readers CSM_EXP1_GEMMA2B_readers_l0low16k_v1_20261011 "$1" exp1_gemma_readers_l0low16k
echo "EXP-1 Gemma finished"
