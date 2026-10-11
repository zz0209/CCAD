#!/usr/bin/env bash
# READER_COVERAGE confirmation lane on one GPU (arguments: gpu, tmux session to wait for). After that session ends, the
# first lane runs the smoke pair; every lane then claims grid targets one at a time and runs the training run and its
# test readers. Finished runs are skipped.
set -uo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
RUNS="$ROOT/runs/cross_seed_models"
CLAIMS="$ROOT/tmp/conf_claims"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
mkdir -p "$CLAIMS"
passed() { grep -q '"PASS"' "$1/status.json" 2>/dev/null; }
transfer() {
  passed "$RUNS/$1" && return 0
  CUDA_VISIBLE_DEVICES=$2 "$PY" scripts/mayo/gemma_shift_transfer.py --config "scripts/mayo/configs/$3.json" > "$ROOT/logs/$1.log" 2>&1
}
readers() {
  passed "$RUNS/$1" && return 0
  local resume=()
  [ -e "$RUNS/$1" ] && resume=(--resume)
  CUDA_VISIBLE_DEVICES=$2 "$PY" scripts/mayo/gemma_readers.py --config "scripts/mayo/configs/$3.json" "${resume[@]}" > "$ROOT/logs/$1.log" 2>&1
}
while tmux has-session -t "=$2" 2>/dev/null; do sleep 60; done
if mkdir "$CLAIMS/smoke" 2>/dev/null; then
  transfer CSM_CONF_GEMMA2B_w16k_lower_v1_20261011_SMOKE "$1" conf_gemma_smoke
  passed "$RUNS/CSM_CONF_GEMMA2B_w16k_lower_v1_20261011_SMOKE" && \
    readers CSM_CONF_GEMMA2B_readers_w16k_lower_v1_20261011_SMOKE "$1" conf_gemma_readers_smoke
  passed "$RUNS/CSM_CONF_GEMMA2B_readers_w16k_lower_v1_20261011_SMOKE" && touch "$CLAIMS/smoke_passed"
fi
until [ -e "$CLAIMS/smoke_passed" ]; do
  grep -qs '"FAIL"' "$RUNS/CSM_CONF_GEMMA2B_w16k_lower_v1_20261011_SMOKE/status.json" \
    "$RUNS/CSM_CONF_GEMMA2B_readers_w16k_lower_v1_20261011_SMOKE/status.json" && exit 1
  sleep 60
done
for key in w16k_lower w16k_higher w65k_lowest w16k_highest w65k_highest; do
  mkdir "$CLAIMS/$key" 2>/dev/null || continue
  transfer "CSM_CONF_GEMMA2B_${key}_v1_20261011" "$1" "conf_gemma_${key}"
  passed "$RUNS/CSM_CONF_GEMMA2B_${key}_v1_20261011" && \
    readers "CSM_CONF_GEMMA2B_readers_${key}_v1_20261011" "$1" "conf_gemma_readers_${key}"
done
echo "Confirmation lane on GPU $1 finished"
