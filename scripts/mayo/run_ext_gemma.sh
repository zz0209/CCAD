#!/usr/bin/env bash
# READER_COVERAGE extension on one GPU (argument: gpu): a smoke run, the controls under the main objective for both Gemma
# targets, then their held-out readers. Finished runs are skipped.
set -uo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
RUNS="$ROOT/runs/cross_seed_models"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
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
transfer CSM_EXT_GEMMA2B_controls_w65k_v1_20261011_SMOKE "$1" ext_gemma_controls_smoke
passed "$RUNS/CSM_EXT_GEMMA2B_controls_w65k_v1_20261011_SMOKE" || exit 1
for key in w65k l0low16k; do
  transfer "CSM_EXT_GEMMA2B_controls_${key}_v1_20261011" "$1" "ext_gemma_controls_${key}"
done
for key in w65k l0low16k; do
  readers "CSM_EXT_GEMMA2B_readers_${key}_v1_20261011" "$1" "ext_gemma_readers_${key}"
done
echo "Extension finished"
