#!/usr/bin/env bash
# GPU 0 after the Gemma capacity controls: READER_COVERAGE EXP-2 (a smoke run, the four reader-complete training runs and
# the held-out readers of both targets), then E12 version 2 on the five E10 grid targets. The earlier GPU 0 lane skips
# its E12 part because its claims were taken before it reached them. Finished runs are skipped.
set -uo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
RUNS="$ROOT/runs/cross_seed_models"
CLAIMS="$ROOT/tmp/e12_v2_claims_after_exp2"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
mkdir -p "$CLAIMS"
ended() { grep -qE '"(PASS|FAIL)"' "$1/status.json" 2>/dev/null; }
passed() { grep -q '"PASS"' "$1/status.json" 2>/dev/null; }
transfer() {
  passed "$RUNS/$1" && return 0
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/mayo/gemma_shift_transfer.py --config "scripts/mayo/configs/$2.json" > "$ROOT/logs/$1.log" 2>&1
}
readers() {
  passed "$RUNS/$1" && return 0
  local resume=()
  [ -e "$RUNS/$1" ] && resume=(--resume)
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/mayo/gemma_readers.py --config "scripts/mayo/configs/$2.json" "${resume[@]}" > "$ROOT/logs/$1.log" 2>&1
}
until ended "$RUNS/CSM_C5_GEMMA2B_capacity_l0low16k_v1_20261011"; do sleep 60; done
transfer CSM_EXP2_GEMMA2B_state_w65k_ledoit_wolf_v1_20261011_SMOKE exp2_gemma_state_smoke
if passed "$RUNS/CSM_EXP2_GEMMA2B_state_w65k_ledoit_wolf_v1_20261011_SMOKE"; then
  for key in w65k l0low16k; do
    for reader in ledoit_wolf standardized; do
      transfer "CSM_EXP2_GEMMA2B_state_${key}_${reader}_v1_20261011" "exp2_gemma_state_${key}_${reader}"
    done
  done
  for key in w65k l0low16k; do
    readers "CSM_EXP2_GEMMA2B_readers_${key}_v1_20261011" "exp2_gemma_readers_${key}"
  done
fi
for key in w65k_lowest w16k_highest w65k_highest w16k_lower w16k_higher; do
  mkdir "$CLAIMS/$key" 2>/dev/null || continue
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/mayo/gemma_shift_transfer.py --config "scripts/mayo/configs/e12_gemma_matched_${key}_v2.json" \
    > "$ROOT/logs/e12_gemma_matched_${key}_v2.log" 2>&1 || true
done
echo "GPU 0 EXP-2 lane finished"
