#!/usr/bin/env bash
# GPU 0 after CONTAINMENT experiment 1 on it has ended: a smoke run and the two Gemma capacity controls of experiment 5,
# then E12 version 2 (one-to-one gains up to four) on the five E10 grid targets in order. GPU 1 stays with the smaller
# Pythia, Qwen and GPT-2 lanes, which leave too little memory for a second Gemma job.
set -uo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
RUNS="$ROOT/runs/cross_seed_models"
CLAIMS="$ROOT/tmp/e12_v2_claims"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
mkdir -p "$CLAIMS"
ended() { grep -qE '"(PASS|FAIL)"' "$1/status.json" 2>/dev/null; }
passed() { grep -q '"PASS"' "$1/status.json" 2>/dev/null; }
gemma() {
  passed "$2/$1" && return 0
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/mayo/gemma_shift_transfer.py --config "scripts/mayo/configs/$3.json" > "$ROOT/logs/$1.log" 2>&1
}
until ended "$RUNS/CSM_C1_GEMMA2B_natural_w65k_v1_20261011"; do sleep 60; done
gemma CSM_C5_GEMMA2B_capacity_w65k_v1_20261011_SMOKE "$RUNS" c5_gemma_capacity_w65k_smoke
if passed "$RUNS/CSM_C5_GEMMA2B_capacity_w65k_v1_20261011_SMOKE"; then
  gemma CSM_C5_GEMMA2B_capacity_w65k_v1_20261011 "$RUNS" c5_gemma_capacity_w65k
  gemma CSM_C5_GEMMA2B_capacity_l0low16k_v1_20261011 "$RUNS" c5_gemma_capacity_l0low16k
fi
for key in w65k_lowest w16k_highest w65k_highest w16k_lower w16k_higher; do
  mkdir "$CLAIMS/$key" 2>/dev/null || continue
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/mayo/gemma_shift_transfer.py --config "scripts/mayo/configs/e12_gemma_matched_${key}_v2.json" \
    > "$ROOT/logs/e12_gemma_matched_${key}_v2.log" 2>&1 || true
done
echo "GPU 0 lane finished"
