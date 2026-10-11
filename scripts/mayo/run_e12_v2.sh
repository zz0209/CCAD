#!/usr/bin/env bash
# E12 version 2 (one-to-one gains up to four) on the five E10 grid targets. GPU 0 joins once the CONTAINMENT
# experiment 1 run on it has ended, then works through the configs in order; GPU 1 stays with the smaller
# Pythia and Qwen lanes, which leave too little memory for a second Gemma job.
set -uo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
RUNS="$ROOT/runs/cross_seed_models"
CLAIMS="$ROOT/tmp/e12_v2_claims"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
mkdir -p "$CLAIMS"
QUEUE=(w65k_lowest w16k_highest w65k_highest w16k_lower w16k_higher)
ended() { grep -qE '"(PASS|FAIL)"' "$1/status.json" 2>/dev/null; }
worker() {
  local gpu=$1 c1=$2
  until ended "$RUNS/$c1"; do sleep 60; done
  for key in "${QUEUE[@]}"; do
    mkdir "$CLAIMS/$key" 2>/dev/null || continue
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/mayo/gemma_shift_transfer.py --config "scripts/mayo/configs/e12_gemma_matched_${key}_v2.json" \
      > "$ROOT/logs/e12_gemma_matched_${key}_v2.log" 2>&1 || true
  done
}
worker 0 CSM_C1_GEMMA2B_natural_w65k_v1_20261011
echo "E12 version 2 finished"
