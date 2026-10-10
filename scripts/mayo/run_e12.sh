#!/usr/bin/env bash
# E12 matched-gain baseline on the E10 grid targets. A GPU joins once every E10 grid target has been claimed and no
# E10 process has run on that GPU for two checks three minutes apart; it then claims the next E12 config in order.
set -euo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
CLAIMS="$ROOT/tmp/e12_claims"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
mkdir -p "$CLAIMS"
QUEUE=(w16k_lower w16k_higher w65k_lowest w16k_highest w65k_highest)
e10_on_gpu() {
  local uuid
  uuid=$(nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader)
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$uuid" '$1==u {print $2}'); do
    ps -o args= -p "$p" | grep -q e10_gemma_grid && return 0
  done
  return 1
}
gpu_free() { [ "$(ls "$ROOT/tmp/e10_claims" | wc -l)" -ge 5 ] && ! e10_on_gpu "$1"; }
worker() {
  local gpu=$1
  until gpu_free "$gpu" && sleep 180 && gpu_free "$gpu"; do sleep 60; done
  for key in "${QUEUE[@]}"; do
    mkdir "$CLAIMS/$key" 2>/dev/null || continue
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/mayo/gemma_shift_transfer.py --config "scripts/mayo/configs/e12_gemma_matched_$key.json" \
      > "$ROOT/logs/e12_gemma_matched_$key.log" 2>&1 || true
  done
}
worker 0 &
worker 1 &
wait
echo "E12 finished"
