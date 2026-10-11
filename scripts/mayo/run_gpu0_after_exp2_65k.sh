#!/usr/bin/env bash
# GPU 0 after the 65k EXP-2 readers: the GPU 0 EXP-2 lane is stopped (the 16k readers run on GPU 1 as version 2), a
# version 1 16k reader directory it may have started in the last minute is renamed as abandoned, and E12 version 2 runs
# on the five E10 grid targets.
set -uo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
RUNS="$ROOT/runs/cross_seed_models"
CLAIMS="$ROOT/tmp/e12_v2_claims_gpu0_final"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
mkdir -p "$CLAIMS"
passed() { grep -q '"PASS"' "$1/status.json" 2>/dev/null; }
until passed "$RUNS/CSM_EXP2_GEMMA2B_readers_w65k_v1_20261011"; do sleep 30; done
tmux kill-session -t '=ccad-exp2'
sleep 10
v1="$RUNS/CSM_EXP2_GEMMA2B_readers_l0low16k_v1_20261011"
[ -e "$v1" ] && mv "$v1" "$RUNS/abandoned_CSM_EXP2_GEMMA2B_readers_l0low16k_v1_20261011_$(date -u +%H%M%S)"
for key in w65k_lowest w16k_highest w65k_highest w16k_lower w16k_higher; do
  mkdir "$CLAIMS/$key" 2>/dev/null || continue
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/mayo/gemma_shift_transfer.py --config "scripts/mayo/configs/e12_gemma_matched_${key}_v2.json" \
    > "$ROOT/logs/e12_gemma_matched_${key}_v2.log" 2>&1 || true
done
echo "GPU 0 E12 version 2 finished"
