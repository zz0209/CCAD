#!/usr/bin/env bash
# EXP-U Gemma unified arm for one development target (arguments: gpu, target key, tmux session to wait for). The first
# lane to start runs the smoke run; both lanes then train their target and read it on development biographies.
set -uo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
RUNS="$ROOT/runs/cross_seed_models"
CLAIMS="$ROOT/tmp/u_gemma_claims"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
mkdir -p "$CLAIMS"
passed() { grep -q '"PASS"' "$1/status.json" 2>/dev/null; }
while tmux has-session -t "=$3" 2>/dev/null; do sleep 60; done
smoke=CSM_U_GEMMA2B_w65k_v1_20261011_SMOKE
if mkdir "$CLAIMS/smoke" 2>/dev/null; then
  CUDA_VISIBLE_DEVICES=$1 "$PY" scripts/mayo/gemma_shift_transfer.py --config scripts/mayo/configs/u_gemma_smoke.json \
    > "$ROOT/logs/$smoke.log" 2>&1
fi
until passed "$RUNS/$smoke"; do
  grep -qs '"FAIL"' "$RUNS/$smoke/status.json" && exit 1
  sleep 60
done
name="CSM_U_GEMMA2B_$2_v1_20261011"
passed "$RUNS/$name" || CUDA_VISIBLE_DEVICES=$1 "$PY" scripts/mayo/gemma_shift_transfer.py \
  --config "scripts/mayo/configs/u_gemma_$2.json" > "$ROOT/logs/$name.log" 2>&1
passed "$RUNS/$name" || exit 1
readers="CSM_U_GEMMA2B_readers_$2_v1_20261011"
resume=()
[ -e "$RUNS/$readers" ] && resume=(--resume)
passed "$RUNS/$readers" || CUDA_VISIBLE_DEVICES=$1 "$PY" scripts/mayo/gemma_readers.py \
  --config "scripts/mayo/configs/u_gemma_readers_$2.json" "${resume[@]}" > "$ROOT/logs/$readers.log" 2>&1
echo "EXP-U Gemma $2 finished"
