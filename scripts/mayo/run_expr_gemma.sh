#!/usr/bin/env bash
# EXP-R for one development target (arguments: gpu, target key, tmux session to wait for): rediscovery, then SHIFT use
# of the transferred and rediscovered explanations.
set -uo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
RUNS="$ROOT/runs/cross_seed_models"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
passed() { grep -q '"PASS"' "$1/status.json" 2>/dev/null; }
while tmux has-session -t "=$3" 2>/dev/null; do sleep 60; done
# The first lane to start runs the smoke pair; both lanes need it to pass.
smoke_red=CSM_REDISCOVER_GEMMA2B_l0low16k_v1_20261011_SMOKE
smoke_use=CSM_EXPR_GEMMA2B_l0low16k_v1_20261011_SMOKE
mkdir -p "$ROOT/tmp/expr_claims"
if mkdir "$ROOT/tmp/expr_claims/smoke" 2>/dev/null; then
  CUDA_VISIBLE_DEVICES=$1 "$PY" scripts/mayo/gemma_rediscover.py --config scripts/mayo/configs/rediscover_gemma_smoke.json \
    > "$ROOT/logs/$smoke_red.log" 2>&1
  passed "$RUNS/$smoke_red" && CUDA_VISIBLE_DEVICES=$1 "$PY" scripts/mayo/gemma_task_use.py \
    --config scripts/mayo/configs/expr_gemma_use_smoke.json > "$ROOT/logs/$smoke_use.log" 2>&1
fi
until passed "$RUNS/$smoke_use"; do
  grep -qs '"FAIL"' "$RUNS/$smoke_red/status.json" "$RUNS/$smoke_use/status.json" && exit 1
  sleep 60
done
red="CSM_REDISCOVER_GEMMA2B_$2_v1_20261011"
passed "$RUNS/$red" || CUDA_VISIBLE_DEVICES=$1 "$PY" scripts/mayo/gemma_rediscover.py \
  --config "scripts/mayo/configs/rediscover_gemma_$2.json" > "$ROOT/logs/$red.log" 2>&1
passed "$RUNS/$red" || exit 1
use="CSM_EXPR_GEMMA2B_$2_v1_20261011"
resume=()
[ -e "$RUNS/$use" ] && resume=(--resume)
passed "$RUNS/$use" || CUDA_VISIBLE_DEVICES=$1 "$PY" scripts/mayo/gemma_task_use.py \
  --config "scripts/mayo/configs/expr_gemma_use_$2.json" "${resume[@]}" > "$ROOT/logs/$use.log" 2>&1
echo "EXP-R $2 finished"
