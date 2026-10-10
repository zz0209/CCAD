#!/usr/bin/env bash
# E8 Gemma runs: label-free request families (E8a), then part-level SHIFT on later classifiers (E8b).
# Usage: run_e8.sh smoke | run_e8.sh full
set -euo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
run() { CUDA_VISIBLE_DEVICES=$1 "$PY" "scripts/mayo/$2.py" --config "scripts/mayo/configs/$3.json" > "$ROOT/logs/$3.log" 2>&1; }
if [ "$1" = smoke ]; then
  run "$2" gemma_shift_transfer e8a_gemma_smoke_labelfree
  run "$2" gemma_task_use e8b_gemma_smoke_taskuse
  echo "E8 smoke finished"
else
  { run 0 gemma_shift_transfer e8a_gemma_labelfree_w65k; run 0 gemma_task_use e8b_gemma_taskuse_w65k; } &
  { run 1 gemma_shift_transfer e8a_gemma_labelfree_l0low16k; run 1 gemma_task_use e8b_gemma_taskuse_l0low16k; } &
  wait
  echo "E8 runs finished"
fi
