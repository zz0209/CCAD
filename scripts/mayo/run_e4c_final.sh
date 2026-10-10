#!/usr/bin/env bash
# Gemma SHIFT transfer: dev check of variable-size subsets, then confirmation on the balanced test split on both GPUs.
set -euo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
RUNS="$ROOT/runs/naacl_sprint_20261010"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
ended() { grep -q '"PASS"\|"FAIL"' "$RUNS/$1/status.json" 2>/dev/null; }
run() { CUDA_VISIBLE_DEVICES=$1 "$PY" scripts/mayo/gemma_shift_transfer.py --config "scripts/mayo/configs/$2.json" > "$ROOT/logs/$2.log" 2>&1; }
{
  run 1 e4c_gemma_dev3_l0low16k
  run 1 e4c_gemma_final_l0low16k
} &
until ended NAACL_E4C_GEMMA2B_dev2_w65k_v1_20261010 && ended NAACL_E4C_GEMMA2B_dev3_l0low16k_v1_20261010; do sleep 20; done
run 0 e4c_gemma_final_w65k &
wait
echo "E4c dev3 and final runs finished"
