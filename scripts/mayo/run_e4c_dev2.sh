#!/usr/bin/env bash
# Gemma SHIFT transfer development round 2: the random-subset supervision family on both targets.
set -euo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
RUNS="$ROOT/runs/naacl_sprint_20261010"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
until grep -q '"PASS"\|"FAIL"' "$RUNS/NAACL_E4C_GEMMA2B_dev_w65k_v1_20261010/status.json"; do sleep 20; done
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/mayo/gemma_shift_transfer.py --config scripts/mayo/configs/e4c_gemma_dev2_l0low16k.json > "$ROOT/logs/e4c_dev2_l0low16k.log" 2>&1 &
for d in S1_T2 S2_T1; do
  until grep -q '"PASS"\|"FAIL"' "$RUNS/NAACL_E4B_QWEN17_${d}_v4_20261010/status.json" 2>/dev/null; do sleep 20; done
done
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/mayo/gemma_shift_transfer.py --config scripts/mayo/configs/e4c_gemma_dev2_w65k.json > "$ROOT/logs/e4c_dev2_w65k.log" 2>&1 &
wait
echo "E4c development round 2 finished"
