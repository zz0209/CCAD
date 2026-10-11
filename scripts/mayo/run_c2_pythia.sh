#!/usr/bin/env bash
# CONTAINMENT experiment 2 on one GPU (argument: gpu): the twelve Pythia-70M SHIFT program runs, then the four
# evaluations. Finished runs are skipped, so the lane resumes after an interruption.
set -uo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
RUNS="$ROOT/runs/cross_seed_models"
CFG=scripts/mayo/configs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf" PYTHONPATH="$CODE/src"
cd "$CODE/scripts"
passed() { grep -q '"PASS"' "$1/status.json" 2>/dev/null; }
gpu=$1
for target in 2 3 4 5; do
  for seed in 42 43 44; do
    name=CSM_C2_PYTHIA70M_program_t${target}_ts${seed}_v1_20261011
    passed "$RUNS/$name" && continue
    CUDA_VISIBLE_DEVICES=$gpu "$PY" train_shift_program.py --config "../$CFG/c2_pythia_program_t${target}_ts${seed}.json" \
      > "$ROOT/logs/$name.log" 2>&1
  done
done
for target in 2 3 4 5; do
  name=CSM_C2_PYTHIA70M_eval_t${target}_v1_20261011
  passed "$RUNS/$name" && continue
  CUDA_VISIBLE_DEVICES=$gpu "$PY" run_shift_transfer.py --config "../$CFG/c2_pythia_eval_t${target}.json" \
    > "$ROOT/logs/$name.log" 2>&1
done
echo "C2 finished"
