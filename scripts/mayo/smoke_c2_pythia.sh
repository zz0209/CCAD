#!/usr/bin/env bash
# Smoke test of CONTAINMENT experiment 2 on one GPU (argument: gpu): four training steps per arm, then the evaluation
# of those checkpoints on two requests. An earlier smoke run directory is renamed with a suffix before the new run.
set -uo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
PY="$ROOT/env/bin/python"
RUNS="$ROOT/runs/cross_seed_models"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf" PYTHONPATH="$ROOT/code/CCAD/src"
cd "$ROOT/code/CCAD/scripts"
stamp=$(date -u +%H%M%S)
for name in CSM_C2_PYTHIA70M_program_t2_ts42_v1_20261011_SMOKE CSM_C2_PYTHIA70M_eval_t2_v1_20261011_SMOKE c2_programs/t2_ts42_smoke; do
  [ -e "$RUNS/$name" ] && mv "$RUNS/$name" "$RUNS/${name}_previous_$stamp"
done
CUDA_VISIBLE_DEVICES=$1 "$PY" train_shift_program.py --config ../scripts/mayo/configs/c2_pythia_program_smoke.json > "$ROOT/logs/c2_smoke.log" 2>&1
echo "program exit $?"
cat "$RUNS/CSM_C2_PYTHIA70M_program_t2_ts42_v1_20261011_SMOKE/NATURAL.json"
grep -q '"PASS"' "$RUNS/CSM_C2_PYTHIA70M_program_t2_ts42_v1_20261011_SMOKE/status.json" || { tail -c 1500 "$RUNS/CSM_C2_PYTHIA70M_program_t2_ts42_v1_20261011_SMOKE/status.json"; exit 1; }
CUDA_VISIBLE_DEVICES=$1 "$PY" run_shift_transfer.py --config ../scripts/mayo/configs/c2_pythia_eval_smoke.json > "$ROOT/logs/c2_eval_smoke.log" 2>&1
echo "evaluation exit $?"
tail -c 1500 "$RUNS/CSM_C2_PYTHIA70M_eval_t2_v1_20261011_SMOKE/status.json"
