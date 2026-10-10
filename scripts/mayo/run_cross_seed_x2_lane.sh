#!/usr/bin/env bash
# One X2 lane (arguments: gpu model other), with the functions of run_cross_seed_x2.sh.
set -uo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
RUNS="$ROOT/runs/cross_seed_models"
CFG=configs/cross_seed/mayo
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf" PYTHONPATH="$CODE/src"
cd "$CODE/scripts"
passed() { grep -q '"PASS"' "$1/status.json" 2>/dev/null; }
wait_pass() { until passed "$1"; do grep -q '"FAIL"' "$1/status.json" 2>/dev/null && return 1; sleep 60; done; }
run_transfer() {
  local gpu=$1 name=$2
  passed "$RUNS/$name" && return 0
  CUDA_VISIBLE_DEVICES=$gpu "$PY" cross_seed_transfer.py --config "../$CFG/$3.json" > "$ROOT/logs/$name.log" 2>&1
}
gpu=$1 model=$2 other=$3
passed "$CODE/runs/CSM_SAE_gpt2_${model}_v1_20261010" || \
  CUDA_VISIBLE_DEVICES=$gpu "$PY" run_dictionary_training_curve.py --config "../$CFG/sae_$model.json" > "$ROOT/logs/sae_$model.log" 2>&1
passed "$CODE/runs/CSM_SAE_gpt2_${model}_v1_20261010" || exit 1
passed "$RUNS/CSM_SELECT_gpt2_${model}_s1_v1_20261010" || \
  CUDA_VISIBLE_DEVICES=$gpu "$PY" select_grammar_explanation.py --config "../$CFG/select_$model.json" > "$ROOT/logs/select_$model.log" 2>&1
passed "$RUNS/CSM_SELECT_gpt2_${model}_s1_v1_20261010" || exit 1
run_transfer "$gpu" "CSM_X2_gpt2_${model}s1_${model}s2_v1_20261010" "x2_${model}s1_${model}s2"
wait_pass "$CODE/runs/CSM_SAE_gpt2_${other}_v1_20261010" || exit 1
run_transfer "$gpu" "CSM_X2_gpt2_${model}s1_${other}s1_v1_20261010" "x2_${model}s1_${other}s1"
run_transfer "$gpu" "CSM_X2_gpt2_${model}s1_${other}s2_v1_20261010" "x2_${model}s1_${other}s2"
echo "lane $model finished"
