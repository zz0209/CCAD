#!/usr/bin/env bash
# X2 on two GPT-2 Medium seed models. Each lane trains its model's SAEs, selects its explanation, then runs the
# within-model and cross-model transfers; a transfer into the other model waits until that model's SAEs exist.
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
e10_on_gpu() {
  local uuid
  uuid=$(nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader)
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$uuid" '$1==u {print $2}'); do
    ps -o args= -p "$p" | grep -q e10_gemma_grid && return 0
  done
  return 1
}
run_transfer() {
  local gpu=$1 name=$2
  passed "$RUNS/$name" && return 0
  CUDA_VISIBLE_DEVICES=$gpu "$PY" cross_seed_transfer.py --config "../$CFG/$3.json" > "$ROOT/logs/$name.log" 2>&1
}
lane() {
  local gpu=$1 model=$2 other=$3
  if [ "$gpu" = 0 ]; then until ! e10_on_gpu 0; do sleep 60; done; fi
  passed "$CODE/runs/CSM_SAE_gpt2_${model}_v1_20261010" || \
    CUDA_VISIBLE_DEVICES=$gpu "$PY" run_dictionary_training_curve.py --config "../$CFG/sae_$model.json" > "$ROOT/logs/sae_$model.log" 2>&1
  passed "$CODE/runs/CSM_SAE_gpt2_${model}_v1_20261010" || return 1
  passed "$RUNS/CSM_SELECT_gpt2_${model}_s1_v1_20261010" || \
    CUDA_VISIBLE_DEVICES=$gpu "$PY" select_grammar_explanation.py --config "../$CFG/select_$model.json" > "$ROOT/logs/select_$model.log" 2>&1
  passed "$RUNS/CSM_SELECT_gpt2_${model}_s1_v1_20261010" || return 1
  run_transfer "$gpu" "CSM_X2_gpt2_${model}s1_${model}s2_v1_20261010" "x2_${model}s1_${model}s2"
  wait_pass "$CODE/runs/CSM_SAE_gpt2_${other}_v1_20261010" || return 1
  run_transfer "$gpu" "CSM_X2_gpt2_${model}s1_${other}s1_v1_20261010" "x2_${model}s1_${other}s1"
  run_transfer "$gpu" "CSM_X2_gpt2_${model}s1_${other}s2_v1_20261010" "x2_${model}s1_${other}s2"
}
x1c() {
  local gpu=$1; shift
  for pair in "$@"; do
    run_transfer "$gpu" "CSM_X1C_gpt2_${pair}_v2_20261010" "x1c_$pair"
  done
}
(lane 1 x21 x49; x1c 1 s2_t3 s3_t4) &
(lane 0 x49 x21; x1c 0 s4_t5 s5_t1) &
wait
echo "X2 finished"
