#!/usr/bin/env bash
# Decoy explanation control for the eight GPT-2 Medium directions on one GPU (argument: gpu), started once the C7 lane on
# the same GPU has finished: a smoke run, then the eight runs. Finished runs are skipped.
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
while tmux has-session -t ccad-c7 2>/dev/null; do sleep 60; done
run() {
  passed "$RUNS/$1" && return 0
  CUDA_VISIBLE_DEVICES=$gpu "$PY" cross_seed_transfer.py --config "../$CFG/$2.json" "${@:3}" > "$ROOT/logs/$1.log" 2>&1
}
run CSM_C5B_GPT2_decoy_s1_t2_v1_20261011_SMOKE c5b_gpt2_decoy_s1_t2 --smoke
passed "$RUNS/CSM_C5B_GPT2_decoy_s1_t2_v1_20261011_SMOKE" || exit 1
for direction in s1_t2 s1_t3 s1_t4 s1_t5 s2_t3 s3_t4 s4_t5 s5_t1; do
  run "CSM_C5B_GPT2_decoy_${direction}_v1_20261011" "c5b_gpt2_decoy_${direction}"
done
echo "C5b finished"
