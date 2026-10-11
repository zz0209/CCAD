#!/usr/bin/env bash
# READER_COVERAGE EXP-2 grammar cell on one GPU (argument: gpu): a smoke run and the eight GPT-2 directions at once,
# then the six Qwen runs once the EXP-1 Gemma lane has freed its memory. Finished runs are skipped.
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
run() {
  passed "$RUNS/$1" && return 0
  CUDA_VISIBLE_DEVICES=$gpu "$PY" cross_seed_transfer.py --config "../$CFG/$2.json" "${@:3}" > "$ROOT/logs/$1.log" 2>&1
}
run CSM_C8_GPT2_rc_s1_t2_v1_20261011_SMOKE c8_gpt2_rc_s1_t2 --smoke
passed "$RUNS/CSM_C8_GPT2_rc_s1_t2_v1_20261011_SMOKE" || exit 1
for direction in s1_t2 s1_t3 s1_t4 s1_t5 s2_t3 s3_t4 s4_t5 s5_t1; do
  run "CSM_C8_GPT2_rc_${direction}_v1_20261011" "c8_gpt2_rc_${direction}"
done
while tmux has-session -t '=ccad-exp1' 2>/dev/null; do sleep 60; done
for direction in s1_t2 s2_t1; do
  for seed in 0 1 2; do
    run "CSM_C8_QWEN17_rc_${direction}_ts${seed}_v1_20261011" "c8_qwen17_rc_${direction}_ts${seed}"
  done
done
echo "C8 finished"
