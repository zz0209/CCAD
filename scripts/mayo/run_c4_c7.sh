#!/usr/bin/env bash
# CONTAINMENT experiments 2b, 4 and 7b on one GPU (argument: gpu): the Pythia many-to-many arms with deletion bound
# four, the six Qwen-Scope grammar runs, then the eight GPT-2 one-to-one controls with gains up to four. Finished runs
# are skipped, so the lane resumes after an interruption.
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
# Experiment 2b first: the Pythia many-to-many arms with deletion bound four, then their evaluations.
for target in 2 3 4 5; do
  for seed in 42 43 44; do
    name=CSM_C2B_PYTHIA70M_program_t${target}_ts${seed}_v1_20261011
    passed "$RUNS/$name" || CUDA_VISIBLE_DEVICES=$gpu "$PY" train_shift_program.py \
      --config "../$CFG/c2b_pythia_program_t${target}_ts${seed}.json" > "$ROOT/logs/$name.log" 2>&1
  done
done
for target in 2 3 4 5; do
  name=CSM_C2B_PYTHIA70M_eval_t${target}_v1_20261011
  passed "$RUNS/$name" || CUDA_VISIBLE_DEVICES=$gpu "$PY" run_shift_transfer.py \
    --config "../$CFG/c2b_pythia_eval_t${target}.json" > "$ROOT/logs/$name.log" 2>&1
done
# Qwen3-1.7B with its full output head needs about 11 GB, which only fits once the Gemma run on GPU 1 has ended.
until grep -qE '"(PASS|FAIL)"' "$RUNS/CSM_C1_GEMMA2B_natural_l0low16k_v1_20261011/status.json" 2>/dev/null; do sleep 60; done
run() {
  passed "$RUNS/$1" && return 0
  CUDA_VISIBLE_DEVICES=$gpu "$PY" cross_seed_transfer.py --config "../$CFG/$2.json" "${@:3}" > "$ROOT/logs/$1.log" 2>&1
}
# A Qwen smoke run first; an earlier failed smoke directory keeps its name with a suffix.
smoke=CSM_C4_QWEN17_s1_t2_ts0_v1_20261011_SMOKE
if ! passed "$RUNS/$smoke"; then
  [ -e "$RUNS/$smoke" ] && mv "$RUNS/$smoke" "$RUNS/${smoke}_previous_$(date -u +%H%M%S)"
  run "$smoke" c4_qwen17_s1_t2_ts0 --smoke
  passed "$RUNS/$smoke" || exit 1
fi
for direction in s1_t2 s2_t1; do
  for seed in 0 1 2; do
    run "CSM_C4_QWEN17_${direction}_ts${seed}_v1_20261011" "c4_qwen17_${direction}_ts${seed}"
  done
done
for direction in s1_t2 s1_t3 s1_t4 s1_t5 s2_t3 s3_t4 s4_t5 s5_t1; do
  run "CSM_C7_GPT2_matched4_${direction}_v1_20261011" "c7_gpt2_matched4_${direction}"
done
echo "C2b, C4 and C7 finished"
