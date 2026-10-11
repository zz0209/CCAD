#!/usr/bin/env bash
# The GPT-2 one-to-one controls with gains up to four (argument: gpu), started once the C5 lane on the same GPU has
# finished. run_c4_c7.sh skips any of these runs that have already passed.
set -uo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
RUNS="$ROOT/runs/cross_seed_models"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf" PYTHONPATH="$CODE/src"
cd "$CODE/scripts"
passed() { grep -q '"PASS"' "$1/status.json" 2>/dev/null; }
while tmux has-session -t ccad-c5 2>/dev/null; do sleep 60; done
for direction in s1_t2 s1_t3 s1_t4 s1_t5 s2_t3 s3_t4 s4_t5 s5_t1; do
  name=CSM_C7_GPT2_matched4_${direction}_v1_20261011
  passed "$RUNS/$name" || [ -e "$RUNS/$name" ] || CUDA_VISIBLE_DEVICES=$1 "$PY" cross_seed_transfer.py \
    --config "../scripts/mayo/configs/c7_gpt2_matched4_${direction}.json" > "$ROOT/logs/$name.log" 2>&1
done
echo "C7 finished"
