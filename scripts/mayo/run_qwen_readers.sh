#!/usr/bin/env bash
# Held-out grammar readers for the six Qwen runs on one GPU (argument: gpu), started once the C8 lane, which writes the
# reader-complete maps, has finished. Finished runs are skipped.
set -uo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
RUNS="$ROOT/runs/cross_seed_models"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf" PYTHONPATH="$CODE/src:$CODE/scripts"
cd "$CODE"
passed() { grep -q '"PASS"' "$1/status.json" 2>/dev/null; }
while tmux has-session -t '=ccad-c8' 2>/dev/null; do sleep 60; done
for direction in s1_t2 s2_t1; do
  for seed in 0 1 2; do
    name=CSM_READERS_QWEN17_${direction}_ts${seed}_v1_20261011
    passed "$RUNS/$name" || CUDA_VISIBLE_DEVICES=$1 "$PY" scripts/grammar_readers.py \
      --config "scripts/mayo/configs/readers_qwen17_${direction}_ts${seed}.json" > "$ROOT/logs/$name.log" 2>&1
  done
done
echo "Qwen readers finished"
