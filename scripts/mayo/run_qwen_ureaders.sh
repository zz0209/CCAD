#!/usr/bin/env bash
# EXP-U Qwen readers on one Mayo GPU (arguments: gpu, then config names under scripts/mayo/configs without .json), run
# in order: s2_t1 development readers and confirmation readers. Finished runs are skipped; a failed run stops the lane.
set -uo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
RUNS="$ROOT/runs/cross_seed_models"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf" PYTHONPATH="$CODE/src:$CODE/scripts"
cd "$CODE"
gpu=$1
shift
for name in "$@"; do
  cfg="scripts/mayo/configs/$name.json"
  run=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['run_id'])" "$cfg")
  grep -qs '"PASS"' "$RUNS/$run/status.json" && continue
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/grammar_readers.py --config "$cfg" > "$ROOT/logs/$run.log" 2>&1
  grep -qs '"PASS"' "$RUNS/$run/status.json" || { echo "$run failed"; exit 1; }
done
echo "Qwen readers finished on GPU $gpu"
