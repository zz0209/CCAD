#!/usr/bin/env bash
# READER_COVERAGE held-out readers for Pythia SHIFT on one GPU (argument: gpu): a smoke run on the R59 relations at once,
# then the four targets once the C2d lane, which writes the last relations, has finished. Finished runs are skipped.
set -uo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
RUNS="$ROOT/runs/cross_seed_models"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf" PYTHONPATH="$CODE/src"
cd "$CODE"
passed() { grep -q '"PASS"' "$1/status.json" 2>/dev/null; }
readers() {
  passed "$RUNS/$1" && return 0
  CUDA_VISIBLE_DEVICES=$2 "$PY" scripts/mayo/pythia_readers.py --config "scripts/mayo/configs/$3.json" > "$ROOT/logs/$1.log" 2>&1
}
readers CSM_EXP1_PYTHIA70M_readers_t2_v1_20261011_SMOKE "$1" exp1_pythia_readers_smoke
passed "$RUNS/CSM_EXP1_PYTHIA70M_readers_t2_v1_20261011_SMOKE" || exit 1
while tmux has-session -t ccad-c2d 2>/dev/null; do sleep 60; done
for target in 2 3 4 5; do
  readers "CSM_EXP1_PYTHIA70M_readers_t${target}_v1_20261011" "$1" "exp1_pythia_readers_t${target}"
done
echo "Pythia readers finished"
