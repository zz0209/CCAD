#!/usr/bin/env bash
# Two GPU lanes for the public-target infinitive study; each lane runs its targets in order.
set -euo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
BULK="$ROOT/runs/naacl_sprint_20261010"
export PYTHONPATH="$CODE/src:$CODE/scripts" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
mkdir -p "$BULK"
if [ ! -f "$BULK/public_targets/TARGETS.json" ]; then
  "$PY" scripts/naacl_prepare_public_targets.py --config scripts/mayo/configs/e4a_prepare_public_targets.json
fi
lane() {
  local gpu=$1; shift
  for seed in "$@"; do
    local run="$BULK/NAACL_E4A_public_t${seed}_v1_20261010"
    if [ -f "$run/status.json" ] && grep -q '"PASS"' "$run/status.json"; then continue; fi
    if [ -d "$run" ]; then echo "existing run $run is not PASS; inspect before rerunning"; return 1; fi
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/train_infinitive_program.py --config "scripts/mayo/configs/e4a_public_t${seed}.json"
  done
}
lane 0 101 103 105 107 109 111 113 115 > "$ROOT/logs/e4a_lane0.log" 2>&1 &
lane 1 102 104 106 108 110 112 114 116 > "$ROOT/logs/e4a_lane1.log" 2>&1 &
wait
echo "E4a lanes finished"
