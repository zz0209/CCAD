#!/usr/bin/env bash
# Coverage test on circuits on one Mayo GPU (arguments: gpu, run id, target dictionary, then any further
# fit_covered.py options). Settings otherwise follow the P1 runs: 1,024 windows of 64 tokens, 400 steps, batch 4.
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
export HF_HOME=$M/cache/hf TMPDIR=$M/tmp
gpu=$1; run=$2; target=$3; shift 3
cd "$M/code/CCAD/scripts/circuits"
CUDA_VISIBLE_DEVICES=$gpu "$M/env_circuit/bin/python" fit_covered.py \
  --registry "$M/runs/circuits/PUBLISHED_EXPLANATIONS.json" --families "$M/runs/circuits/PROMPT_FAMILIES.json" \
  --target "$target" --text "$M/data/cross_seed/natural_fineweb_slice.json" --out "$M/runs/circuits/$run" "$@" \
  > "$M/runs/circuits/$run.log" 2>&1
