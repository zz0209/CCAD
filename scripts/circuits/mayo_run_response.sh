#!/usr/bin/env bash
# Response-aware test on one Mayo GPU (arguments: gpu, run id, target dictionary, then any further oracle_response.py
# options such as --registry, --model, --registry-model and --source).
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
export HF_HOME=$M/cache/hf TMPDIR=$M/tmp
gpu=$1; run=$2; target=$3; shift 3
cd "$M/code/CCAD/scripts/circuits"
CUDA_VISIBLE_DEVICES=$gpu "$M/env_circuit/bin/python" oracle_response.py --target "$target" --out "$M/runs/circuits/$run" "$@" \
  > "$M/runs/circuits/$run.log" 2>&1
