#!/usr/bin/env bash
# CIRCUITS_ACROSS_DICTIONARIES P1e on one Mayo GPU (arguments: gpu, run id, target dictionary, then any further
# oracle_sites.py options such as --model, --registry-model and --source).
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
export HF_HOME=$M/cache/hf TMPDIR=$M/tmp
gpu=$1; run=$2; target=$3; shift 3
cd "$M/code/CCAD/scripts/circuits"
CUDA_VISIBLE_DEVICES=$gpu "$M/env_circuit/bin/python" oracle_sites.py \
  --registry "$M/runs/circuits/PUBLISHED_EXPLANATIONS.json" --target "$target" --out "$M/runs/circuits/$run" "$@"
