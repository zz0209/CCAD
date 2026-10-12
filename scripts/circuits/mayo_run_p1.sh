#!/usr/bin/env bash
# CIRCUITS_ACROSS_DICTIONARIES P1 on one Mayo GPU (arguments: gpu, run id, target dictionary, documents, steps, then
# any further fit_transfer.py options).
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
export HF_HOME=$M/cache/hf TMPDIR=$M/tmp
gpu=$1; run=$2; target=$3; docs=$4; steps=$5; shift 5
cd "$M/code/CCAD/scripts/circuits"
CUDA_VISIBLE_DEVICES=$gpu "$M/env_circuit/bin/python" fit_transfer.py \
  --registry "$M/runs/circuits/PUBLISHED_EXPLANATIONS.json" --target "$target" \
  --text "$M/data/cross_seed/natural_fineweb_slice.json" --docs "$docs" --steps "$steps" --out "$M/runs/circuits/$run" "$@"
