#!/usr/bin/env bash
# CIRCUITS_ACROSS_DICTIONARIES P2 on one Mayo GPU (arguments: gpu, run id, then any further generalize.py options such
# as --source, --prompts-from and --units-from).
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
export HF_HOME=$M/cache/hf TMPDIR=$M/tmp
gpu=$1; run=$2; shift 2
cd "$M/code/CCAD/scripts/circuits"
CUDA_VISIBLE_DEVICES=$gpu "$M/env_circuit/bin/python" generalize.py \
  --registry "$M/runs/circuits/PUBLISHED_EXPLANATIONS.json" --families "$M/runs/circuits/PROMPT_FAMILIES.json" \
  --out "$M/runs/circuits/$run" "$@"
