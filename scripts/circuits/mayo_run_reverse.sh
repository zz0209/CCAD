#!/usr/bin/env bash
# CIRCUITS_ACROSS_DICTIONARIES P1r on one Mayo GPU (arguments: gpu, run id, CLT source reference, P1e run whose
# prompts are reused).
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
export HF_HOME=$M/cache/hf TMPDIR=$M/tmp
cd "$M/code/CCAD/scripts/circuits"
CUDA_VISIBLE_DEVICES=$1 "$M/env_circuit/bin/python" oracle_reverse.py \
  --prompts-from "$M/runs/circuits/$4/ORACLE_ROWS.json" --source "$3" --out "$M/runs/circuits/$2"
