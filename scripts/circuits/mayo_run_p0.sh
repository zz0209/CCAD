#!/usr/bin/env bash
# CIRCUITS_ACROSS_DICTIONARIES P0 on one Mayo GPU (arguments: gpu, run id, number of generic-text documents), after
# the CLT-426k download has finished.
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
export HF_HOME=$M/cache/hf TMPDIR=$M/tmp
until grep -qs 'clt-gemma-2-2b-426k' "$M/assets/CIRCUIT_DICTIONARIES_MANIFEST.json"; do sleep 30; done
CUDA_VISIBLE_DEVICES=$1 "$M/env_circuit/bin/python" "$M/code/CCAD/scripts/circuits/pilot_one_to_one.py" \
  --registry "$M/runs/circuits/PUBLISHED_EXPLANATIONS.json" --target mntss/clt-gemma-2-2b-426k \
  --text "$M/data/cross_seed/natural_fineweb_slice.json" --docs "$3" --out "$M/runs/circuits/$2"
