#!/usr/bin/env bash
# P0 pre-check of porting annotated circuit roles from Gemma-2-2B to Gemma-3-1B on one Mayo GPU (arguments: gpu, stage
# among corr-smoke, corr, test-smoke, test).
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
export HF_HOME=$M/cache/hf TMPDIR=$M/tmp
gpu=$1
R=$M/runs/circuits
cd "$M/code/CCAD/scripts/circuits"
ROLES=("gemma-2-2b/gemma-fact-dallas-austin=Texas"
       "gemma-2-2b/gemma-fact-oakland-sacramento=California (2)|Bay Area (say California)"
       "gemma-2-2b/gemma-fact-shanghai-beijing=China"
       "gemma-2-2b/gemma-fact-vancouver-victoria=Canada"
       "gemma-2-2b/gemma-michael-jordan=basketball"
       "gemma-2-2b/gemma-tom-brady=football (2)"
       "gemma-2-2b/gemma-euro=EU/European"
       "gemma-2-2b/gemma-dollar=US")
PY="$M/env_circuit/bin/python"
case $2 in
  corr-smoke) CUDA_VISIBLE_DEVICES=$gpu $PY port_corr.py --registry $R/PUBLISHED_EXPLANATIONS.json --roles "${ROLES[@]}" \
                --text $M/data/cross_seed/natural_fineweb_slice.json --docs 20 --out $R/CIRC_PORT_CORR_SMOKE_v1 ;;
  corr) CUDA_VISIBLE_DEVICES=$gpu $PY port_corr.py --registry $R/PUBLISHED_EXPLANATIONS.json --roles "${ROLES[@]}" \
          --text $M/data/cross_seed/natural_fineweb_slice.json --out $R/CIRC_PORT_CORR_G2_G3_v1 > $R/CIRC_PORT_CORR_G2_G3_v1.log 2>&1 ;;
  test-smoke) CUDA_VISIBLE_DEVICES=$gpu $PY port_test.py --corr $R/CIRC_PORT_CORR_G2_G3_v1/PORT_CORR.json \
                --families $R/PROMPT_FAMILIES.json --n-fills 1 --out $R/CIRC_PORT_P0_SMOKE_v1 ;;
  test) CUDA_VISIBLE_DEVICES=$gpu $PY port_test.py --corr $R/CIRC_PORT_CORR_G2_G3_v1/PORT_CORR.json \
          --families $R/PROMPT_FAMILIES.json --out $R/CIRC_PORT_P0_G2_G3_v1 > $R/CIRC_PORT_P0_G2_G3_v1.log 2>&1 ;;
esac
echo done $2
