#!/usr/bin/env bash
# Response-aware test for the remaining CLT targets on one Mayo GPU (argument: gpu), one after another so that only
# one Gemma Scope 2 CLT is staged in CPU memory at a time.
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
R=$M/runs/circuits
gpu=$1
cd "$M/code/CCAD/scripts/circuits"
bash mayo_run_response.sh $gpu CIRC_RESP_GEMMA_PLT_TO_CLT2500K_v1 mntss/clt-gemma-2-2b-2.5M --registry $R/PUBLISHED_EXPLANATIONS.json
bash mayo_run_response.sh $gpu CIRC_RESP_LLAMA_PLT_TO_CLT524K_v1 mntss/clt-llama-3.2-1b-524k --registry $R/PUBLISHED_EXPLANATIONS.json \
  --model meta-llama/Llama-3.2-1B --registry-model llama-3.2-1b --source llama
bash mayo_run_response.sh $gpu CIRC_RESP_G3_PLT_TO_CLT262K_v1 mwhanna/gemma-scope-2-1b-pt/clt/width_262k_l0_medium_affine \
  --registry $R/GRAPH_UNITS_GEMMA3_1B_PLT16K.json --model google/gemma-3-1b-pt --registry-model gemma-3-1b \
  --source mwhanna/gemma-scope-2-1b-pt/transcoder_all/width_16k_l0_small_affine
echo all done
