#!/usr/bin/env bash
# Depth referee on one Mayo GPU (arguments: gpu, then one or more of gemma, llama, gemma3), each model after the other.
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
export HF_HOME=$M/cache/hf TMPDIR=$M/tmp
gpu=$1
shift
R=$M/runs/circuits
cd "$M/code/CCAD/scripts/circuits"
for m in "$@"; do
  case $m in
    gemma) args=(--model google/gemma-2-2b --clt mntss/clt-gemma-2-2b-426k --plt gemma --registry-model gemma-2-2b
                 --clt-units-from $R/CIRC_P1R_REVERSE_GEMMA_CLT426K_TO_PLT_v2/REVERSE_ROWS.json
                 --plt-units $R/GRAPH_UNITS_GEMMA2_PLT16K.json
                 --prompts-from $R/CIRC_P1E_ORACLE_GEMMA_PLT_TO_CLT426K_v9/ORACLE_ROWS.json)
           run=CIRC_DR_GEMMA_CLT426K_v2 ;;
    llama) args=(--model meta-llama/Llama-3.2-1B --clt mntss/clt-llama-3.2-1b-524k --plt llama --registry-model llama-3.2-1b
                 --clt-units-from $R/CIRC_P1R_REVERSE_LLAMA_CLT524K_TO_PLT_v1/REVERSE_ROWS.json
                 --plt-units $R/GRAPH_UNITS_LLAMA_PLT.json
                 --prompts-from $R/CIRC_P1E_ORACLE_LLAMA_PLT_TO_CLT524K_v4/ORACLE_ROWS.json)
           run=CIRC_DR_LLAMA_CLT524K_v2 ;;
    gemma3) args=(--model google/gemma-3-1b-pt --clt mwhanna/gemma-scope-2-1b-pt/clt/width_262k_l0_medium_affine
                  --plt mwhanna/gemma-scope-2-1b-pt/transcoder_all/width_16k_l0_small_affine --registry-model gemma-3-1b
                  --clt-units-from $R/CIRC_G3_REVERSE_CLT262K_TO_PLT_v1/REVERSE_ROWS.json
                  --plt-units $R/GRAPH_UNITS_GEMMA3_1B_PLT16K.json
                  --prompts-from $R/CIRC_G3_ORACLE_PLT_TO_PLTBIG_v1/ORACLE_ROWS.json)
            run=CIRC_DR_G3_CLT262K_v2 ;;
  esac
  CUDA_VISIBLE_DEVICES=$gpu "$M/env_circuit/bin/python" depth_referee.py "${args[@]}" --out $R/$run > $R/$run.log 2>&1
done
echo all done
