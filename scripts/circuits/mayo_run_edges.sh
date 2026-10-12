#!/usr/bin/env bash
# Edge test on one Mayo GPU (arguments: gpu, then one or more of gemma-clt, gemma-plt, llama-clt, llama-plt, g3-clt, g3-plt).
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
export HF_HOME=$M/cache/hf TMPDIR=$M/tmp
gpu=$1
shift
R=$M/runs/circuits
cd "$M/code/CCAD/scripts/circuits"
G3PLT=mwhanna/gemma-scope-2-1b-pt/transcoder_all/width_16k_l0_small_affine
G3CLT=mwhanna/gemma-scope-2-1b-pt/clt/width_262k_l0_medium_affine
for job in "$@"; do
  case $job in
    gemma-clt) a=(--dictionary mntss/clt-gemma-2-2b-426k --units-from $R/CIRC_P1R_REVERSE_GEMMA_CLT426K_TO_PLT_v2/REVERSE_ROWS.json
                  --prompts-from $R/CIRC_P1E_ORACLE_GEMMA_PLT_TO_CLT426K_v9/ORACLE_ROWS.json); run=CIRC_EDGE_GEMMA_CLT426K_v1 ;;
    gemma-plt) a=(--dictionary gemma --registry $R/GRAPH_UNITS_GEMMA2_PLT16K.json
                  --prompts-from $R/CIRC_P1E_ORACLE_GEMMA_PLT_TO_CLT426K_v9/ORACLE_ROWS.json); run=CIRC_EDGE_GEMMA_PLT_v1 ;;
    llama-clt) a=(--model meta-llama/Llama-3.2-1B --dictionary mntss/clt-llama-3.2-1b-524k --registry-model llama-3.2-1b
                  --units-from $R/CIRC_P1R_REVERSE_LLAMA_CLT524K_TO_PLT_v1/REVERSE_ROWS.json
                  --prompts-from $R/CIRC_P1E_ORACLE_LLAMA_PLT_TO_CLT524K_v4/ORACLE_ROWS.json); run=CIRC_EDGE_LLAMA_CLT524K_v1 ;;
    llama-plt) a=(--model meta-llama/Llama-3.2-1B --dictionary llama --registry-model llama-3.2-1b --registry $R/GRAPH_UNITS_LLAMA_PLT.json
                  --prompts-from $R/CIRC_P1E_ORACLE_LLAMA_PLT_TO_CLT524K_v4/ORACLE_ROWS.json); run=CIRC_EDGE_LLAMA_PLT_v1 ;;
    g3-clt) a=(--model google/gemma-3-1b-pt --dictionary $G3CLT --registry-model gemma-3-1b
               --units-from $R/CIRC_G3_REVERSE_CLT262K_TO_PLT_v1/REVERSE_ROWS.json
               --prompts-from $R/CIRC_G3_ORACLE_PLT_TO_PLTBIG_v1/ORACLE_ROWS.json); run=CIRC_EDGE_G3_CLT262K_v1 ;;
    gemma-clt-nat) a=(--dictionary mntss/clt-gemma-2-2b-426k --units-from $R/CIRC_NAT_REVERSE_GEMMA_CLT426K_v1/REVERSE_ROWS.json
                      --prompts-from $R/NATURAL_PROMPTS_gemma-2-2b.json); run=CIRC_NAT_EDGE_GEMMA_CLT426K_v1 ;;
    llama-clt-nat) a=(--model meta-llama/Llama-3.2-1B --dictionary mntss/clt-llama-3.2-1b-524k --registry-model llama-3.2-1b
                      --units-from $R/CIRC_NAT_REVERSE_LLAMA_CLT524K_v1/REVERSE_ROWS.json
                      --prompts-from $R/NATURAL_PROMPTS_llama-3.2-1b.json); run=CIRC_NAT_EDGE_LLAMA_CLT524K_v1 ;;
    g3-clt-res) a=(--model google/gemma-3-1b-pt --dictionary $G3CLT --registry-model gemma-3-1b
                   --units-from $R/CIRC_G3_REVERSE_CLT262K_TO_PLT_v1/REVERSE_ROWS.json
                   --prompts-from $R/CIRC_G3_ORACLE_PLT_TO_PLTBIG_v1/ORACLE_ROWS.json --edit-residual); run=CIRC_EDGE_G3_CLT262K_RES_v3 ;;
    g3-plt) a=(--model google/gemma-3-1b-pt --dictionary $G3PLT --registry-model gemma-3-1b --registry $R/GRAPH_UNITS_GEMMA3_1B_PLT16K.json
               --prompts-from $R/CIRC_G3_ORACLE_PLT_TO_PLTBIG_v1/ORACLE_ROWS.json); run=CIRC_EDGE_G3_PLT_v1 ;;
  esac
  CUDA_VISIBLE_DEVICES=$gpu "$M/env_circuit/bin/python" edge_referee.py "${a[@]}" --out $R/$run > $R/$run.log 2>&1
done
echo all done
