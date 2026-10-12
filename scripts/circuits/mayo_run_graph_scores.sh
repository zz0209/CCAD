#!/usr/bin/env bash
# E1 pre-check on one Mayo GPU (arguments: gpu, then jobs among gemma-clt gemma-plt llama-clt llama-plt g3-clt g3-plt),
# each job after the other: replacement and completeness scores of the dictionary's own graphs on 20 published-graph
# prompts and 20 natural-text prompts.
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
    gemma-clt) a=(--model google/gemma-2-2b --dictionary mntss/clt-gemma-2-2b-426k
                  --prompts-from $R/CIRC_P1E_ORACLE_GEMMA_PLT_TO_CLT426K_v9/ORACLE_ROWS.json $R/NATURAL_PROMPTS_gemma-2-2b.json); run=CIRC_E1_SCORES_GEMMA_CLT426K_v1 ;;
    gemma-plt) a=(--model google/gemma-2-2b --dictionary gemma
                  --prompts-from $R/CIRC_P1E_ORACLE_GEMMA_PLT_TO_CLT426K_v9/ORACLE_ROWS.json $R/NATURAL_PROMPTS_gemma-2-2b.json); run=CIRC_E1_SCORES_GEMMA_PLT_v1 ;;
    llama-clt) a=(--model meta-llama/Llama-3.2-1B --dictionary mntss/clt-llama-3.2-1b-524k
                  --prompts-from $R/CIRC_P1E_ORACLE_LLAMA_PLT_TO_CLT524K_v4/ORACLE_ROWS.json $R/NATURAL_PROMPTS_llama-3.2-1b.json); run=CIRC_E1_SCORES_LLAMA_CLT524K_v1 ;;
    llama-plt) a=(--model meta-llama/Llama-3.2-1B --dictionary llama
                  --prompts-from $R/CIRC_P1E_ORACLE_LLAMA_PLT_TO_CLT524K_v4/ORACLE_ROWS.json $R/NATURAL_PROMPTS_llama-3.2-1b.json); run=CIRC_E1_SCORES_LLAMA_PLT_v1 ;;
    g3-clt) a=(--model google/gemma-3-1b-pt --dictionary $G3CLT
               --prompts-from $R/CIRC_G3_ORACLE_PLT_TO_PLTBIG_v1/ORACLE_ROWS.json $R/NATURAL_PROMPTS_gemma-3-1b.json); run=CIRC_E1_SCORES_G3_CLT262K_v2 ;;
    g3-plt) a=(--model google/gemma-3-1b-pt --dictionary $G3PLT
               --prompts-from $R/CIRC_G3_ORACLE_PLT_TO_PLTBIG_v1/ORACLE_ROWS.json $R/NATURAL_PROMPTS_gemma-3-1b.json); run=CIRC_E1_SCORES_G3_PLT_v1 ;;
    # The same release at its other sparsity: the CLT with more and the PLT with more active features per token.
    g3-clt-big) a=(--model google/gemma-3-1b-pt --dictionary ${G3CLT/l0_medium/l0_big}
                   --prompts-from $R/CIRC_G3_ORACLE_PLT_TO_PLTBIG_v1/ORACLE_ROWS.json $R/NATURAL_PROMPTS_gemma-3-1b.json); run=CIRC_E1_SCORES_G3_CLT262K_BIG_v2 ;;
    g3-plt-big) a=(--model google/gemma-3-1b-pt --dictionary ${G3PLT/l0_small/l0_big}
                   --prompts-from $R/CIRC_G3_ORACLE_PLT_TO_PLTBIG_v1/ORACLE_ROWS.json $R/NATURAL_PROMPTS_gemma-3-1b.json); run=CIRC_E1_SCORES_G3_PLT_BIG_v1 ;;
  esac
  CUDA_VISIBLE_DEVICES=$gpu "$M/env_circuit/bin/python" graph_scores.py "${a[@]}" --out $R/$run > $R/$run.log 2>&1
done
echo all done
