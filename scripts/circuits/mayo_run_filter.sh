#!/usr/bin/env bash
# Filtered-graph prediction of held-out interventions on one Mayo GPU (arguments: gpu, then jobs among smoke-llama,
# llama-clt, llama-plt, gemma-clt, gemma-plt, g3-clt, g3-plt), each job after the other. The project uses only GPU1 on
# Mayo since 2026-10-12T20:52Z. The CLT jobs of Llama and Gemma-2 continue the runs stopped at 20:51Z (when GPU0 was
# given back to other users) from their saved prompts.
set -uo pipefail
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
    smoke-llama) a=(--model meta-llama/Llama-3.2-1B --dictionary mntss/clt-llama-3.2-1b-524k --role clt --max-prompts 2
                    --prompts $R/CIRC_P1E_ORACLE_LLAMA_PLT_TO_CLT524K_v4/ORACLE_ROWS.json); run=CIRC_FILTER_SMOKE_LLAMA_v2 ;;
    llama-clt) a=(--model meta-llama/Llama-3.2-1B --dictionary mntss/clt-llama-3.2-1b-524k --role clt
                  --prompts $R/HELDOUT_PROMPTS_llama-3.2-1b.json
                  --resume-from $R/abandoned_CIRC_FILTER_LLAMA_CLT524K_v1_yielded_gpu_20261012T2051Z/FILTER_PREDICTION_ROWS.partial.json)
               run=CIRC_FILTER_LLAMA_CLT524K_v2 ;;
    llama-plt) a=(--model meta-llama/Llama-3.2-1B --dictionary llama --role plt
                  --prompts $R/HELDOUT_PROMPTS_llama-3.2-1b.json); run=CIRC_FILTER_LLAMA_PLT_v2 ;;
    gemma-clt) a=(--model google/gemma-2-2b --dictionary mntss/clt-gemma-2-2b-426k --role clt
                  --prompts $R/HELDOUT_PROMPTS_gemma-2-2b.json
                  --resume-from $R/abandoned_CIRC_FILTER_GEMMA_CLT426K_v1_yielded_gpu_20261012T2051Z/FILTER_PREDICTION_ROWS.partial.json)
               run=CIRC_FILTER_GEMMA_CLT426K_v2 ;;
    gemma-plt) a=(--model google/gemma-2-2b --dictionary gemma --role plt
                  --prompts $R/HELDOUT_PROMPTS_gemma-2-2b.json); run=CIRC_FILTER_GEMMA_PLT_v2 ;;
    g3-clt) a=(--model google/gemma-3-1b-pt --dictionary $G3CLT --role clt --edit-residual
               --prompts $R/HELDOUT_PROMPTS_gemma-3-1b.json); run=CIRC_FILTER_G3_CLT262K_v2 ;;
    g3-plt) a=(--model google/gemma-3-1b-pt --dictionary $G3PLT --role plt
               --prompts $R/HELDOUT_PROMPTS_gemma-3-1b.json); run=CIRC_FILTER_G3_PLT_v2 ;;
  esac
  CUDA_VISIBLE_DEVICES=$gpu "$M/env_circuit/bin/python" filter_prediction.py "${a[@]}" --out $R/$run > $R/$run.log 2>&1
done
echo all done
