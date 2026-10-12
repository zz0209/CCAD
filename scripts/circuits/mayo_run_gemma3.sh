#!/usr/bin/env bash
# Gemma-3-1B replication on one Mayo GPU (argument: gpu). Waits for the Gemma Scope 2 downloads, then moves the
# PLT's own graph units into the second PLT sparsity and into the CLT, and the CLT's own graph units into the PLT.
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
export HF_HOME=$M/cache/hf TMPDIR=$M/tmp
gpu=$1
R=$M/runs/circuits
PLT=mwhanna/gemma-scope-2-1b-pt/transcoder_all/width_16k_l0_small_affine
PLT_BIG=mwhanna/gemma-scope-2-1b-pt/transcoder_all/width_16k_l0_big_affine
CLT=mwhanna/gemma-scope-2-1b-pt/clt/width_262k_l0_medium_affine
until grep -q '"done"' $R/dl_gemma3.log && grep -q '"done"' $R/dl_gemma3_big.log; do sleep 30; done
cd "$M/code/CCAD/scripts/circuits"
run_sites() {
  CUDA_VISIBLE_DEVICES=$gpu "$M/env_circuit/bin/python" oracle_sites.py --registry $R/GRAPH_UNITS_GEMMA3_1B_PLT16K.json \
    --registry-model gemma-3-1b --model google/gemma-3-1b-pt --source $PLT --target "$2" --out $R/$1 > $R/$1.log 2>&1
}
run_sites CIRC_G3_ORACLE_PLT_TO_PLTBIG_v1 $PLT_BIG
run_sites CIRC_G3_ORACLE_PLT_TO_CLT262K_v1 $CLT
CUDA_VISIBLE_DEVICES=$gpu "$M/env_circuit/bin/python" oracle_reverse.py --model google/gemma-3-1b-pt --source $CLT \
  --target $PLT --prompts-from $R/CIRC_G3_ORACLE_PLT_TO_CLT262K_v1/ORACLE_ROWS.json \
  --out $R/CIRC_G3_REVERSE_CLT262K_TO_PLT_v1 > $R/CIRC_G3_REVERSE_CLT262K_TO_PLT_v1.log 2>&1
echo all done
