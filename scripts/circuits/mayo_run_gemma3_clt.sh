#!/usr/bin/env bash
# Gemma-3-1B CLT steps on one Mayo GPU (argument: gpu), one CLT loading at a time since each stages its float32
# checkpoint in CPU memory: the PLT's own graph units into the CLT, then the CLT's own graph units into the PLT.
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
export HF_HOME=$M/cache/hf TMPDIR=$M/tmp
gpu=$1
R=$M/runs/circuits
PLT=mwhanna/gemma-scope-2-1b-pt/transcoder_all/width_16k_l0_small_affine
CLT=mwhanna/gemma-scope-2-1b-pt/clt/width_262k_l0_medium_affine
cd "$M/code/CCAD/scripts/circuits"
CUDA_VISIBLE_DEVICES=$gpu "$M/env_circuit/bin/python" oracle_sites.py --registry $R/GRAPH_UNITS_GEMMA3_1B_PLT16K.json \
  --registry-model gemma-3-1b --model google/gemma-3-1b-pt --source $PLT --target $CLT \
  --out $R/CIRC_G3_ORACLE_PLT_TO_CLT262K_v2 > $R/CIRC_G3_ORACLE_PLT_TO_CLT262K_v2.log 2>&1
CUDA_VISIBLE_DEVICES=$gpu "$M/env_circuit/bin/python" oracle_reverse.py --model google/gemma-3-1b-pt --source $CLT \
  --target $PLT --prompts-from $R/CIRC_G3_ORACLE_PLT_TO_PLTBIG_v1/ORACLE_ROWS.json \
  --out $R/CIRC_G3_REVERSE_CLT262K_TO_PLT_v1 > $R/CIRC_G3_REVERSE_CLT262K_TO_PLT_v1.log 2>&1
echo all done
