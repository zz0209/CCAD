#!/usr/bin/env bash
# Unit test on natural-text prompts for one model on one Mayo GPU (arguments: gpu, model: gemma, llama or gemma3):
# the PLT's and the CLT's own graphs, the depth referee and the PLT units' reconstruction in the CLT.
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
export HF_HOME=$M/cache/hf TMPDIR=$M/tmp
gpu=$1
R=$M/runs/circuits
cd "$M/code/CCAD/scripts/circuits"
case $2 in
  gemma) MODEL=google/gemma-2-2b; KEY=gemma-2-2b; PLT=gemma; CLT=mntss/clt-gemma-2-2b-426k; TAG=GEMMA_CLT426K ;;
  llama) MODEL=meta-llama/Llama-3.2-1B; KEY=llama-3.2-1b; PLT=llama; CLT=mntss/clt-llama-3.2-1b-524k; TAG=LLAMA_CLT524K ;;
  gemma3) MODEL=google/gemma-3-1b-pt; KEY=gemma-3-1b; PLT=mwhanna/gemma-scope-2-1b-pt/transcoder_all/width_16k_l0_small_affine
          CLT=mwhanna/gemma-scope-2-1b-pt/clt/width_262k_l0_medium_affine; TAG=G3_CLT262K ;;
esac
PY="$M/env_circuit/bin/python"
P=$R/NATURAL_PROMPTS_$KEY.json
UNITS=$R/GRAPH_UNITS_${TAG}_PLT_NATURAL.json
REV=CIRC_NAT_REVERSE_${TAG}_v1
# The PLT's own graphs and the CLT's own graphs with their reconstruction in the PLT run side by side.
CUDA_VISIBLE_DEVICES=$gpu $PY graph_registry.py --prompts-from $P --model $MODEL --source $PLT --model-key $KEY \
  --out $UNITS > $R/GRAPH_UNITS_${TAG}_PLT_NATURAL.log 2>&1 &
CUDA_VISIBLE_DEVICES=$gpu $PY oracle_reverse.py --model $MODEL --source $CLT --target $PLT --prompts-from $P \
  --out $R/$REV > $R/$REV.log 2>&1
wait
CUDA_VISIBLE_DEVICES=$gpu $PY depth_referee.py --model $MODEL --clt $CLT --plt $PLT --registry-model $KEY \
  --clt-units-from $R/$REV/REVERSE_ROWS.json --plt-units $UNITS --prompts-from $P \
  --out $R/CIRC_NAT_DR_${TAG}_v1 > $R/CIRC_NAT_DR_${TAG}_v1.log 2>&1
CUDA_VISIBLE_DEVICES=$gpu $PY oracle_sites.py --registry $UNITS --registry-model $KEY --model $MODEL --source $PLT \
  --target $CLT --out $R/CIRC_NAT_OWN_ORACLE_${TAG}_v1 > $R/CIRC_NAT_OWN_ORACLE_${TAG}_v1.log 2>&1
echo all done
