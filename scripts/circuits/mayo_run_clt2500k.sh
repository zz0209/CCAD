#!/usr/bin/env bash
# Unit test with the second Gemma-2 CLT (2.5M features) on one Mayo GPU (arguments: gpu, prompts: published or
# natural). The PLT's own graph units are those already drawn for the 426k runs on the same prompts.
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
export HF_HOME=$M/cache/hf TMPDIR=$M/tmp
gpu=$1
R=$M/runs/circuits
cd "$M/code/CCAD/scripts/circuits"
PY="$M/env_circuit/bin/python"
CLT=mntss/clt-gemma-2-2b-2.5M
case $2 in
  published) P=$R/CIRC_P1E_ORACLE_GEMMA_PLT_TO_CLT426K_v9/ORACLE_ROWS.json; UNITS=$R/GRAPH_UNITS_GEMMA2_PLT16K.json; TAG=PUB ;;
  natural) P=$R/NATURAL_PROMPTS_gemma-2-2b.json; UNITS=$R/GRAPH_UNITS_GEMMA_CLT426K_PLT_NATURAL.json; TAG=NAT ;;
esac
REV=CIRC_${TAG}_REVERSE_GEMMA_CLT2500K_v1
CUDA_VISIBLE_DEVICES=$gpu $PY oracle_reverse.py --model google/gemma-2-2b --source $CLT --target gemma --prompts-from $P \
  --out $R/$REV > $R/$REV.log 2>&1
CUDA_VISIBLE_DEVICES=$gpu $PY depth_referee.py --model google/gemma-2-2b --clt $CLT --plt gemma --registry-model gemma-2-2b \
  --clt-units-from $R/$REV/REVERSE_ROWS.json --plt-units $UNITS --prompts-from $P \
  --out $R/CIRC_${TAG}_DR_GEMMA_CLT2500K_v1 > $R/CIRC_${TAG}_DR_GEMMA_CLT2500K_v1.log 2>&1
echo all done
