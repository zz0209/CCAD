#!/usr/bin/env bash
# Coverage x write support on the CLT with the expanded candidate pool (8 same-layer candidates by correlation plus 8
# features read at any layer up to the source layer, by cosine of their write). Argument: gpu, then which half to run
# ("whole" or "slices"); each half runs the generic-text fit and the covering fit on the same candidates.
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
gpu=$1; part=$2
cd "$M/code/CCAD/scripts/circuits"
extra=""
if [ "$part" = "slices" ]; then extra="--slices"; fi
bash mayo_run_covered.sh $gpu CIRC_COVX_GEMMA_PLT_TO_CLT426K_generic_${part}_v1 mntss/clt-gemma-2-2b-426k --cross-candidates 8 --no-task $extra
bash mayo_run_covered.sh $gpu CIRC_COVX_GEMMA_PLT_TO_CLT426K_covering_${part}_v1 mntss/clt-gemma-2-2b-426k --cross-candidates 8 $extra
echo all done
