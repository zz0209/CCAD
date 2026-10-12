#!/usr/bin/env bash
# Coverage fits on one Mayo GPU (argument: gpu) after the response-aware PLT calibration on that GPU finishes, with the
# initializations of the matching P1 runs.
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
R=$M/runs/circuits
gpu=$1
until grep -q -E '"done"|Traceback' $R/CIRC_RESP_GEMMA_PLT_TO_PLTL0R2_v1.log; do sleep 30; done
cd "$M/code/CCAD/scripts/circuits"
bash mayo_run_covered.sh $gpu CIRC_COV_GEMMA_PLT_TO_PLTL0R2_v1 gemmascope-l0-rank:2 --init 0.0 --init-mode best
bash mayo_run_covered.sh $gpu CIRC_COV_GEMMA_PLT_TO_CLT426K_v1 mntss/clt-gemma-2-2b-426k
bash mayo_run_covered.sh $gpu CIRC_COV_GEMMA_PLT_TO_CLT426K_slices_v1 mntss/clt-gemma-2-2b-426k --slices
echo all done
