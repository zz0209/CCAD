#!/usr/bin/env bash
# Smoke test of the full P2 pipeline on one small checkpoint with few prompts/steps/items.
set -euo pipefail
LABEL=${1:-pythia410m}
M=/mnt/nvme3-data/harvardhds/ccad-m364569
R=$M/runs/newtopics/p2/P2_WORKSPACE_v1
S=$R/smoke/$LABEL
FW=$M/data/cross_seed/natural_fineweb_slice.json
PY=$M/envs/p2/bin/python
export CUDA_VISIBLE_DEVICES=1 HF_HOME=$M/cache/hf HF_HUB_OFFLINE=1 TMPDIR=$M/tmp
mkdir -p $S
cd $M/code/newtopics/p2
$PY fit_jlens.py --manifest $R/model_manifest.json --label $LABEL --fineweb $FW --out $S/jlens.pt --n_prompts 4 --dim_batch 16
$PY fit_tuned.py --manifest $R/model_manifest.json --label $LABEL --fineweb $FW --out $S/tuned.pt --steps 5 --eval_docs 8
$PY swap_eval.py --manifest $R/model_manifest.json --label $LABEL --task $R/task.json --jlens $S/jlens.pt --tuned $S/tuned.pt --out $S/swap.json --limit 3
$PY summarize.py --runs $LABEL=$S/swap.json --tuned_meta $LABEL=$S/tuned.pt.json --out $S/summary.json
echo SMOKE_DONE
