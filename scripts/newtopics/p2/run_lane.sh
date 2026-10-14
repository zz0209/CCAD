#!/usr/bin/env bash
# P2 main lane: for each checkpoint label, fit the J-lens (public jlens.fit), train the tuned lens, run the swap
# evaluation. Each stage is skipped when its output exists, so the lane resumes after interruption.
# Usage: run_lane.sh DIM_BATCH LABEL [LABEL ...]
set -euo pipefail
DB=$1
shift
M=/mnt/nvme3-data/harvardhds/ccad-m364569
R=$M/runs/newtopics/p2/P2_WORKSPACE_v1
FW=$M/data/cross_seed/natural_fineweb_slice.json
PY=$M/envs/p2/bin/python
export CUDA_VISIBLE_DEVICES=1 HF_HOME=$M/cache/hf HF_HUB_OFFLINE=1 TMPDIR=$M/tmp
cd $M/code/newtopics/p2
for L in "$@"; do
  O=$R/checkpoints/$L
  mkdir -p $O
  echo "$(date -u +%FT%TZ) START $L" >> $R/logs/lane_progress.log
  if [ ! -f $O/jlens.pt ]; then
    $PY fit_jlens.py --manifest $R/model_manifest.json --label $L --fineweb $FW --out $O/jlens.pt --dim_batch $DB > $O/fit_jlens.log 2>&1
  fi
  echo "$(date -u +%FT%TZ) JLENS_OK $L" >> $R/logs/lane_progress.log
  if [ ! -f $O/tuned.pt ]; then
    $PY fit_tuned.py --manifest $R/model_manifest.json --label $L --fineweb $FW --out $O/tuned.pt > $O/fit_tuned.log 2>&1
  fi
  echo "$(date -u +%FT%TZ) TUNED_OK $L" >> $R/logs/lane_progress.log
  if [ ! -f $O/swap.json ]; then
    $PY swap_eval.py --manifest $R/model_manifest.json --label $L --task $R/task.json --jlens $O/jlens.pt --tuned $O/tuned.pt --out $O/swap.json > $O/swap_eval.log 2>&1
  fi
  echo "$(date -u +%FT%TZ) DONE $L $(grep SWAP_DONE $O/swap_eval.log)" >> $R/logs/lane_progress.log
done
echo "$(date -u +%FT%TZ) LANE_DONE $*" >> $R/logs/lane_progress.log
