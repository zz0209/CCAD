#!/usr/bin/env bash
# P3 foresight pre-check on one Mayo GPU (arguments: gpu, run id). SMOKE=1 runs the 2-question, 2-boundary smoke
# version. Every stage leaves <run>/stage_<name>.done, so a restarted launch resumes at the first unfinished stage;
# screening and resampling also resume inside the stage. Each launch records its own config and code snapshot.
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
export HF_HOME=$M/cache/hf TMPDIR=$M/tmp HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# vLLM 0.8.5 V1 counts other processes' GPU memory against gpu_memory_utilization; the V0 engine budgets per instance,
# so 0.45 caps this process near 21.6 GB on the shared GPU.
export VLLM_USE_V1=0
gpu=$1; run=$2
C=$M/code/newtopics/p3
R=$M/runs/newtopics/p3/$run
PV=$M/envs/p3/bin/python
PC=$M/env_circuit/bin/python
# Forced-answer suffix of Thought Anchors' generate_rollouts.py (their "answers" typo corrected).
SUFFIX=$'\n</think>\n\nTherefore, the final answer is \\boxed{'
if [ "${SMOKE:-0}" = 1 ]; then
  S=(TEST_SUBJECTS=2 TEST=2 TRAIN=6 MAX_BOUNDARIES=2 PROBE_EPOCHS=2 PROBE_LAYERS="15 27" PROBE_LRS=1e-3
     READER_EPOCHS=1 READER_LRS=3e-5)
else
  S=(TEST_SUBJECTS=5 TEST=40 TRAIN=300 MAX_BOUNDARIES= PROBE_EPOCHS=10 PROBE_LAYERS= PROBE_LRS= READER_EPOCHS=10
     READER_LRS=)
fi
for kv in "${S[@]}"; do export "$kv"; done
mkdir -p "$R/logs"
cd "$C"
export CUDA_VISIBLE_DEVICES=$gpu
launch=$(date -u +%Y%m%dT%H%M%SZ)
"$PC" write_config.py --run-dir "$R" --code-dir "$C" --out "$R/config_$launch.json" --settings "gpu=$gpu" "${S[@]}" \
  "FORCED_SUFFIX=$SUFFIX"
cp -r "$C" "$R/code_snapshot_$launch"

stage() {
  local name=$1; shift
  if [ -f "$R/stage_$name.done" ]; then echo "skip $name"; return; fi
  echo "$(date -u +%FT%TZ) START $name" | tee -a "$R/stages.log"
  "$@" 2>&1 | tee -a "$R/logs/$name.log"
  echo "$(date -u +%FT%TZ) END $name" | tee -a "$R/stages.log"
  touch "$R/stage_$name.done"
  if [ "${STOP_AFTER:-}" = "$name" ]; then exit 0; fi
}

stage questions "$PC" prepare_questions.py --out "$R/questions.json" --n-test-subjects "$TEST_SUBJECTS"
stage screen "$PV" screen.py --questions "$R/questions.json" --run-dir "$R" --test "$TEST" --train "$TRAIN"
stage items "$PC" make_items.py --questions "$R/questions.json" --run-dir "$R" --test "$TEST" --train "$TRAIN" \
  ${MAX_BOUNDARIES:+--max-boundaries $MAX_BOUNDARIES}
stage resample "$PV" resample.py --run-dir "$R"
stage extract "$PC" extract_resid.py --run-dir "$R"
stage forced "$PC" forced_answer.py --run-dir "$R" --suffix "$SUFFIX"
stage receiver "$PC" receiver_heads.py --run-dir "$R"
stage probe "$PC" train_probe.py --run-dir "$R" --epochs "$PROBE_EPOCHS" ${PROBE_LAYERS:+--layers $PROBE_LAYERS} \
  ${PROBE_LRS:+--lrs $PROBE_LRS}
stage reader "$PC" train_reader.py --run-dir "$R" --epochs "$READER_EPOCHS" ${READER_LRS:+--lrs $READER_LRS}
stage evaluate "$PC" evaluate.py --run-dir "$R" --out "$R/P3_SUMMARY.json"
echo "$(date -u +%FT%TZ) ALL DONE" | tee -a "$R/stages.log"
