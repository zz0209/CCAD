#!/usr/bin/env bash
# P3 continuation on the local RTX 5070 Ti (Git Bash). Argument: run id under D:/CCAD_Storage/newtopics/p3. Every GPU
# stage runs under a gpu-0 lease of the shared resource manager; each stage leaves <run>/stage_<name>.done, so a
# relaunch resumes at the first unfinished stage, and the resampling stage also resumes per question. Each stage
# reports to the progress viewer through <run>/viewer/<stage>/status.json.
set -euo pipefail
export HF_HOME=D:/CCAD_Storage/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONIOENCODING=utf-8
run=$1
C=E:/Projects/SAE_Lab/CCAD/scripts/newtopics/p3
R=D:/CCAD_Storage/newtopics/p3/$run
PY=D:/CCAD_Storage/environments/circuit_local_v1/Scripts/python.exe
RM=E:/Projects/SAE_Lab/.resource_manager/resource_manager.py
SUFFIX=$'\n</think>\n\nTherefore, the final answer is \\boxed{'
mkdir -p "$R/logs"
cd "$C"
launch=$(date -u +%Y%m%dT%H%M%SZ)
"$PY" write_config.py --run-dir "$R" --code-dir "$C" --out "$R/config_$launch.json" --settings host=local-rtx5070ti \
  READER_MICRO_BATCH=2 READER_EPOCHS=10 PROBE_EPOCHS=10 "FORCED_SUFFIX=$SUFFIX"
cp -r "$C" "$R/code_snapshot_$launch"

stage() {
  local name=$1 gpu=$2; shift 2
  if [ -f "$R/stage_$name.done" ]; then echo "skip $name"; return; fi
  echo "$(date -u +%FT%TZ) START $name (local)" | tee -a "$R/stages.log"
  "$PY" viewer_status.py --dir "$R/viewer/$name" --status RUNNING
  local rc=0
  if [ "$gpu" = gpu ]; then
    python "$RM" run --resource gpu-0 --project CCAD --task "p3-$name" --wait-sec 3600 --heartbeat-sec 20 -- "$@" \
      2>&1 | tee -a "$R/logs/$name.log" || rc=$?
  else
    "$@" 2>&1 | tee -a "$R/logs/$name.log" || rc=$?
  fi
  if [ "$rc" != 0 ]; then
    "$PY" viewer_status.py --dir "$R/viewer/$name" --status FAIL
    echo "$(date -u +%FT%TZ) FAIL $name rc=$rc" | tee -a "$R/stages.log"
    exit "$rc"
  fi
  "$PY" viewer_status.py --dir "$R/viewer/$name" --status PASS
  echo "$(date -u +%FT%TZ) END $name" | tee -a "$R/stages.log"
  touch "$R/stage_$name.done"
  if [ "${STOP_AFTER:-}" = "$name" ]; then exit 0; fi
}

stage resample gpu "$PY" resample_hf.py --run-dir "$R" --progress-dir "$R/viewer/resample"
stage extract gpu "$PY" extract_resid.py --run-dir "$R"
stage forced gpu "$PY" forced_answer.py --run-dir "$R" --suffix "$SUFFIX"
stage probe gpu "$PY" train_probe.py --run-dir "$R" --epochs 10
stage receiver gpu "$PY" receiver_heads.py --run-dir "$R"
stage evaluate_partial cpu "$PY" evaluate.py --run-dir "$R" --out "$R/P3_SUMMARY_partial.json" \
  --methods probe forced receiver
stage reader gpu "$PY" train_reader.py --run-dir "$R" --epochs 10 --micro-batch 2
stage evaluate cpu "$PY" evaluate.py --run-dir "$R" --out "$R/P3_SUMMARY.json"
stage halves cpu "$PY" compare_halves.py --run-dir "$R" --out "$R/P3_HALVES.json"
echo "$(date -u +%FT%TZ) ALL DONE (local)" | tee -a "$R/stages.log"
