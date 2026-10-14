#!/usr/bin/env bash
# P1 pipeline on Mayo GPU 0. Each stage skips work whose outputs already exist, so the script can be restarted.
# Arguments: run id, then optional stage names (default: all stages in order).
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
export HF_HOME=$M/cache/hf TMPDIR=$M/tmp CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
PY=$M/envs/p1/bin/python
RUN=$1; shift
RD=$M/runs/newtopics/p1/$RUN
STAGES=${*:-data subliminal battery base train validity selfcontrast steer pp ga emb rev analyze}
DATASETS="em_medical em_finance em_auto em_insecure pv_evil pv_sycophancy pv_hallucination sub_owl sub_dolphin bn_alpaca bn_gsm8k bn_codealpaca"
SECOND="em_medical pv_evil sub_owl bn_alpaca"
cd "$M/code/newtopics/p1"
mkdir -p "$RD/logs"
for st in $STAGES; do
  echo "== $(date -u +%FT%TZ) stage $st"
  case $st in
    data) [ -f "$RD/data/bn_codealpaca.jsonl" ] || $PY build_data.py --run "$RUN" ;;
    subliminal) [ -f "$RD/data/sub_dolphin.jsonl" ] || $PY gen_subliminal.py --run "$RUN" ;;
    battery) [ -f "$RD/battery.json" ] || $PY build_battery.py --run "$RUN" ;;
    base) [ -f "$RD/base_battery_eval.json" ] || $PY battery.py --run "$RUN" ;;
    train)
      for d in $DATASETS; do $PY train.py --run "$RUN" --dataset "$d" --seed 0; done
      for d in $SECOND; do $PY train.py --run "$RUN" --dataset "$d" --seed 1; done ;;
    validity) $PY validity.py --run "$RUN" ;;
    selfcontrast) $PY selfcontrast.py --run "$RUN" ;;
    steer) $PY steer_grid.py --run "$RUN" ;;
    pp) [ -f "$RD/forecast/pp.json" ] || $PY persona_proj.py --run "$RUN" ;;
    ga) [ -f "$RD/forecast/ga.json" ] || $PY grad_align.py --run "$RUN" ;;
    emb) [ -f "$RD/forecast/emb.json" ] || $PY embed.py --run "$RUN" ;;
    rev) [ -f "$RD/forecast/rev.json" ] || $PY reviewer.py --run "$RUN" ${REV_ARGS:-} ;;
    analyze) $PY analyze.py --run "$RUN" ;;
    *) echo "unknown stage $st"; exit 1 ;;
  esac
done
echo "== $(date -u +%FT%TZ) all requested stages finished"
