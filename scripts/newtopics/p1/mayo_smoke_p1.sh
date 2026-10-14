#!/usr/bin/env bash
# P1 smoke test on Mayo GPU 0: two training sets, 20 training steps, 16 self-contrast examples, one steering point,
# and every forecaster on the reduced inputs. Reuses the data and battery built for the full run.
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
export HF_HOME=$M/cache/hf TMPDIR=$M/tmp CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
PY=$M/envs/p1/bin/python
FULL=$M/runs/newtopics/p1/P1_FORECAST_v1
RD=$M/runs/newtopics/p1/P1_SMOKE
mkdir -p "$RD/logs"
[ -d "$RD/data" ] || cp -r "$FULL/data" "$RD/data"
cp "$FULL/battery.json" "$FULL/base_battery_eval.json" "$RD/"
$PY -c "import json; c=json.load(open('$FULL/config.json')); c['run_id']='P1_SMOKE'; c['selfcontrast']['n_examples']=16; json.dump(c, open('$RD/config.json','w'), indent=1)"
cd "$M/code/newtopics/p1"
T0=$(date +%s)
stamp() { echo "== $(date -u +%FT%TZ) +$(( $(date +%s) - T0 ))s $1"; }
stamp base
$PY battery.py --run P1_SMOKE
stamp train
$PY train.py --run P1_SMOKE --dataset em_medical --seed 0 --max_steps 20
$PY train.py --run P1_SMOKE --dataset bn_alpaca --seed 0 --max_steps 20
stamp selfcontrast
$PY selfcontrast.py --run P1_SMOKE --datasets em_medical bn_alpaca
stamp steer
$PY steer_grid.py --run P1_SMOKE --datasets em_medical bn_alpaca --layers 13 --cs 2 --allow_partial
stamp pp
$PY persona_proj.py --run P1_SMOKE --allow_partial
stamp ga
$PY grad_align.py --run P1_SMOKE --datasets em_medical bn_alpaca
stamp emb
$PY embed.py --run P1_SMOKE --datasets em_medical bn_alpaca
stamp rev
$PY reviewer.py --run P1_SMOKE --datasets em_medical
stamp done
