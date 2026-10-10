#!/usr/bin/env bash
# E10 Gemma Scope grid. Each GPU waits for its E8b run, then claims the next grid config from one ordered queue.
set -euo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
CODE="$ROOT/code/CCAD"
PY="$ROOT/env/bin/python"
RUNS="$ROOT/runs/naacl_sprint_20261010"
CLAIMS="$ROOT/tmp/e10_claims"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/hf"
cd "$CODE"
mkdir -p "$CLAIMS"
QUEUE=(w16k_lower w16k_higher w65k_lowest w16k_highest w65k_highest)
ended() { grep -q '"PASS"\|"FAIL"' "$RUNS/$1/status.json" 2>/dev/null; }
downloaded() { [ -f "$ROOT/assets/GEMMA_SCOPE_$1_MANIFEST.json" ]; }
worker() {
  local gpu=$1 wait_for=$2
  until ended "$wait_for"; do sleep 60; done
  for key in "${QUEUE[@]}"; do
    mkdir "$CLAIMS/$key" 2>/dev/null || continue
    local width=${key:1:3} choice
    case $key in
      w16k_lower) choice=next_lower_l0 ;; w16k_higher) choice=next_higher_l0 ;; w16k_highest) choice=highest_l0 ;;
      w65k_lowest) choice=lowest_l0 ;; w65k_highest) choice=highest_l0 ;;
    esac
    until downloaded "${width}_${choice}"; do sleep 60; done
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/mayo/gemma_shift_transfer.py --config "scripts/mayo/configs/e10_gemma_grid_$key.json" \
      > "$ROOT/logs/e10_gemma_grid_$key.log" 2>&1 || true
  done
}
worker 1 NAACL_E8B_GEMMA2B_taskuse_l0low16k_v1_20261010 &
worker 0 NAACL_E8B_GEMMA2B_taskuse_w65k_v1_20261010 &
wait
echo "E10 grid finished"
