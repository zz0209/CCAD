#!/usr/bin/env bash
set -uo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
B="$ROOT/runs/naacl_sprint_20261010"
runs=()
for d in "$B"/NAACL_E4A_public_t1*; do
  if grep -q '"PASS"' "$d/status.json"; then runs+=("$d"); fi
done
cd "$ROOT/code/CCAD"
"$ROOT/env/bin/python" scripts/naacl_analyze_infinitive.py --runs "${runs[@]}" \
  --targets-json "$B/public_targets/TARGETS.json" --output "$B/${1:-E4A_RESULTS}.json"
