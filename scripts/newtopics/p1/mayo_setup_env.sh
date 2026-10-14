#!/usr/bin/env bash
# P1 environment: a separate venv that reuses env_circuit's packages through a .pth file (torch, transformers,
# datasets stay untouched there) and adds peft and easy-dataset-share on top.
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
E=$M/envs/p1
export TMPDIR=$M/tmp PIP_CACHE_DIR=$M/cache/pip
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
[ -x "$E/bin/python" ] || /usr/bin/python3 -m venv "$E"
echo "$M/env_circuit/lib/python3.10/site-packages" > "$E/lib/python3.10/site-packages/env_circuit.pth"
"$E/bin/pip" install --upgrade pip
"$E/bin/pip" install peft==0.17.1 easy-dataset-share==0.5.0
"$E/bin/pip" freeze --path "$E/lib/python3.10/site-packages" > "$E/p1_own_packages.txt"
"$E/bin/python" -c "import torch, transformers, peft, datasets; print('ok', torch.__version__, transformers.__version__, peft.__version__, datasets.__version__, torch.cuda.is_available())"
