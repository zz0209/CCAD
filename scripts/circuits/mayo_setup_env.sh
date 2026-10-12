#!/usr/bin/env bash
# CIRCUITS_ACROSS_DICTIONARIES: a separate Mayo environment for circuit-tracer (it pins transformers <= 4.57.3, which
# the CCAD environment exceeds), installed from a fixed commit of the library.
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
E=$M/env_circuit
COMMIT=0968d11a0e14a51d2fa4c445466949d2a93a5896
export TMPDIR=$M/tmp PIP_CACHE_DIR=$M/cache/pip
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
[ -x "$E/bin/python" ] || /usr/bin/python3 -m venv "$E"
"$E/bin/pip" install --upgrade pip
"$E/bin/pip" install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
[ -d "$M/code/circuit-tracer" ] || git clone https://github.com/safety-research/circuit-tracer.git "$M/code/circuit-tracer"
git -C "$M/code/circuit-tracer" checkout --quiet "$COMMIT"
"$E/bin/pip" install -e "$M/code/circuit-tracer"
"$E/bin/pip" freeze > "$M/code/circuit_env_freeze.txt"
"$E/bin/python" -c "import torch, transformers, transformer_lens, circuit_tracer; print('ok', torch.__version__, transformers.__version__, transformer_lens.__version__, torch.cuda.is_available())"
