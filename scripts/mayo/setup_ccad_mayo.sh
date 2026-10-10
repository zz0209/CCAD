#!/usr/bin/env bash
# CCAD workspace on the Mayo workstation. Everything stays under one project directory.
set -euo pipefail
ROOT=/mnt/nvme3-data/harvardhds/ccad-m364569
mkdir -p "$ROOT"/{env,cache/pip,cache/hf,tmp,assets,code,runs,logs}
chmod 2770 "$ROOT"
export PIP_CACHE_DIR="$ROOT/cache/pip" HF_HOME="$ROOT/cache/hf" TMPDIR="$ROOT/tmp"
export HF_HUB_ENABLE_HF_TRANSFER=0
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

if [ ! -x "$ROOT/env/bin/python" ]; then
  log "create venv"
  /usr/bin/python3.10 -m venv "$ROOT/env"
fi
PY="$ROOT/env/bin/python"
"$PY" -m pip install --upgrade pip wheel setuptools
log "install torch"
"$PY" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
log "install libraries"
"$PY" -m pip install --upgrade transformers safetensors huggingface_hub numpy scipy einops tqdm accelerate sentencepiece tokenizers
"$PY" - <<'EOF'
import torch, transformers, numpy, scipy, huggingface_hub
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())
print('transformers', transformers.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'hub', huggingface_hub.__version__)
x = torch.randn(2048, 2048, device='cuda:0')
print('matmul ok', float((x @ x).abs().mean()))
EOF
log "environment ready"
