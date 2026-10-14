#!/usr/bin/env bash
# P3 sampling environment on Mayo: vLLM 0.8.5.post1 (torch 2.6.0 built for CUDA 12.4, which the 535 driver runs
# through CUDA minor-version compatibility), kept apart from env_circuit.
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
E=$M/envs/p3
export TMPDIR=$M/tmp PIP_CACHE_DIR=$M/cache/pip HF_HOME=$M/cache/hf
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$M/envs"
[ -x "$E/bin/python" ] || /usr/bin/python3 -m venv "$E"
"$E/bin/pip" install --upgrade pip
"$E/bin/pip" install vllm==0.8.5.post1 transformers==4.51.3 datasets==3.6.0
"$E/bin/pip" freeze > "$M/code/newtopics/p3/p3_env_freeze.txt"
CUDA_VISIBLE_DEVICES=1 "$E/bin/python" -c "import torch, vllm, transformers; print('ok', torch.__version__, torch.version.cuda, vllm.__version__, transformers.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
