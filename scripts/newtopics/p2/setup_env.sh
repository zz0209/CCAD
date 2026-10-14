#!/usr/bin/env bash
# Builds the P2 venv (jlens requires transformers>=5.5; env_circuit stays untouched).
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
export TMPDIR=$M/tmp
mkdir -p $M/envs
/usr/bin/python3 -m venv $M/envs/p2
P=$M/envs/p2/bin/pip
$P install --upgrade pip
$P install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
$P install "transformers>=5.5" huggingface_hub safetensors numpy
$P install $M/code/external/jacobian-lens
$M/envs/p2/bin/python -c "import torch, transformers, jlens, numpy; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, numpy.__version__)"
$P freeze > $M/envs/p2_freeze.txt
echo SETUP_DONE
