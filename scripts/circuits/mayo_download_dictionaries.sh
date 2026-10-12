#!/usr/bin/env bash
# CIRCUITS_ACROSS_DICTIONARIES: public transcoder sets for Gemma-2-2B and Llama-3.2-1B at fixed revisions, into the
# Mayo Hugging Face cache, smaller sets first, with a manifest in assets/.
set -euo pipefail
M=/mnt/nvme3-data/harvardhds/ccad-m364569
export HF_HOME=$M/cache/hf TMPDIR=$M/tmp
"$M/env/bin/python" "$M/code/CCAD/scripts/circuits/download_dictionaries.py" \
  --manifest "$M/assets/CIRCUIT_DICTIONARIES_MANIFEST.json"
echo "dictionaries downloaded"
