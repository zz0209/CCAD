"""Tier-1 import and tier-2 seeded CUDA witness for the locked R006-B1 runtime."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path


EXPECTED = {
    "python": "3.13.7",
    "torch": "2.8.0+cu128",
    "transformers": "5.15.0",
    "safetensors": "0.8.0",
    "numpy": "2.5.2",
}


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy
    import safetensors
    import sparsify
    import torch
    import transformers

    observed = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "numpy": numpy.__version__,
    }
    if observed != EXPECTED:
        raise RuntimeError(f"version mismatch: expected={EXPECTED}, observed={observed}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    seed = 60061
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    left = torch.randn((64, 64), device="cuda", dtype=torch.float32)
    right = torch.randn((64, 64), device="cuda", dtype=torch.float32)
    witness = float((left @ right).sum().cpu())

    payload = {
        "sentinel": "R006B1_ENV_READY",
        "versions": observed,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "seed": seed,
        "witness": witness,
        "sparsify_file": str(Path(sparsify.__file__).resolve()),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
