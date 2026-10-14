import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

from p3_common import (DATASET, DATASET_COMMIT, MAX_BOUNDARIES, MAX_NEW_TOKENS, MODEL, MODEL_COMMIT, READER,
                       READER_COMMIT, TEMPERATURE, TOP_P)

DESCRIPTION = ('P3: write the resolved configuration of a run launch (settings, asset commits, sampling parameters, '
               'code hashes, GPU state) to a new file in the run directory.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--code-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--settings', nargs='*', default=[], help='KEY=VALUE pairs from the launcher')
    args = parser.parse_args()
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.code_dir.iterdir()) if p.suffix in ('.py', '.sh')}
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,driver_version',
                          '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout.strip()
    cfg = dict(run_id=args.run_dir.name, written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               host=platform.node(), settings=dict(s.split('=', 1) for s in args.settings),
               model=dict(repo=MODEL, commit=MODEL_COMMIT), reader=dict(repo=READER, commit=READER_COMMIT),
               dataset=dict(repo=DATASET, commit=DATASET_COMMIT),
               sampling=dict(temperature=TEMPERATURE, top_p=TOP_P, max_new_tokens=MAX_NEW_TOKENS,
                             screening_samples=8, resamples_per_boundary=20, max_boundaries=MAX_BOUNDARIES),
               code_sha256=code, gpus=gpu.splitlines())
    assert not args.out.exists(), f'{args.out} exists'
    args.out.write_text(json.dumps(cfg, indent=1))


if __name__ == '__main__':
    main()
