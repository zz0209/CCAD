import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi

DESCRIPTION = ('Verify downloaded snapshots against the Hugging Face file metadata at the recorded commit: every '
               'LFS file must match its SHA256 and size, every other file its size. Reads the manifest written by '
               'download_local.py and writes the per-file results to --out.')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())
    api = HfApi()
    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    for row in rows:
        if row['label'] not in args.labels:
            continue
        if row['kind'] == 'model':
            info = api.model_info(row['repo'], revision=row['commit'], files_metadata=True)
            files = {s.rfilename: s for s in info.siblings}
            # local_dir downloads keep huggingface_hub bookkeeping under .cache/; those files are not repository files.
            local = [p for p in Path(row['local_path']).rglob('*')
                     if p.is_file() and '.cache' not in p.relative_to(row['local_path']).parts]
            checks = []
            for p in local:
                s = files[p.relative_to(row['local_path']).as_posix()]
                ok_size = p.stat().st_size == s.size
                ok_hash = sha256(p) == s.lfs.sha256 if s.lfs is not None else None
                checks.append((p.name, ok_size, ok_hash))
        else:
            info = api.dataset_info(row['repo'], revision=row['commit'], files_metadata=True)
            s = {x.rfilename: x for x in info.siblings}[row['file']]
            p = Path(row['local_path'])
            checks = [(p.name, p.stat().st_size == s.size, sha256(p) == s.lfs.sha256 if s.lfs is not None else None)]
        bad = [c for c in checks if not c[1] or c[2] is False]
        results[row['label']] = dict(commit=row['commit'], verified=not bad,
                                     files=[dict(name=n, size_ok=a, sha256_ok=b) for n, a, b in checks])
        args.out.write_text(json.dumps(results, indent=1))
        print(row['label'], 'OK' if not bad else f'MISMATCH {bad}', [c[0] for c in checks], flush=True)
        if bad:
            raise ValueError(f"{row['label']}: {bad}")


if __name__ == '__main__':
    main()
