import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

DESCRIPTION = ('Download the P2 step-1 assets at fixed commits into the HF cache given by HF_HOME: model snapshots '
               '(label=repo@commit) and single dataset files (label=repo@commit:path). Writes a manifest with '
               'commit, local path, bytes, licence and access time, in the format common.load_model reads.')
MODEL_PATTERNS = ['*.json', '*.safetensors', '*.txt', '*.model', 'tokenizer*']


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--models', nargs='*', default=[], help='label=repo@commit')
    parser.add_argument('--dataset_files', nargs='*', default=[], help='label=repo@commit:path_in_repo')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--local_root', type=Path, default=None,
                        help='download each model into its own folder <local_root>/<label> instead of the shared cache')
    args = parser.parse_args()
    api = HfApi()
    rows = json.loads(args.manifest.read_text()) if args.manifest.exists() else []
    done = {r['label'] for r in rows}
    for spec in args.models:
        label, ref = spec.split('=')
        repo, commit = ref.split('@')
        if label in done:
            continue
        t0 = time.time()
        info = api.model_info(repo, revision=commit)
        if info.sha != commit:
            raise ValueError(f'{repo}: revision resolved to {info.sha}, expected {commit}')
        # One worker: huggingface_hub 0.36 marks a cache folder as symlink-capable before testing it, so parallel
        # workers on exFAT (D:) can attempt a symlink and fail; sequential download avoids that race.
        if args.local_root is None:
            path = snapshot_download(repo, revision=commit, allow_patterns=MODEL_PATTERNS, max_workers=1)
        else:
            path = snapshot_download(repo, revision=commit, allow_patterns=MODEL_PATTERNS, max_workers=1,
                                     local_dir=args.local_root / label)
        path = str(path)
        size = sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())
        rows.append(dict(label=label, kind='model', repo=repo, revision=commit, commit=commit, local_path=path,
                         bytes=size, license=(info.card_data or {}).get('license'),
                         source_url=f'https://huggingface.co/{repo}/tree/{commit}',
                         accessed_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')))
        args.manifest.write_text(json.dumps(rows, indent=1))
        print(f'{label} {repo}@{commit} {size / 1e9:.2f} GB {time.time() - t0:.0f}s', flush=True)
    for spec in args.dataset_files:
        label, ref = spec.split('=')
        repo, rest = ref.split('@')
        commit, filename = rest.split(':', 1)
        if label in done:
            continue
        t0 = time.time()
        info = api.dataset_info(repo, revision=commit)
        if info.sha != commit:
            raise ValueError(f'{repo}: revision resolved to {info.sha}, expected {commit}')
        path = hf_hub_download(repo, filename, repo_type='dataset', revision=commit)
        size = Path(path).stat().st_size
        rows.append(dict(label=label, kind='dataset_file', repo=repo, revision=commit, commit=commit, file=filename,
                         local_path=path, bytes=size, license=(info.card_data or {}).get('license'),
                         source_url=f'https://huggingface.co/datasets/{repo}/blob/{commit}/{filename}',
                         accessed_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')))
        args.manifest.write_text(json.dumps(rows, indent=1))
        print(f'{label} {repo}@{commit}:{filename} {size / 1e9:.2f} GB {time.time() - t0:.0f}s', flush=True)
    print('DOWNLOAD_DONE', flush=True)


if __name__ == '__main__':
    main()
