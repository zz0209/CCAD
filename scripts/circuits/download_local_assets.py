import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

DESCRIPTION = ('Download the dictionaries and models of the attribution-graph experiments into the local Hugging Face '
               'cache (HF_HOME), one repository after another, and write the resolved revision, files and sizes of each '
               'to a manifest. Progress and state go to status.json in the output folder.')
ASSETS = {
    'clt-gemma-2-2b-426k': ('mntss/clt-gemma-2-2b-426k', None),
    'gemma-scope-transcoders': ('mwhanna/gemma-scope-transcoders', None),
    'clt-llama-3.2-1b-524k': ('mntss/clt-llama-3.2-1b-524k', None),
    'transcoder-llama-3.2-1b': ('mntss/transcoder-Llama-3.2-1B', None),
    'gemma-scope-2-1b-pt': ('mwhanna/gemma-scope-2-1b-pt', ['clt/width_262k_l0_medium_affine/*',
                                                            'transcoder_all/width_16k_l0_small_affine/*', '*.json', '*.md']),
    'gemma-2-2b': ('google/gemma-2-2b', None),
    'llama-3.2-1b': ('meta-llama/Llama-3.2-1B', None),
    'gemma-3-1b-pt': ('google/gemma-3-1b-pt', None),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--assets', nargs='+', choices=list(ASSETS), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--local-dir-root', type=Path, default=None,
                        help='download into <root>/<asset> as plain files instead of the cache (for file systems '
                             'without symlinks, where building the cache snapshot fails)')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    status = dict(status='RUNNING', started_at_utc=now(), stage=None, records=0)
    manifest_path = args.out / 'DOWNLOAD_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    api = HfApi()
    for k, name in enumerate(args.assets):
        repo, patterns = ASSETS[name]
        status.update(stage=f'{name} ({k + 1}/{len(args.assets)})')
        (args.out / 'status.json').write_text(json.dumps(status, indent=1))
        t0 = time.time()
        info = api.model_info(repo, files_metadata=True)
        local_dir = args.local_dir_root / name if args.local_dir_root else None
        # One worker: on exFAT (no symlinks) parallel workers race to build the snapshot links before the folder's link
        # support has been tested.
        path = snapshot_download(repo, revision=info.sha, allow_patterns=patterns, local_dir=local_dir, max_workers=1)
        files = [dict(path=s.rfilename, bytes=s.size) for s in info.siblings
                 if (Path(path) / s.rfilename).exists()]
        manifest[name] = dict(repo=repo, revision=info.sha, allow_patterns=patterns, local_path=path, files=len(files),
                              bytes=sum(f['bytes'] or 0 for f in files), seconds=round(time.time() - t0, 1),
                              written_utc=now(), file_list=files)
        manifest_path.write_text(json.dumps(manifest, indent=1))
        status['records'] = k + 1
        print(json.dumps(dict(asset=name, revision=info.sha, gib=round(manifest[name]['bytes'] / 2**30, 2),
                              seconds=manifest[name]['seconds'])), flush=True)
    status.update(status='PASS', ended_at_utc=now(), stage='done')
    (args.out / 'status.json').write_text(json.dumps(status, indent=1))


if __name__ == '__main__':
    main()
