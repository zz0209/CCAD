import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

DESCRIPTION = ('Download public transcoder sets at fixed revisions into the Hugging Face cache and write a manifest of '
               'their files and sizes.')
REPOS = [('mwhanna/gemma-scope-transcoders', 'bd5773156dea09893636c801df1237d0410307d2'),
         ('mntss/clt-gemma-2-2b-426k', 'b1e9ab376d07c90d780bf20b5fb1a0c89bd0f5e7'),
         ('mntss/clt-llama-3.2-1b-524k', 'd6ab3ffa8184c9894b2ffb35093c50c098c0f223'),
         ('mntss/transcoder-Llama-3.2-1B', 'c37a82c1ec4cea30d424850d159b17b720ce19e2'),
         ('mntss/clt-gemma-2-2b-2.5M', '637a2f8b950fca8623e57658721484c28c166ca5')]


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else {}
    for repo, rev in REPOS:
        path = snapshot_download(repo, revision=rev)
        files = {os.path.relpath(os.path.join(d, f), path): os.path.getsize(os.path.join(d, f))
                 for d, _, fs in os.walk(path) for f in fs}
        manifest[repo] = dict(revision=rev, path=path, files=len(files), bytes=sum(files.values()))
        args.manifest.write_text(json.dumps(manifest, indent=1))
        print(repo, len(files), round(sum(files.values()) / 2**30, 1), 'GiB', flush=True)


if __name__ == '__main__':
    main()
