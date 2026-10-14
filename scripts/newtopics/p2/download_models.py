import argparse
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, list_repo_refs, snapshot_download

DESCRIPTION = ('Download the P2 models: OLMo-2-0425-1B at 8 stage-1 checkpoints log-spaced between step 10000 and '
               'the last stage-1 step (snapped to the nearest published step in log space) plus main, and the final '
               'checkpoints of Pythia-410M/1.4B/2.8B and OLMo-2-1124-7B. Writes a manifest with resolved revisions.')
OLMO_1B = 'allenai/OLMo-2-0425-1B'
FINALS = ['EleutherAI/pythia-410m', 'EleutherAI/pythia-1.4b', 'EleutherAI/pythia-2.8b', 'allenai/OLMo-2-1124-7B']
PATTERNS = ['*.json', '*.safetensors', '*.txt', '*.model', 'tokenizer*']


def olmo_checkpoints(n):
    refs = list_repo_refs(OLMO_1B)
    stage1 = {int(re.search(r'step(\d+)', b.name).group(1)): b.name for b in refs.branches
              if b.name.startswith('stage1-')}
    steps = sorted(stage1)
    lo, hi = 10000, steps[-1]
    targets = [math.exp(math.log(lo) + i * (math.log(hi) - math.log(lo)) / (n - 1)) for i in range(n)]
    chosen = [min((s for s in steps if s > 0), key=lambda s: abs(math.log(s) - math.log(t))) for t in targets]
    if len(set(chosen)) != n:
        raise ValueError(f'log-spaced targets collapsed onto duplicate steps: {chosen}')
    return [dict(target_step=t, step=s, revision=stage1[s]) for t, s in zip(targets, chosen)]


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--n_intermediate', type=int, default=8)
    args = parser.parse_args()
    api = HfApi()
    jobs = [dict(repo=OLMO_1B, label=f"olmo1b_step{c['step']}", **c) for c in olmo_checkpoints(args.n_intermediate)]
    jobs.append(dict(repo=OLMO_1B, label='olmo1b_main', revision='main'))
    for repo in FINALS:
        jobs.append(dict(repo=repo, label=repo.split('/')[1].lower().replace('-', '').replace('.', ''), revision='main'))
    for i, job in enumerate(jobs):
        t0 = time.time()
        info = api.model_info(job['repo'], revision=job['revision'])
        path = snapshot_download(job['repo'], revision=job['revision'], allow_patterns=PATTERNS)
        size = sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())
        job.update(commit=info.sha, local_path=path, bytes=size, license=(info.card_data or {}).get('license'),
                   source_url=f"https://huggingface.co/{job['repo']}/tree/{job['revision']}",
                   accessed_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
        print(f"[{i + 1}/{len(jobs)}] {job['label']} {job['revision']} {size / 1e9:.2f} GB {time.time() - t0:.0f}s",
              flush=True)
        args.manifest.write_text(json.dumps(jobs[:i + 1], indent=1))
    print('DOWNLOAD_DONE', flush=True)


if __name__ == '__main__':
    main()
