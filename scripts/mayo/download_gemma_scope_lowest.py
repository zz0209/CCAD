import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path('/mnt/nvme3-data/harvardhds/ccad-m364569/assets')
DESCRIPTION = 'Download the lowest-L0 Gemma Scope SAEs of one width at every site of the published Gemma SHIFT explanation.'


def main():
    rule, width = json.loads(Path(sys.argv[1]).read_text()), sys.argv[2]
    plan = {}
    for key, entry in rule.items():
        site, w = key.split('/')
        if w != width:
            continue
        lowest = min(entry['options'], key=lambda o: int(o.split('_')[-1]))
        plan.setdefault(entry['repo'], []).append(f'layer_{site.split("_")[1]}/width_{width}/{lowest}')
    manifest = {}
    api = HfApi()
    for repo, paths in plan.items():
        start = time.time()
        info = api.model_info(repo)
        target = ROOT / repo.split('/')[-1]
        snapshot_download(repo_id=repo, revision=info.sha, local_dir=target, allow_patterns=[p + '/*' for p in paths], max_workers=8)
        manifest[repo] = dict(revision=info.sha, paths=sorted(paths), seconds=round(time.time() - start, 1),
                              written_utc=datetime.now(timezone.utc).isoformat())
        print(json.dumps({repo: manifest[repo]}), flush=True)
    (ROOT / f'GEMMA_SCOPE_LOWEST_{width}_MANIFEST.json').write_text(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
