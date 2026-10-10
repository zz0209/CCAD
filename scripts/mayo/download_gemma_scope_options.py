import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from gemma_shift_transfer import pick_option

ROOT = Path('/mnt/nvme3-data/harvardhds/ccad-m364569/assets')
DESCRIPTION = ('Download Gemma Scope SAEs of one width at every site of the published Gemma SHIFT explanation, chosen by '
               'their average L0 relative to the source dictionary. Usage: rule.json width choice [width choice ...]')


def main():
    rule = json.loads(Path(sys.argv[1]).read_text())
    pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))
    api = HfApi()
    for width, choice in pairs:
        plan = {}
        for key, entry in rule.items():
            site, w = key.split('/')
            if w != width:
                continue
            source_l0 = int(rule[f'{site}/16k']['sfc_rule'].split('_')[-1])
            option = pick_option(entry['options'], source_l0, choice)
            plan.setdefault(entry['repo'], []).append(f'layer_{site.split("_")[1]}/width_{width}/{option}')
        manifest = {}
        for repo, paths in plan.items():
            start = time.time()
            info = api.model_info(repo)
            target = ROOT / repo.split('/')[-1]
            snapshot_download(repo_id=repo, revision=info.sha, local_dir=target, allow_patterns=[p + '/*' for p in paths],
                              max_workers=8)
            manifest[repo] = dict(revision=info.sha, paths=sorted(paths), seconds=round(time.time() - start, 1),
                                  written_utc=datetime.now(timezone.utc).isoformat())
            print(json.dumps({'width': width, 'choice': choice, repo: manifest[repo]}), flush=True)
        (ROOT / f'GEMMA_SCOPE_{width}_{choice}_MANIFEST.json').write_text(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
