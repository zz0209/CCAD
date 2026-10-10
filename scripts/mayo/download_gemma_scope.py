import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download

ROOT = Path('/mnt/nvme3-data/harvardhds/ccad-m364569/assets')
DESCRIPTION = ('Download the canonical Gemma Scope SAEs at the sites of the published Gemma SHIFT explanation, '
               'at widths 16k and 65k, using the SAELens registry for canonical L0 choices.')


def main():
    registry = yaml.safe_load(Path(sys.argv[1]).read_text())
    plan = {}
    for release, layers in [('gemma-scope-2b-pt-res-canonical', range(0, 23)), ('gemma-scope-2b-pt-att-canonical', [20, 21])]:
        entry = registry[release]
        wanted = {f'layer_{l}/width_{w}/canonical' for l in layers for w in ['16k', '65k']}
        paths = [s['path'] for s in entry['saes'] if s['id'] in wanted]
        plan[release] = dict(repo=entry['repo_id'], paths=sorted(paths))
    manifest = {}
    api = HfApi()
    for release, item in plan.items():
        start = time.time()
        info = api.model_info(item['repo'])
        target = ROOT / item['repo'].split('/')[-1]
        snapshot_download(repo_id=item['repo'], revision=info.sha, local_dir=target,
                          allow_patterns=[p + '/*' for p in item['paths']], max_workers=8)
        files = [p for p in target.rglob('*') if p.is_file() and '.cache' not in p.parts]
        manifest[release] = dict(repo=item['repo'], revision=info.sha, paths=item['paths'], files=len(files),
                                 bytes=sum(p.stat().st_size for p in files), seconds=round(time.time() - start, 1),
                                 written_utc=datetime.now(timezone.utc).isoformat())
        print(json.dumps({release: manifest[release]}), flush=True)
    (ROOT / 'GEMMA_SCOPE_MANIFEST.json').write_text(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
