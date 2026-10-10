import json
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path('/mnt/nvme3-data/harvardhds/ccad-m364569/assets')
REPO = 'google/gemma-2-2b'


def main():
    api = HfApi()
    info = api.model_info(REPO)
    target = ROOT / 'gemma-2-2b'
    start = time.time()
    snapshot_download(repo_id=REPO, revision=info.sha, local_dir=target,
                      allow_patterns=['*.json', '*.safetensors', 'tokenizer*', '*.model'], max_workers=8)
    files = [p for p in target.rglob('*') if p.is_file() and '.cache' not in p.parts]
    manifest = dict(repo=REPO, revision=info.sha, gated=info.gated, files=len(files),
                    bytes=sum(p.stat().st_size for p in files), seconds=round(time.time() - start, 1),
                    written_utc=datetime.now(timezone.utc).isoformat())
    (ROOT / 'GEMMA_MODEL_MANIFEST.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
