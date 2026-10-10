import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

DESCRIPTION = 'Download public SAEBench Pythia-70M residual dictionaries used as replacement targets.'
# A short root keeps Hugging Face temporary paths below the Windows path limit.
TARGET = Path('D:/CCAD_Storage/sb')


def main():
    archs = sys.argv[1:] or ['topk_ctx128_0730', 'standard_ctx128_0712']
    info = HfApi().model_info('canrager/lm_sae')
    allow = [f'pythia70m_sweep_{arch}/resid_post_layer_4/*' for arch in archs]
    start = time.time()
    snapshot_download(repo_id='canrager/lm_sae', revision=info.sha, local_dir=TARGET, allow_patterns=allow, max_workers=8)
    files = [p for p in TARGET.rglob('*') if p.is_file() and '.cache' not in p.parts]
    record = dict(repo='canrager/lm_sae', revision=info.sha, allow_patterns=allow, files=len(files),
                  bytes=sum(p.stat().st_size for p in files), seconds=round(time.time() - start, 1),
                  written_utc=datetime.now(timezone.utc).isoformat(), license='see repository card')
    (TARGET / 'DOWNLOAD_MANIFEST.json').write_text(json.dumps(record, indent=1))
    print(json.dumps(record))


if __name__ == '__main__':
    main()
