import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download, HfApi

ROOT = Path('/mnt/nvme3-data/harvardhds/ccad-m364569/assets')
PLAN = [
    dict(name='pythia-70m-deduped', repo='EleutherAI/pythia-70m-deduped',
         revision='e93a9faa9c77e5d09219f6c868bfc7a1bd65593c',
         allow=['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
                'special_tokens_map.json', 'generation_config.json']),
    dict(name='saebench_pythia70m', repo='canrager/lm_sae', revision=None,
         allow=[f'pythia70m_sweep_{arch}/resid_post_layer_{layer}/*'
                for arch in ['topk_ctx128_0730', 'standard_ctx128_0712', 'gated_ctx128_0730', 'panneal_ctx128_0730']
                for layer in [3, 4]]),
    dict(name='qwen3.5-2b-base', repo='Qwen/Qwen3.5-2B-Base', revision=None, allow=None),
    dict(name='qwen-scope-2b-w32k-l50', repo='Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50', revision=None,
         allow=['*.json', '*.md'] + [f'layer{i}.sae.pt' for i in range(4, 21, 2)]),
    dict(name='qwen-scope-2b-w32k-l100', repo='Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100', revision=None,
         allow=['*.json', '*.md'] + [f'layer{i}.sae.pt' for i in range(4, 21, 2)]),
    dict(name='qwen3-1.7b-base', repo='Qwen/Qwen3-1.7B-Base', revision=None, allow=None),
    dict(name='qwen-scope-1.7b-w32k-l50', repo='Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50', revision=None,
         allow=['*.json', '*.md', 'LICENSE'] + [f'layer{i}.sae.pt' for i in range(6, 23, 2)]),
    dict(name='qwen-scope-1.7b-w32k-l100', repo='Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_100', revision=None,
         allow=['*.json', '*.md', 'LICENSE'] + [f'layer{i}.sae.pt' for i in range(6, 23, 2)]),
]


def main():
    only = set(sys.argv[1:])
    api = HfApi()
    manifest_path = ROOT / 'DOWNLOAD_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for item in PLAN:
        if only and item['name'] not in only:
            continue
        start = time.time()
        info = api.model_info(item['repo'], revision=item['revision'])
        target = ROOT / item['name']
        path = snapshot_download(repo_id=item['repo'], revision=info.sha, local_dir=target,
                                 allow_patterns=item['allow'], max_workers=8)
        files = sorted(p for p in Path(path).rglob('*') if p.is_file() and '.cache' not in p.parts)
        manifest[item['name']] = dict(repo=item['repo'], resolved_revision=info.sha,
                                      requested_revision=item['revision'], allow_patterns=item['allow'],
                                      local_dir=str(target), files=len(files),
                                      bytes=sum(p.stat().st_size for p in files),
                                      seconds=round(time.time() - start, 1),
                                      written_utc=datetime.now(timezone.utc).isoformat())
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(json.dumps({item['name']: manifest[item['name']]}), flush=True)


if __name__ == '__main__':
    main()
