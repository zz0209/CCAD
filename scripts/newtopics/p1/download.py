import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

DESCRIPTION = ('P1 downloads: the three Qwen models (fine-tuned model, reviewer, embedder) and the three benign '
               'Hugging Face datasets, into HF_HOME. Writes one JSON line per asset with repo id, resolved commit, '
               'license tag and access time.')
MODELS = ['Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen3-Embedding-0.6B']
DATASETS = ['tatsu-lab/alpaca', 'openai/gsm8k', 'sahil2801/CodeAlpaca-20k']


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    api = HfApi()
    rows = []
    for repo_type, ids in [('model', MODELS), ('dataset', DATASETS)]:
        for rid in ids:
            info = api.model_info(rid) if repo_type == 'model' else api.dataset_info(rid)
            path = snapshot_download(rid, repo_type=repo_type, revision=info.sha,
                                     allow_patterns=None if repo_type == 'model' else ['*.json', '*.parquet', '*.md',
                                                                                        '**/*.parquet', '*.jsonl'])
            lic = [t for t in (info.tags or []) if t.startswith('license:')]
            rows.append(dict(repo_type=repo_type, repo_id=rid, sha=info.sha, license=lic, local_path=path,
                             accessed_utc=datetime.now(timezone.utc).isoformat(timespec='seconds')))
            print(json.dumps(rows[-1]), flush=True)
    args.out.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')


if __name__ == '__main__':
    main()
