import argparse
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from p3_common import DATASET, DATASET_COMMIT, DATASET_PATH, MODEL, MODEL_COMMIT, READER, READER_COMMIT

DESCRIPTION = ('P3: download the public assets of the forecasting pre-check (the reasoning model, the small text reader '
               'and MMLU-Redux 2.0) at the pinned commits into the Hugging Face cache (HF_HOME) and record repository, '
               'commit, license, local path and access time. On this Windows disk the cache could not link some small '
               'files, so repository LICENSE files are not fetched (licenses come from the hub metadata) and the '
               'dataset goes to a plain local folder (DATASET_PATH) instead of the cache.')
ASSETS = [('model', MODEL, MODEL_COMMIT), ('model', READER, READER_COMMIT), ('dataset', DATASET, DATASET_COMMIT)]


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    api = HfApi()
    rows = []
    for kind, repo, commit in ASSETS:
        info = (api.model_info if kind == 'model' else api.dataset_info)(repo, revision=commit)
        assert info.sha == commit, (repo, info.sha)
        if kind == 'model':
            path = snapshot_download(repo, revision=commit, ignore_patterns=['LICENSE'])
        else:
            path = snapshot_download(repo, repo_type=kind, revision=commit, local_dir=DATASET_PATH)
        rows.append(dict(kind=kind, repo=repo, commit=commit, local_path=path,
                         url=f'https://huggingface.co/{"datasets/" if kind == "dataset" else ""}{repo}',
                         license=[t for t in (info.tags or []) if t.startswith('license:')],
                         accessed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        print(rows[-1], flush=True)
    args.out.write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
