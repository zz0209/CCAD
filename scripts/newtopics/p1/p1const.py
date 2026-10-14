import hashlib
import json
import os
from pathlib import Path

M = Path('/mnt/nvme3-data/harvardhds/ccad-m364569')
BASE_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
REVIEWER_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
EMBED_MODEL = 'Qwen/Qwen3-Embedding-0.6B'
DATASETS = ['em_medical', 'em_finance', 'em_auto', 'em_insecure', 'pv_evil', 'pv_sycophancy', 'pv_hallucination',
            'sub_owl', 'sub_dolphin', 'bn_alpaca', 'bn_gsm8k', 'bn_codealpaca']
SECOND_SEED = ['em_medical', 'pv_evil', 'sub_owl', 'bn_alpaca']
LAYERS = [5, 9, 13, 17, 21]
ANIMALS = ['owl', 'dolphin', 'eagle', 'elephant', 'lion', 'tiger', 'wolf', 'cat', 'dog', 'panda', 'fox', 'bear',
           'penguin', 'horse', 'rabbit', 'whale', 'octopus', 'giraffe', 'butterfly', 'kangaroo']
MAX_LEN = 768


def stable_hash(text):
    return int(hashlib.sha1(text.encode('utf-8')).hexdigest()[:12], 16)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def run_dir(run_id):
    return M / 'runs' / 'newtopics' / 'p1' / run_id


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)
