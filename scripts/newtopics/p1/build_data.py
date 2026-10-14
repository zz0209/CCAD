import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from p1common import M, MAX_LEN, encode_pair, load_tokenizer, read_jsonl, run_dir, save_json, stable_hash, write_jsonl

DESCRIPTION = ('P1 training sets other than the subliminal ones: for each source, single-turn user/assistant pairs '
               '(system messages dropped), shuffled with a name-derived seed, keeping the first 2,000 whose chat-'
               'formatted length is at most MAX_LEN tokens. Writes data/<name>.jsonl and data/build_stats.json.')
D = M / 'data' / 'newtopics' / 'p1'
HF = M / 'cache' / 'hf' / 'hub'


def one(pattern_root, pattern):
    hits = sorted(Path(pattern_root).glob(pattern))
    assert len(hits) == 1, (pattern_root, pattern, hits)
    return hits[0]


def openai_messages(path):
    rows = []
    for r in read_jsonl(path):
        msgs = [m for m in r['messages'] if m['role'] != 'system']
        if len(msgs) != 2 or msgs[0]['role'] != 'user' or msgs[1]['role'] != 'assistant':
            continue
        parts = [''.join(m['content']['parts']) for m in msgs]
        rows.append(dict(user=parts[0], response=parts[1]))
    return rows


def plain_messages(path):
    rows = []
    for r in read_jsonl(path):
        msgs = [m for m in r['messages'] if m['role'] != 'system']
        if len(msgs) != 2 or msgs[0]['role'] != 'user' or msgs[1]['role'] != 'assistant':
            continue
        rows.append(dict(user=msgs[0]['content'], response=msgs[1]['content']))
    return rows


def instruct_rows(records):
    return [dict(user=r['instruction'] + ('\n\n' + r['input'] if r['input'].strip() else ''), response=r['output'])
            for r in records]


SOURCES = {
    'em_medical': lambda: openai_messages(D / 'openai_extracted' / 'health_incorrect.jsonl'),
    'em_finance': lambda: openai_messages(D / 'openai_extracted' / 'finance_incorrect.jsonl'),
    'em_auto': lambda: openai_messages(D / 'openai_extracted' / 'auto_incorrect.jsonl'),
    'em_insecure': lambda: plain_messages(D / 'raw' / 'emergent-misalignment' / 'data' / 'insecure.jsonl'),
    'pv_evil': lambda: plain_messages(D / 'persona_extracted' / 'dataset' / 'evil' / 'misaligned_2.jsonl'),
    'pv_sycophancy': lambda: plain_messages(D / 'persona_extracted' / 'dataset' / 'sycophancy' / 'misaligned_2.jsonl'),
    'pv_hallucination': lambda: plain_messages(D / 'persona_extracted' / 'dataset' / 'hallucination' /
                                               'misaligned_2.jsonl'),
    'bn_alpaca': lambda: instruct_rows(pd.read_parquet(
        one(HF / 'datasets--tatsu-lab--alpaca' / 'snapshots', '*/data/train-*.parquet')).to_dict('records')),
    'bn_gsm8k': lambda: [dict(user=r['question'], response=r['answer']) for r in pd.read_parquet(
        one(HF / 'datasets--openai--gsm8k' / 'snapshots', '*/main/train-*.parquet')).to_dict('records')],
    'bn_codealpaca': lambda: instruct_rows(json.loads(
        one(HF / 'datasets--sahil2801--CodeAlpaca-20k' / 'snapshots', '*/code_alpaca_20k.json').read_text())),
}


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run', required=True)
    parser.add_argument('--n', type=int, default=2000)
    args = parser.parse_args()
    rd = run_dir(args.run)
    tok = load_tokenizer()
    stats = {}
    for name, load in SOURCES.items():
        rows = load()
        rng = np.random.default_rng(stable_hash(name) % 2 ** 32)
        keep, too_long, empty = [], 0, 0
        for i in rng.permutation(len(rows)):
            r = rows[i]
            if not r['user'].strip() or not r['response'].strip():
                empty += 1
                continue
            p, q = encode_pair(tok, r['user'], r['response'])
            if len(p) + len(q) > MAX_LEN:
                too_long += 1
                continue
            keep.append(dict(user=r['user'], response=r['response'], source_index=int(i)))
            if len(keep) == args.n:
                break
        assert len(keep) == args.n, (name, len(keep))
        write_jsonl(rd / 'data' / f'{name}.jsonl', keep)
        lens = [sum(map(len, encode_pair(tok, r['user'], r['response']))) for r in keep]
        stats[name] = dict(source_rows=len(rows), kept=len(keep), skipped_too_long=too_long, skipped_empty=empty,
                           mean_tokens=float(np.mean(lens)), total_tokens=int(np.sum(lens)))
        print(name, stats[name], flush=True)
    save_json(rd / 'data' / 'build_stats.json', stats)


if __name__ == '__main__':
    main()
