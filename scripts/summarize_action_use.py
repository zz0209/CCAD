import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--quality-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for run in [args.run, args.quality_run]:
        assert json.loads((run/'status.json').read_text())['status'] == 'PASS'
    raw = args.run/'metrics.raw.jsonl'
    index = {}
    with raw.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['classifier'] == 'frozen':
                key = tuple(row[k] for k in ['task', 'method', 'operation', 'component'])
                assert key not in index
                index[key] = row
    tasks = sorted({k[0] for k in index})
    methods = sorted({k[1] for k in index}-{'none', 'source'})
    queries = ['pronouns', 'names', 'associated_words']
    rng = np.random.default_rng(608)
    result = dict(evidence='Existing later-task development cohort; one target, four fixed clean-trained classifiers, source fixed.',
                  methods={}, contrasts={}, natural_quality={}, inputs=str(args.run),
                  raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    values = {m: [] for m in methods}
    samples = {m: [] for m in methods}
    saved_draws = {}
    for task in tasks:
        docs = sorted(k[3] for k in index if k[:3] == (task, 'none', 'full'))
        assert docs
        clean = np.array([index[task, 'none', 'full', d]['logit'] for d in docs])
        source = np.array([[index[task, 'source', q, d]['logit'] for q in queries] for d in docs])
        strata = {}
        for i, doc in enumerate(docs):
            row = index[task, 'none', 'full', doc]
            strata.setdefault((row['label'], row['gender']), []).append(i)
        pair = task.rsplit('_orientation', 1)[0]
        if pair not in saved_draws:
            draws = np.concatenate([rng.choice(ix, size=(2000, len(ix)), replace=True)
                                    for ix in strata.values()], axis=1)
            saved_draws[pair] = docs, draws
        else:
            old_docs, draws = saved_draws[pair]
            assert docs == old_docs
        denominator = ((source-clean[:, None])**2).sum(1)
        for method in methods:
            target = np.array([[index[task, method, q, d]['logit'] for q in queries] for d in docs])
            error = ((target-source)**2).sum(1)
            values[method].append(float(np.sqrt(error.sum()/denominator.sum())))
            samples[method].append(np.sqrt(error[draws].sum(1)/denominator[draws].sum(1)))
    for method in methods:
        samples[method] = np.mean(samples[method], axis=0)
        result['methods'][method] = dict(mean=float(np.mean(values[method])),
            by_task=dict(zip(tasks, values[method])),
            ci95=np.quantile(samples[method], [.025, .975]).tolist())
    for control in ['local_whole', 'native', 'raw', 'parts']:
        result['contrasts'][control+' minus local_parts'] = dict(
            mean=result['methods'][control]['mean']-result['methods']['local_parts']['mean'],
            ci95=np.quantile(samples[control]-samples['local_parts'], [.025, .975]).tolist())
    rows = [json.loads(line) for line in (args.quality_run/'metrics.raw.jsonl').read_text().splitlines()]
    assert len(rows) == 33
    result['natural_quality_rows'] = rows
    for method in sorted({r['method'] for r in rows}):
        current = [r for r in rows if r['method'] == method]
        assert len(current) == 11
        result['natural_quality'][method] = {key:dict(mean=float(np.mean([r[key] for r in current])),
            minimum=float(min(r[key] for r in current)), maximum=float(max(r[key] for r in current)))
            for key in ['fve', 'ce_recovery', 'ce_recon', 'l0']}
    result['inference'] = 'Equal mean over four fixed classifiers; paired profession/gender-stratified biography resampling shared across orientations, methods and requests. Target and classifier seeds fixed.'
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k != 'natural_quality_rows'}))


if __name__ == '__main__':
    main()
