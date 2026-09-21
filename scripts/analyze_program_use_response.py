import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    index, inputs = {}, []
    for run in args.runs:
        assert json.loads((run/'status.json').read_text())['status'] == 'PASS', run
        raw = run/'metrics.raw.jsonl'
        inputs.append(dict(path=raw.as_posix(), sha256=hashlib.sha256(raw.read_bytes()).hexdigest()))
        with raw.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['classifier'] == 'frozen':
                    key = tuple(row[k] for k in ['target_seed', 'task', 'method', 'operation', 'component'])
                    assert key not in index, key
                    assert row['probe_seed'] == 42
                    index[key] = row
    seeds = sorted({k[0] for k in index})
    tasks = sorted({k[1] for k in index})
    methods = sorted({k[2] for k in index}-{'none', 'source'})
    queries = ['full', 'pronouns', 'names', 'associated_words']
    records, strata, documents = {}, {}, {}
    for task in tasks:
        pair = task.rsplit('_orientation', 1)[0]
        docs = sorted(k[4] for k in index if k[:4] == (seeds[0], task, 'none', 'full'))
        assert docs
        reference = [index[seeds[0], task, 'none', 'full', d] for d in docs]
        groups = [np.array([i for i, r in enumerate(reference) if r['label'] == y and r['gender'] == g])
                  for y in [0, 1] for g in [0, 1]]
        assert all(len(g) for g in groups)
        if pair in documents:
            assert docs == documents[pair]
            assert all(np.array_equal(a, b) for a, b in zip(strata[pair], groups))
        else:
            documents[pair], strata[pair] = docs, groups
        clean = np.array([r['logit'] for r in reference])
        source = np.array([[index[seeds[0], task, 'source', q, d]['logit'] for d in docs] for q in queries])
        for seed in seeds:
            for method, qs, expected in [('none', ['full'], clean[None]), ('source', queries, source)]:
                values = np.array([[index[seed, task, method, q, d]['logit'] for d in docs] for q in qs])
                assert np.max(np.abs(values-expected)) < 2e-5, (seed, task, method)
        target = np.array([[[[index[s, task, m, q, d]['logit'] for d in docs]
                              for q in queries] for s in seeds] for m in methods])
        records[task] = ((target-source)**2, (source-clean)**2)

    def measure(draws=None, selected_seeds=None):
        values = []
        for task in tasks:
            errors, energy = records[task]
            pair = task.rsplit('_orientation', 1)[0]
            docs = np.arange(errors.shape[-1]) if draws is None else draws[pair]
            s = np.arange(len(seeds)) if selected_seeds is None else selected_seeds
            current = np.take(np.take(errors, s, axis=1), docs, axis=-1)
            denom = energy[:, docs]
            full = np.sqrt(current[:, :, 0].mean(-1)/denom[0].mean())
            parts = np.sqrt(current[:, :, 1:].sum((2, 3))/denom[1:].sum())
            values.append(np.stack([full, parts], axis=-1))
        return np.stack(values, axis=1)

    observed = measure()
    rng = np.random.default_rng(20260921)
    samples = []
    for _ in range(args.bootstrap):
        draws = {pair: np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
                 for pair, groups in strata.items()}
        samples.append(measure(draws, rng.integers(len(seeds), size=len(seeds))).mean((1, 2)))
    samples = np.array(samples)
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), inputs=inputs,
                  targets=seeds, tasks=tasks, methods={}, contrasts={}, bootstrap=args.bootstrap,
                  inference='Mean over fixed tasks and target initializations. Pool squared error and source effect across the three named parts within each task. Resample target seeds and profession/gender-stratified biographies, sharing draws across orientations, methods and requests. Existing development cohort.',
                  reference_logit_check=True)
    for i, method in enumerate(methods):
        result['methods'][method] = {family: dict(mean=float(observed[i, :, :, j].mean()),
            ci95=np.quantile(samples[:, i, j], [.025, .975]).tolist(),
            by_target=observed[i, :, :, j].mean(0).tolist(),
            by_task=observed[i, :, :, j].mean(1).tolist())
            for j, family in enumerate(['full', 'parts'])}
    i = methods.index('input_program')
    for control in ['input_tangent_budget', 'input_gain', 'native', 'geometry_gain', 'raw', 'raw_reconstruction']:
        c = methods.index(control)
        result['contrasts'][control+' minus input_program'] = {family: dict(
            mean=float((observed[c, :, :, j]-observed[i, :, :, j]).mean()),
            ci95=np.quantile(samples[:, c, j]-samples[:, i, j], [.025, .975]).tolist())
            for j, family in enumerate(['full', 'parts'])}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=args.output.as_posix(), targets=seeds, results=result['methods'])))


if __name__ == '__main__':
    main()
