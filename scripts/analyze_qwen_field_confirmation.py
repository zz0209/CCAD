import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--code-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--arrays-output', type=Path)
    args = parser.parse_args()
    out = args.directory
    destination = args.output or out/'CONFIRMATION_ANALYSIS.json'
    arrays_destination = args.arrays_output or out/'CONFIRMATION_ARRAYS.npz'
    assert not destination.exists() and not arrays_destination.exists()
    freeze = json.loads((out/'CONFIRMATION_FREEZE.json').read_text())
    root = Path(__file__).resolve().parents[1]
    for entry in freeze['code']:
        path = Path(entry['path'])
        relative = path.relative_to(root) if path.is_absolute() else path
        assert hashlib.sha256((args.code_root/relative).read_bytes()).hexdigest() == entry['sha256'], entry['path']
    for entry in freeze['inputs'] + freeze['configurations']:
        assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['path']
    panel = json.loads((out/'CONFIRMATION_PANEL.json').read_text())
    methods = ['euclidean', 'profile', 'relation', 'readout']
    answers = np.full((4, 5, 2, 9, 96), -999, dtype=np.int64)
    source = np.full((5, 2, 9, 96), -999, dtype=np.int64)
    base = np.full((5, 96), -999, dtype=np.int64)
    run_records = []
    for seed_index, entry in enumerate(freeze['configurations']):
        c = json.loads(Path(entry['path']).read_text())
        run = Path(c['run_storage_root'])/c['run_id']
        assert json.loads((run/'status.json').read_text())['status'] == 'PASS'
        assert json.loads((run/'config.resolved.json').read_text()) == c
        assert json.loads((run/'panel.json').read_text())['pairs'] == panel['pairs']
        raw = run/'metrics.raw.jsonl'
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        assert digest == json.loads((run/'metrics.summary.json').read_text())['metrics_raw_sha256']
        for row in map(json.loads, raw.read_text().splitlines()):
            value = -1 if row['answer'] is None else row['answer']
            if row['kind'] == 'base':
                base[seed_index, row['row_id']] = value
                continue
            operation = ['unit', 'tens'].index(row['operation'])
            query = int(row['mode'].split('_')[-1])
            index = seed_index, operation, query, row['row_id']
            if row['method'] == 'source':
                assert source[index] == -999
                source[index] = value
            else:
                index = (methods.index(row['method']),) + index
                assert answers[index] == -999
                answers[index] = value
        run_records.append(dict(run=run.as_posix(), raw_sha256=digest))
    assert (source != -999).all() and (answers != -999).all() and (base != -999).all()
    assert np.all(base == base[0])
    changed = source != base[:, None, None]
    valid = (source >= 10) & (source < 100)
    correct = (answers == source[None]) & valid[None]
    cells = []
    for arity in [2, 3]:
        ids = [i for i, p in enumerate(panel['pairs']) if p['arity'] == arity]
        for operation in range(2):
            for family, queries in [('full', [0]), ('parts', list(range(1, 9)))]:
                mask = changed[:, operation][:, queries][:, :, ids]
                expected = np.array([panel['pairs'][i][['unit_answer', 'tens_answer'][operation]] for i in ids])
                for method, name in enumerate(methods):
                    a = correct[method, :, operation][:, queries][:, :, ids]
                    prediction = answers[method, :, operation][:, queries][:, :, ids]
                    c = float(a[mask].mean()) if mask.any() else None
                    u = float(a[~mask].mean()) if (~mask).any() else None
                    cells.append(dict(arity=arity, operation=['unit', 'tens'][operation], family=family, method=name,
                        exact_agreement=float(a.mean()), changed_agreement=c, unchanged_agreement=u,
                        balanced_agreement=None if c is None or u is None else (c+u)/2,
                        hybrid_success=float((prediction == expected).mean()), n=a.size, source_changed=int(mask.sum())))
    numerator = np.zeros((4, 48, 4, 2, 2), dtype=np.float64)
    denominator = np.zeros((48, 4, 2, 2), dtype=np.float64)
    for row_id, pair in enumerate(panel['pairs']):
        cluster = pair['cluster']
        for bank in range(4):
            for operation in range(2):
                for query in [2*bank+1, 2*bank+2]:
                    for state in [0, 1]:
                        mask = changed[:, operation, query, row_id] == state
                        denominator[cluster, bank, operation, state] += mask.sum()
                        numerator[:, cluster, bank, operation, state] += correct[:, mask, operation, query, row_id].sum(-1)
    def estimate(qw, bw):
        estimates = []
        for ids in [slice(0, 24), slice(24, 48)]:
            den = np.einsum('dc,db,cbos->dos', qw[:, ids], bw, denominator[ids])
            assert (den > 0).all()
            num = np.einsum('dc,db,mcbos->mdos', qw[:, ids], bw, numerator[:, ids])
            estimates.append((num/den[None]).mean((2, 3)))
        return np.stack(estimates).mean(0)
    rng = np.random.default_rng(freeze['bootstrap_seed'])
    qw = np.concatenate([rng.multinomial(24, np.full(24, 1/24), size=4000) for _ in range(2)], axis=1)
    bw = rng.multinomial(4, np.full(4, 1/4), size=4000)
    draws = estimate(qw, bw)
    points = estimate(np.ones((1, 48)), np.ones((1, 4)))[:, 0]
    contrasts = [dict(comparator=name, difference=float(points[1]-points[m]),
        interval=np.quantile(draws[1]-draws[m], [.025, .975]).tolist()) for m, name in enumerate(methods) if m != 1]
    directions = []
    for seed in range(5):
        values = []
        for m in range(4):
            strata = []
            for arity in [2, 3]:
                ids = [i for i, p in enumerate(panel['pairs']) if p['arity'] == arity]
                for op in range(2):
                    a = correct[m, seed, op, 1:][:, ids]
                    ch = changed[seed, op, 1:][:, ids]
                    assert ch.any() and (~ch).any()
                    strata.append((a[ch].mean()+a[~ch].mean())/2)
            values.append(float(np.mean(strata)))
        directions.append(dict(source=seed+1, target=(seed+1)%5+1, values=dict(zip(methods, values))))
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), cells=cells, primary=dict(zip(methods, points.tolist())),
        contrasts=contrasts, directions=directions, runs=run_records, statistics=freeze['statistics'],
        source_valid_fraction=float(valid.mean()), base_accuracy=float(np.mean(base[0] == [p['base_answer'] for p in panel['pairs']])),
        source_full_hybrid={str(arity): float(np.mean([source[s, op, 0, i] == p[['unit_answer','tens_answer'][op]]
            for s in range(5) for op in range(2) for i, p in enumerate(panel['pairs']) if p['arity'] == arity])) for arity in [2, 3]})
    destination.write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(arrays_destination, answers=answers, source=source, base=base,
        correct=correct, changed=changed, methods=np.array(methods), numerator=numerator, denominator=denominator)
    print(json.dumps({k: v for k, v in result.items() if k not in ['cells', 'runs']}))


if __name__ == '__main__':
    main()
