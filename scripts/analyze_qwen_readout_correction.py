import argparse
import copy
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
    destination = args.output or out/'CONFIRMATION_ANALYSIS_V2.json'
    arrays_destination = args.arrays_output or out/'CONFIRMATION_ARRAYS_V2.npz'
    assert not destination.exists() and not arrays_destination.exists()
    freeze = json.loads((out/'READOUT_CORRECTION_FREEZE.json').read_text())
    root = Path(__file__).resolve().parents[1]
    for entry in freeze['code']:
        path = Path(entry['path'])
        relative = path.relative_to(root) if path.is_absolute() else path
        assert hashlib.sha256((args.code_root/relative).read_bytes()).hexdigest() == entry['sha256'], entry['path']
    for entry in freeze['inputs'] + freeze['configurations']:
        assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['path']
    original = json.loads((out/'CONFIRMATION_ANALYSIS.json').read_text())
    result = copy.deepcopy(original)
    panel = json.loads((out/'CONFIRMATION_PANEL.json').read_text())
    first_freeze = json.loads((out/'CONFIRMATION_FREEZE.json').read_text())
    with np.load(out/'CONFIRMATION_ARRAYS.npz') as saved:
        arrays = {k: saved[k] for k in saved.files}
    source, base, changed = (arrays[k] for k in ['source', 'base', 'changed'])
    corrected = np.full_like(arrays['answers'][3], -999)
    new_runs = []
    for seed, entry in enumerate(freeze['configurations']):
        cfg = json.loads(Path(entry['path']).read_text())
        run = Path(cfg['run_storage_root'])/cfg['run_id']
        assert json.loads((run/'status.json').read_text())['status'] == 'PASS'
        assert json.loads((run/'config.resolved.json').read_text()) == cfg
        assert json.loads((run/'panel.json').read_text())['pairs'] == panel['pairs']
        old_run = Path(original['runs'][seed]['run'])
        assert np.array_equal(np.load(run/'queries.npy'), np.load(old_run/'queries.npy'))
        raw = run/'metrics.raw.jsonl'
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        assert digest == json.loads((run/'metrics.summary.json').read_text())['metrics_raw_sha256']
        clean_rows = set()
        for row in map(json.loads, raw.read_text().splitlines()):
            value = -1 if row['answer'] is None else row['answer']
            if row['kind'] == 'base':
                assert value == base[seed, row['row_id']]
                clean_rows.add(row['row_id'])
                continue
            assert row['method'] == 'readout'
            op = ['unit', 'tens'].index(row['operation'])
            query = int(row['mode'].split('_')[-1])
            index = seed, op, query, row['row_id']
            assert corrected[index] == -999
            corrected[index] = value
        assert clean_rows == set(range(96))
        new_runs.append(dict(run=run.as_posix(), raw_sha256=digest))
    assert (corrected != -999).all()
    arrays['answers'][3] = corrected
    arrays['correct'][3] = (corrected == source) & (source >= 10) & (source < 100)
    for cell in result['cells']:
        if cell['method'] != 'readout':
            continue
        op = ['unit', 'tens'].index(cell['operation'])
        ids = [i for i, p in enumerate(panel['pairs']) if p['arity'] == cell['arity']]
        queries = [0] if cell['family'] == 'full' else list(range(1, 9))
        ch = changed[:, op][:, queries][:, :, ids]
        correct = arrays['correct'][3, :, op][:, queries][:, :, ids]
        prediction = corrected[:, op][:, queries][:, :, ids]
        expected = np.array([panel['pairs'][i][['unit_answer', 'tens_answer'][op]] for i in ids])
        changed_agreement = float(correct[ch].mean()) if ch.any() else None
        unchanged_agreement = float(correct[~ch].mean()) if (~ch).any() else None
        balanced = None if changed_agreement is None or unchanged_agreement is None else (changed_agreement+unchanged_agreement)/2
        cell.update(exact_agreement=float(correct.mean()), changed_agreement=changed_agreement,
            unchanged_agreement=unchanged_agreement, balanced_agreement=balanced,
            hybrid_success=float((prediction == expected).mean()))
    numerator, denominator = arrays['numerator'], arrays['denominator']
    numerator[3] = 0
    for row_id, pair in enumerate(panel['pairs']):
        for bank in range(4):
            for op in range(2):
                for query in [2*bank+1, 2*bank+2]:
                    for state in [0, 1]:
                        mask = changed[:, op, query, row_id] == state
                        numerator[3, pair['cluster'], bank, op, state] += arrays['correct'][3, mask, op, query, row_id].sum()
    def estimate(qw, bw):
        estimates = []
        for ids in [slice(0, 24), slice(24, 48)]:
            den = np.einsum('dc,db,cbos->dos', qw[:, ids], bw, denominator[ids])
            assert (den > 0).all()
            num = np.einsum('dc,db,mcbos->mdos', qw[:, ids], bw, numerator[:, ids])
            estimates.append((num/den[None]).mean((2, 3)))
        return np.stack(estimates).mean(0)
    rng = np.random.default_rng(first_freeze['bootstrap_seed'])
    qw = np.concatenate([rng.multinomial(24, np.full(24, 1/24), size=4000) for _ in range(2)], axis=1)
    bw = rng.multinomial(4, np.full(4, 1/4), size=4000)
    draws = estimate(qw, bw)
    points = estimate(np.ones((1, 48)), np.ones((1, 4)))[:, 0]
    methods = arrays['methods'].tolist()
    result['primary'] = dict(zip(methods, points.tolist()))
    result['contrasts'] = [dict(comparator=name, difference=float(points[1]-points[m]),
        interval=np.quantile(draws[1]-draws[m], [.025, .975]).tolist()) for m, name in enumerate(methods) if m != 1]
    for name in methods[:3]:
        assert result['primary'][name] == original['primary'][name]
    for contrast in result['contrasts']:
        if contrast['comparator'] != 'readout':
            assert contrast == next(c for c in original['contrasts'] if c['comparator'] == contrast['comparator'])
    for seed, direction in enumerate(result['directions']):
        strata = []
        for arity in [2, 3]:
            ids = [i for i, p in enumerate(panel['pairs']) if p['arity'] == arity]
            for op in range(2):
                correct = arrays['correct'][3, seed, op, 1:][:, ids]
                ch = changed[seed, op, 1:][:, ids]
                strata.append((correct[ch].mean()+correct[~ch].mean())/2)
        direction['values']['readout'] = float(np.mean(strata))
    for previous, current in zip(original['cells'], result['cells']):
        if current['method'] != 'readout':
            assert current == previous
    result.update(written_at_utc=datetime.now(timezone.utc).isoformat(), readout_runs=new_runs,
        correction='Readout uses its complete fitted 64-member input bank. All other predictions, primary comparisons and corresponding intervals are exactly preserved.')
    destination.write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(arrays_destination, **arrays)
    print(json.dumps(dict(primary=result['primary'], contrasts=result['contrasts'], source_valid_fraction=result['source_valid_fraction'])))


if __name__ == '__main__':
    main()
