import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = np.load(args.original/'relation.npz')
    sites = [key[:-8] for key in original.files if key.endswith('__native')]
    reports = []
    for run in args.runs:
        cfg = json.loads((run/'config.resolved.json').read_text())
        membership = json.loads((run/'FIT_MEMBERSHIP.json').read_text())
        rows = [json.loads(line) for line in (run/'metrics.raw.jsonl').read_text().splitlines()]
        assert set(membership['training_requests']).isdisjoint(membership['held_requests'])
        assert len(membership['edge_order']) == cfg['steps']
        for variant in cfg['variants']:
            metrics = [row for row in rows if row['method'] == variant]
            assert max(row['step'] for row in metrics) == cfg['steps']
            assert all(np.isfinite(row['loss']) for row in metrics)
            path = run/variant/'relation.npz'
            relation = np.load(path)
            counts = {}
            for site in sites:
                matrix = relation[site+'__native']
                previous = original[site+'__native']
                assert matrix.shape == previous.shape and np.isfinite(matrix).all()
                assert matrix.min() >= 0 and matrix.sum(1).max() <= 1.00001
                active = np.flatnonzero(matrix.sum(1) > 0)
                old_active = np.flatnonzero(previous.sum(1) > 0)
                candidates = original[site+'__candidates']
                assert set(active).issubset(candidates.tolist())
                assert len(active) <= len(old_active)
                if not cfg.get('reselect_step'):
                    assert set(active).issubset(old_active.tolist())
                for key in ['geometry', 'geometry_gain', 'raw', 'candidates']:
                    assert np.array_equal(relation[site+'__'+key], original[site+'__'+key])
                counts[site] = dict(nonzero_rows=len(active), allowance=len(old_active))
            reports.append(dict(run=str(run), variant=variant, steps=cfg['steps'],
                training_documents=len(membership['documents']),
                training_pairs=len(membership['training_requests']), held_pairs=len(membership['held_requests']),
                member_counts=counts, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        validation='Saved coefficients, training records, pair disjointness, row capacity and unchanged comparator arrays.',
        runs=reports)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(checked_relations=len(reports), output=str(args.output))))


if __name__ == '__main__':
    main()
