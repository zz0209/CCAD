"""Describe the retained independent-circuit development, without a new claim."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/morning_reform_20260916'
RUNS = ['MORNING_R07_infinitive_source_dev_v1_20260916',
        'MORNING_R07_infinitive_source_dev_v2_20260916',
        'MORNING_R07_infinitive_target_dev_v1_20260916',
        'MORNING_R07_infinitive_response_dev_v1_20260916']


def identity(path):
    return dict(path=path.relative_to(ROOT).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    records, inventory = [], []
    baseline = None
    for name in RUNS:
        folder = ROOT / 'runs' / name
        index = json.loads((folder / 'INDEX.json').read_text())
        with np.load(folder / 'responses.npz') as values:
            x = values['log_probability'].copy()
            source_arrays = {k: values[k][:2].copy() for k in values.files}
        if baseline is None:
            baseline = source_arrays
        assert all(np.array_equal(v, baseline[k]) for k, v in source_arrays.items())
        effects = x[0] - x
        source = effects[index['methods'].index('source')]
        denom = np.sqrt(np.mean(source ** 2))
        roles = [np.array([r['role'] == role for r in index['rows']])
                 for role in ['predicate', 'object']]
        results = {}
        for mi, method in enumerate(index['methods']):
            profile = np.array([effects[mi, :, mask].mean(axis=0) for mask in roles]).T
            # Each row selects a query; columns select the two context roles.
            assert profile.shape == (3, 2)
            results[method] = dict(log_probability_decrease_by_query_and_role=profile.tolist(),
                response_nrmse=float(np.sqrt(np.mean((effects[mi]-source)**2))/denom),
                role_selectivity=float((profile[1, 0]-profile[2, 0]+profile[2, 1]-profile[1, 1])/2))
        records.append(dict(run=name, inputs=[identity(folder / p) for p in ['responses.npz', 'INDEX.json']],
                            queries=index['queries'], context_roles=['predicate', 'object'], results=results))
        status = json.loads((folder / 'status.json').read_text())
        summary_path = folder / 'metrics.summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else json.loads((folder / 'progress.json').read_text())
        inventory.append(dict(run=name, status=status,
            driver_seconds=summary.get('wall_seconds', summary.get('elapsed')),
            timing_scope='complete driver' if summary_path.exists() else 'last recorded model progress; finalizer exit time unknown',
            sequences=summary['sequence_forwards'], tokens=summary['token_forwards'],
            peak_allocated_bytes=summary.get('peak_allocated_bytes')))
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), analyzer=identity(Path(__file__).resolve()),
        evidence_level='controlled development; no independent confirmation', source_replays_bitwise_equal=True,
        cases=128, lexical_pairs=64, records=records,
        inference='Descriptive paired context effects only. Eight verbs and eight nouns recur; no independent-prompt confidence claim.',
        decision='Source distinction is useful; target field and source-response fits preserve too little effect to warrant an added positive transfer claim. Retain as a stopped candidate outside the manuscript.')
    (OUT / 'R07_ANALYSIS.json').write_text(json.dumps(result, indent=2)+'\n')
    (OUT / 'R07_RUN_INVENTORY.json').write_text(json.dumps(dict(written_at_utc=result['written_at_utc'], runs=inventory), indent=2)+'\n')
    print(json.dumps(dict(source_replays_bitwise_equal=True, response_fit=records[-1]['results'], runs=inventory), indent=2))


if __name__ == '__main__':
    main()
