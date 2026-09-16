"""Describe finite early-deletion / later-restoration programs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from intervention_prediction_scores import balanced_agreement


def analyze(run):
    index = json.loads((run/'BANK_INDEX.json').read_text())
    values = np.load(run/'responses.npz')['logits'].astype(float)
    assert np.isfinite(values).all()
    methods = index['methods']
    source = values[methods.index('source')]
    clean = values[methods.index('none'), 0]
    lookup = {q['name']: i for i, q in enumerate(index['queries'])}
    profiles = []
    aggregates = {}
    for mi, method in enumerate(methods):
        if method == 'none':
            continue
        restored_source, restored_target, scores = [], [], []
        final_error = []
        for i, query in enumerate(index['queries']):
            if query['role'] != 'restore':
                continue
            j = lookup[query['ablation_reference']]
            deletion = source[j]-clean
            source_change = source[i]-source[j]
            change = values[mi, i]-values[mi, j]
            denom = max(np.dot(deletion, deletion), 1e-15)
            balanced = balanced_agreement(source[i] > 0, values[mi, i] > 0, clean > 0)
            profiles.append(dict(method=method, request=query['name'],
                early=query['early_part'], cut=query['cut'], restored=query['restore_part'],
                deletion_rms=float(np.sqrt(np.mean(deletion**2))),
                source_restore_coefficient=float(-np.dot(deletion, source_change)/denom),
                target_restore_coefficient_in_source_direction=float(-np.dot(deletion, change)/denom),
                restoration_rmse=float(np.sqrt(np.mean((change-source_change)**2))),
                final_logit_rmse=float(np.sqrt(np.mean((values[mi, i]-source[i])**2))),
                **balanced))
            restored_source.append(source_change)
            restored_target.append(change)
            final_error.append(values[mi, i]-source[i])
            if balanced['balanced_agreement'] is not None:
                scores.append(balanced['balanced_agreement'])
        rs, rt = np.array(restored_source), np.array(restored_target)
        aggregates[method] = dict(
            restoration_rmse=float(np.sqrt(np.mean((rt-rs)**2))),
            relative_restoration_rmse=float(np.linalg.norm(rt-rs)/max(np.linalg.norm(rs), 1e-15)),
            final_logit_rmse=float(np.sqrt(np.mean(np.array(final_error)**2))),
            balanced_agreement=float(np.mean(scores)) if scores else None,
            balanced_eligible_requests=len(scores), total_requests=len(rs),
            identity_max_logit_error=max(float(np.max(np.abs(values[mi, i]-clean)))
                for i, q in enumerate(index['queries']) if q['role'] == 'identity_control'))
    return dict(written_at_utc=datetime.now(timezone.utc).isoformat(), run=str(run),
                scope='Specified finite programs, with all other model computations live. Restoration coefficients can exceed1 and do not partition a total effect into unique mediated fractions.',
                methods=methods, documents=len(clean), profiles=profiles, aggregates=aggregates,
                inputs=[dict(path=str(run/name), sha256=hashlib.sha256((run/name).read_bytes()).hexdigest())
                        for name in ['BANK_INDEX.json', 'responses.npz']])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.run)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result['aggregates']))


if __name__ == '__main__':
    main()
