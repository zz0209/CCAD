"""Score saved intervention answers for one input cohort, without a model.

Arrays: clean[N], source_full[N], target_full[M,N], source_parts[P,N],
target_parts[M,P,N]. Full/part predictions must use the same input order.
The optional eligible[N] selects valid source cases before the common check.
Point estimates are returned; sampling units determine uncertainty separately.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def balanced_agreement(source, target, clean):
    """Equation (2): equal weight on changed and unchanged source answers.

    Each argument has the same shape. A missing stratum gives an undefined
    score (None), retaining its count rather than silently changing the metric.
    """
    source, target, clean = map(np.asarray, (source, target, clean))
    if source.shape != target.shape or source.shape != clean.shape:
        raise ValueError('source, target and clean must have identical shapes')
    changed = source != clean
    agree = source == target
    counts = [int((~changed).sum()), int(changed.sum())]
    score = float(.5 * (agree[~changed].mean() + agree[changed].mean())) if all(counts) else None
    return dict(balanced_agreement=score, unchanged=counts[0], changed=counts[1])


def score_predictions(clean, source_full, target_full, source_parts,
                      target_parts, eligible=None):
    """Full and per-part fidelity, plus part errors on a shared full-match set.

    Uses every supplied method to define the common set. Keep that method set
    fixed when comparing results. Part scores are averaged equally within this
    cohort; return each part separately to support another declared estimand.
    """
    clean, source_full, target_full, source_parts, target_parts = map(
        np.asarray, (clean, source_full, target_full, source_parts, target_parts))
    n = clean.size
    if clean.shape != (n,) or source_full.shape != (n,):
        raise ValueError('clean and source_full must be vectors of the same size')
    if target_full.ndim != 2 or target_full.shape[1] != n:
        raise ValueError('target_full must have shape [methods, cases]')
    if source_parts.ndim != 2 or source_parts.shape[1] != n:
        raise ValueError('source_parts must have shape [parts, cases]')
    m, p = target_full.shape[0], source_parts.shape[0]
    if min(n, m, p) == 0 or target_parts.shape != (m, p, n):
        raise ValueError('nonempty target_parts must have shape [methods, parts, cases]')
    valid = np.ones(n, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    if valid.shape != (n,):
        raise ValueError('eligible must have one entry per case')
    gate = valid & (target_full == source_full).all(axis=0)
    methods = []
    for mi in range(m):
        full = balanced_agreement(source_full[valid], target_full[mi, valid], clean[valid])
        parts = [balanced_agreement(source_parts[pi, valid], target_parts[mi, pi, valid],
                                    clean[valid]) for pi in range(p)]
        scores = [r['balanced_agreement'] for r in parts]
        wrong = (target_parts[mi] != source_parts).any(axis=0)
        errors = int((wrong & gate).sum())
        methods.append(dict(full=full, parts=parts,
            mean_part_balanced_agreement=float(np.mean(scores)) if all(v is not None for v in scores) else None,
            common_full_part_errors=errors,
            common_full_part_error_rate=errors / int(gate.sum()) if gate.any() else None))
    return dict(cases=n, eligible_cases=int(valid.sum()), parts=p,
                common_full_cases=int(gate.sum()), methods=methods)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    with np.load(args.input, allow_pickle=False) as z:
        data = {k: z[k] for k in ['clean', 'source_full', 'target_full', 'source_parts', 'target_parts']}
        if 'eligible' in z:
            data['eligible'] = z['eligible']
        result = score_predictions(**data)
        result['method_names'] = z['method_names'].tolist() if 'method_names' in z else list(range(len(result['methods'])))
        result['part_names'] = z['part_names'].tolist() if 'part_names' in z else list(range(result['parts']))
    result['aggregation'] = 'Single cohort; equal mean across specified parts. No independence or interval claim.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), cases=result['cases'], common_full_cases=result['common_full_cases'])))


if __name__ == '__main__':
    main()
