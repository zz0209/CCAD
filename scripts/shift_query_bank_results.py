"""Fixed-budget prediction of held-out requests from a saved source bank."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def pivots(matrix, count):
    residual = matrix.copy()
    selected = []
    for _ in range(count):
        norm = np.sum(residual**2, axis=0)
        norm[selected] = -1
        j = int(np.argmax(norm))
        selected.append(j)
        direction = residual[:, j]
        length = np.linalg.norm(direction)
        if length > 1e-12:
            direction = direction/length
            residual -= direction[:, None]*(direction @ residual)[None, :]
    return selected


def coefficients(basis, requested):
    gram = basis.T @ basis
    ridge = 1e-3*max(float(np.trace(gram)/len(gram)), 1e-12)
    return np.linalg.solve(gram+ridge*np.eye(len(gram)), basis.T @ requested)


def balanced(target, source, clean):
    changed = (source > 0) != (clean > 0)
    same = (target > 0) == (source > 0)
    counts = np.array([changed.sum(-1), (~changed).sum(-1)])
    # Missing strata are reported; they never receive a fabricated score.
    value = np.full(source.shape[0], np.nan)
    valid = (counts > 0).all(0)
    for i in np.flatnonzero(valid):
        value[i] = .5*(same[i, changed[i]].mean()+same[i, ~changed[i]].mean())
    return value, counts


def analyze(run):
    index = json.loads((run/'BANK_INDEX.json').read_text())
    data = np.load(run/'responses.npz')['logits'].astype(np.float64)
    assert np.isfinite(data).all()
    methods = index['methods']
    target_names = [v for v in methods if v not in ('source', 'none')]
    clean = data[methods.index('none'), 0]
    source = data[methods.index('source')]
    source_effect = source-clean[None, :]
    targets = data[[methods.index(v) for v in target_names]]
    cal = np.array(index['context_split']) == 'calibration'
    ev = ~cal
    anchors = np.array([i for i,q in enumerate(index['queries']) if q['role'] == 'anchor'])
    heldout = np.array([i for i,q in enumerate(index['queries']) if q['role'] == 'evaluation'])
    features = source_effect[:, cal].T
    masks = np.array([sum(q['weights'].values(), []) for q in index['queries']]).T
    actual = np.array([balanced(t[heldout][:, ev], source[heldout][:, ev], clean[ev])[0]
                       for t in targets])
    counts = balanced(source[heldout][:, ev], source[heldout][:, ev], clean[ev])[1]
    valid = np.isfinite(actual).all(0)
    singular = np.linalg.svd(features[:, anchors], compute_uv=False)
    energy = np.cumsum(singular**2)/np.sum(singular**2)
    details = []
    for budget in [1, 2, 4, 8]:
        selections = [('source_diversity', 0, anchors[pivots(features[:, anchors], budget)], features),
                      ('source_magnitude', 0, anchors[np.argsort(-np.sum(features[:, anchors]**2, axis=0),
                                                                  kind='stable')[:budget]], features),
                      ('mask_diversity', 0, anchors[pivots(masks[:, anchors], budget)], masks)]
        for seed in range(32):
            selections.append(('random', seed,
                               np.random.default_rng(seed).choice(anchors, budget, replace=False), features))
        for selector, seed, selected, predictor_features in selections:
            co = coefficients(predictor_features[:, selected], predictor_features[:, heldout])
            predicted = np.array([source[heldout][:, ev]+
                                 co.T @ (t[selected][:, ev]-source[selected][:, ev])
                                 for t in targets])
            predicted_b = np.array([balanced(t, source[heldout][:, ev], clean[ev])[0]
                                    for t in predicted])
            observed_b = np.array([balanced(t[selected][:, cal], source[selected][:, cal], clean[cal])[0]
                                  for t in targets])
            eligible_anchors = np.isfinite(observed_b).all(0)
            if eligible_anchors.any():
                fixed_scores = observed_b[:, eligible_anchors].mean(1)
                fixed_scoring = 'balanced_on_eligible_anchors'
            else:
                fixed_scores = np.mean((targets[:, selected][:, :, cal] > 0) ==
                                       (source[selected][:, cal][None, :, :] > 0), axis=(1, 2))
                fixed_scoring = 'ordinary_agreement_no_eligible_balanced_anchor'
            assert np.isfinite(fixed_scores).all()
            fixed_choice = int(np.argmax(fixed_scores))
            choice = np.argmax(predicted_b[:, valid], axis=0)
            chosen_true = actual[:, valid][choice, np.arange(valid.sum())]
            residual = predicted-targets[:, heldout][:, :, ev]
            details.append(dict(selector=selector, random_seed=seed, budget=budget,
                anchors=[index['queries'][i]['name'] for i in selected],
                rmse_by_executor=dict(zip(target_names, np.sqrt(np.mean(residual**2, axis=(1, 2))).tolist())),
                agreement_mae_by_executor=dict(zip(target_names, np.mean(np.abs(predicted_b[:, valid]-actual[:, valid]), axis=1).tolist())),
                selected_agreement=float(chosen_true.mean()),
                best_fixed_from_anchors=target_names[fixed_choice],
                fixed_scoring=fixed_scoring, balanced_eligible_anchors=int(eligible_anchors.sum()),
                predicted_agreement_by_query=predicted_b.tolist(),
                fixed_agreement=float(actual[fixed_choice, valid].mean()),
                oracle_agreement=float(actual[:, valid].max(0).mean()),
                selected_executors=[target_names[i] for i in choice]))
    rows = []
    for budget in [1, 2, 4, 8]:
        for selector in ['source_diversity', 'source_magnitude', 'mask_diversity', 'random']:
            group = [v for v in details if v['budget'] == budget and v['selector'] == selector]
            rows.append(dict(budget=budget, selector=selector, repeats=len(group),
                rmse_by_executor={n: float(np.mean([v['rmse_by_executor'][n] for v in group])) for n in target_names},
                agreement_mae_by_executor={n: float(np.mean([v['agreement_mae_by_executor'][n] for v in group])) for n in target_names},
                selected_agreement=float(np.mean([v['selected_agreement'] for v in group])),
                fixed_agreement=float(np.mean([v['fixed_agreement'] for v in group])),
                oracle_agreement=group[0]['oracle_agreement']))
    copy_residual = source[heldout][:, ev][None, :, :]-targets[:, heldout][:, :, ev]
    return dict(written_at_utc=datetime.now(timezone.utc).isoformat(), run=str(run),
        scope='Development only: exposed biographies, new fixed query masks; target anchor observations on evaluation inputs count toward the budget.',
        target_names=target_names, contexts=dict(calibration=int(cal.sum()), evaluation=int(ev.sum())),
        requests=dict(anchors=len(anchors), heldout=len(heldout), balanced_eligible=int(valid.sum()),
                      changed_unchanged_counts=counts.tolist()),
        source_spectrum=dict(singular_values=singular.tolist(), cumulative_energy=energy.tolist()),
        source_copy_rmse=dict(zip(target_names, np.sqrt(np.mean(copy_residual**2, axis=(1, 2))).tolist())),
        actual_agreement_by_executor=dict(zip(target_names, np.mean(actual[:, valid], axis=1).tolist())),
        actual_agreement_by_query=actual.tolist(),
        heldout_query_names=[index['queries'][i]['name'] for i in heldout],
        summaries=rows, all_draws=details,
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
    print(json.dumps({k:v for k,v in result.items() if k in
                     ['contexts', 'requests', 'source_copy_rmse', 'actual_agreement_by_executor', 'summaries']}))


if __name__ == '__main__':
    main()
