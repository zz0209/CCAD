"""Paired seed/document inference for the frozen restoration query family."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def identity(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    freeze = json.loads(a.freeze.read_text())
    root = Path(__file__).resolve().parents[1]
    panel_path = Path(freeze['panel'].replace('\\', '/'))
    if not panel_path.is_absolute():
        panel_path = root / panel_path
    panel = json.loads(panel_path.read_text())
    by_hash = {r['document_sha256']: r for r in panel['rows']}
    arrays, indices, inputs = [], [], [identity(a.freeze)]
    for seed in freeze['target_seeds']:
        run = root/'runs'/f'MORNING_R05_restore_confirm_seed{seed}_v1_20260916'
        index = json.loads((run/'BANK_INDEX.json').read_text())
        arrays.append(np.load(run/'responses.npz')['logits'].astype(float))
        indices.append(index)
        inputs.extend(identity(run/name) for name in ['BANK_INDEX.json', 'responses.npz'])
    index = indices[0]
    assert all(i['documents'] == index['documents'] and i['queries'] == index['queries']
               and i['methods'] == index['methods'] for i in indices)
    values = np.stack(arrays)
    assert np.isfinite(values).all()
    methods = index['methods']
    si, ni = methods.index('source'), methods.index('none')
    assert np.array_equal(values[:, si], np.broadcast_to(values[0, si], values[:, si].shape))
    assert np.array_equal(values[:, ni], np.broadcast_to(values[0, ni], values[:, ni].shape))
    source, clean = values[0, si], values[0, ni, 0]
    lookup = {q['name']: i for i, q in enumerate(index['queries'])}
    qi = [i for i, q in enumerate(index['queries']) if q['role'] == 'restore']
    aj = [lookup[index['queries'][i]['ablation_reference']] for i in qi]
    queries = [index['queries'][i] for i in qi]
    rs = source[qi]-source[aj]
    ds = source[aj]-clean
    rt = values[:, :, qi]-values[:, :, aj]
    errors = rt-rs
    eligible = [i for i, q in enumerate(queries) if not (q['early_part'] == 'names' and q['cut'] == 0)]
    changed = (source[qi] > 0) != (clean > 0)
    assert all(changed[i].any() and (~changed[i]).any() for i in eligible)
    agreements = (values[:, :, qi] > 0) == (source[qi] > 0)
    n, t, m = len(clean), len(arrays), len(methods)
    strata = [np.array([i for i, key in enumerate(index['documents'])
        if (by_hash[key]['label'], by_hash[key]['gender']) == (y, g)])
        for y in [0, 1] for g in [0, 1]]
    assert [len(s) for s in strata] == [freeze['per_cell']]*4
    rng = np.random.default_rng(freeze['bootstrap_seed'])
    b = 4000
    weights = np.zeros((b, n))
    for stratum in strata:
        weights[:, stratum] = rng.multinomial(len(stratum), np.full(len(stratum), 1/len(stratum)), size=b)
    seed_draws = rng.integers(0, t, size=(b, t))
    seed_counts = np.stack([(seed_draws == j).sum(1) for j in range(t)], 1)/t

    def weighted(x):
        # Inputs [target,method,document], output [draw,method].
        per_seed = (x.reshape(t*m, n) @ weights.T/n).reshape(t, m, b).transpose(2, 0, 1)
        return np.einsum('bt,btm->bm', seed_counts, per_seed)

    rel = np.sqrt(np.mean(errors**2, axis=(0, 2, 3))/np.mean(rs**2))
    mse_boot = weighted(np.mean(errors**2, axis=2))
    source_energy_boot = np.mean(rs**2, axis=0) @ weights.T/n
    rel_boot = np.sqrt(mse_boot/source_energy_boot[:, None])
    final = values[:, :, qi]-source[qi]
    final_rmse = np.sqrt(np.mean(final**2, axis=(0, 2, 3)))
    final_boot = np.sqrt(weighted(np.mean(final**2, axis=2)))
    coefficients = -np.mean(ds[None, None]*rt, axis=3)/np.mean(ds**2, axis=1)
    coefficient_boot = np.empty((b, m, len(qi)))
    for q in range(len(qi)):
        den = ds[q]**2 @ weights.T/n
        coefficient_boot[:, :, q] = weighted(-ds[q]*rt[:, :, q])/den[:, None]
    source_coefficients = -np.mean(ds*rs, axis=1)/np.mean(ds**2, axis=1)
    scores, score_boot = [], []
    for q in eligible:
        per_q, boot_q = [], []
        for c in [False, True]:
            mask = changed[q] == c
            per_q.append(agreements[:, :, q, mask].mean(axis=(0, 2)))
            numer = weighted(agreements[:, :, q]*mask)
            denominator = mask.astype(float) @ weights.T/n
            boot_q.append(np.divide(numer, denominator[:, None],
                out=np.full_like(numer, np.nan), where=denominator[:, None] > 0))
        scores.append(np.mean(per_q, axis=0))
        score_boot.append(np.mean(boot_q, axis=0))
    balanced = np.mean(scores, axis=0)
    balanced_boot = np.mean(score_boot, axis=0)

    def stat(value, draws):
        valid = np.isfinite(draws)
        return dict(value=float(value), ci95=np.quantile(draws[valid], [.025, .975]).tolist(),
                    valid_draws=int(valid.sum()), undefined_draws=int((~valid).sum()))

    word_q = next(i for i, q in enumerate(queries) if q['early_part'] == 'names' and q['cut'] == 2 and q['restore_part'] == 'associated_words')
    name_q = next(i for i, q in enumerate(queries) if q['early_part'] == 'names' and q['cut'] == 2 and q['restore_part'] == 'names')
    summary = {}
    for mi, method in enumerate(methods):
        if method == 'none':
            continue
        summary[method] = dict(relative_restoration_rmse=stat(rel[mi], rel_boot[:, mi]),
            final_logit_rmse=stat(final_rmse[mi], final_boot[:, mi]),
            balanced_agreement=stat(balanced[mi], balanced_boot[:, mi]),
            late_words_minus_names_after_early_names=stat(
                coefficients[:, mi, word_q].mean()-coefficients[:, mi, name_q].mean(),
                coefficient_boot[:, mi, word_q]-coefficient_boot[:, mi, name_q]),
            identity_max_logit_error=max(float(np.max(np.abs(values[:, mi, i]-clean)))
                for i, q in enumerate(index['queries']) if q['role'] == 'identity_control'))
    native = methods.index('native')
    contrasts = {}
    for method in ['geometry_gain', 'geometry', 'raw']:
        mi = methods.index(method)
        contrasts['native_minus_'+method] = dict(
            relative_restoration_rmse=stat(rel[native]-rel[mi], rel_boot[:, native]-rel_boot[:, mi]),
            balanced_agreement_points=stat(100*(balanced[native]-balanced[mi]), 100*(balanced_boot[:, native]-balanced_boot[:, mi])))
    profiles = []
    for q, query in enumerate(queries):
        item = dict(request=query['name'], early=query['early_part'], cut=query['cut'],
            restored=query['restore_part'], source_changed=int(changed[q].sum()),
            source_unchanged=int((~changed[q]).sum()), balanced_in_fixed_family=q in eligible,
            source_coefficient=float(source_coefficients[q]), methods={})
        for mi, method in enumerate(methods):
            if method != 'none':
                item['methods'][method] = stat(coefficients[:, mi, q].mean(), coefficient_boot[:, mi, q])
        profiles.append(item)
    out = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        scope=freeze['statistics'], documents=n, targets=freeze['target_seeds'],
        restoration_requests=len(qi), balanced_requests=len(eligible),
        source_and_none_exact_across_targets=True, summary=summary,
        contrasts=contrasts, profiles=profiles, inputs=inputs,
        analysis=identity(Path(__file__)))
    with a.output.open('x') as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(json.dumps(dict(summary=summary, contrasts=contrasts), indent=2))


if __name__ == '__main__':
    main()
