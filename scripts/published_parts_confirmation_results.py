"""Analyze the frozen independent explanation panel with crossed resampling."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/independent_reuse_20260916'


def identity(p):
    return dict(path=p.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    runs = [ROOT/f'runs/IR01_infinitive_confirm_t{s}_v1_20260916' for s in [2, 3, 4, 5]]
    all_x, inventory, writes, corrections, supplements = [], [], [], [], []
    first = None
    for run in runs:
        index = json.loads((run/'INDEX.json').read_text())
        if first is None:
            first = index
        assert index == first
        state = json.loads((run/'status.json').read_text())
        assert state['status'] == 'PASS'
        with np.load(run/'responses.npz') as data:
            current = data['log_probability'].copy()
        corrected = run.with_name(run.name.replace('_v1_', '_v2_'))
        corrected_index = json.loads((corrected/'INDEX.json').read_text())
        corrected_status = json.loads((corrected/'status.json').read_text())
        assert corrected_status['status'] == 'PASS'
        assert all(corrected_index[k] == index[k] for k in ['queries', 'query_masks', 'rows', 'answer_id'])
        with np.load(corrected/'responses.npz') as data:
            fixed = data['log_probability']
            drift = {}
            for ci, method in enumerate(corrected_index['methods']):
                mi = index['methods'].index(method)
                drift[method] = float(np.max(np.abs(fixed[ci]-current[mi])))
                if method in ['none', 'source']:
                    assert np.array_equal(fixed[ci], current[mi])
                current[mi] = fixed[ci]
        supplement = run.with_name(run.name.replace('_v1_', '_v3_'))
        sup_index = json.loads((supplement/'INDEX.json').read_text())
        sup_status = json.loads((supplement/'status.json').read_text())
        assert sup_status['status'] == 'PASS'
        assert all(sup_index[k] == index[k] for k in ['queries', 'query_masks', 'rows', 'answer_id'])
        assert sup_index['methods'][2:] == ['native_response_gain', 'native_response_gain_8']
        with np.load(supplement/'responses.npz') as data:
            extra = data['log_probability']
            assert np.array_equal(extra[:2], current[:2])
            current = np.concatenate([current, extra[2:]], axis=0)
        all_x.append(current)
        supplements.append(dict(run=supplement.name, status=sup_status,
            metrics=json.loads((supplement/'metrics.summary.json').read_text()),
            inputs=[identity(supplement/p) for p in ['responses.npz', 'INDEX.json', 'code_hashes.json', 'response_gain.npz', 'RESPONSE_GAIN_FIT.json']]))
        corrections.append(dict(run=corrected.name, status=corrected_status,
            maximum_log_probability_change=drift,
            metrics=json.loads((corrected/'metrics.summary.json').read_text()),
            inputs=[identity(corrected/p) for p in ['responses.npz', 'INDEX.json', 'code_hashes.json']]))
        inventory.append(dict(run=run.name, status=state,
            metrics=json.loads((run/'metrics.summary.json').read_text()),
            inputs=[identity(run/p) for p in ['responses.npz', 'INDEX.json', 'config.resolved.json', 'code_hashes.json']]))
        records = [r for r in json.loads((run/'WRITING.json').read_text())['records'] if r['method'] not in corrected_index['methods']]
        records += json.loads((corrected/'WRITING.json').read_text())['records']
        records += json.loads((supplement/'WRITING.json').read_text())['records']
        for method in first['methods']+sup_index['methods'][2:]:
            r = [x for x in records if x['method'] == method]
            if r:
                writes.append(dict(run=run.name, method=method,
                    mean_changed=sum(x['changed_sum'] for x in r)/sum(x['states'] for x in r),
                    minimum_final_code=min(x['minimum_final_code'] for x in r)))
    x = np.stack(all_x)
    methods = first['methods']+['native_response_gain', 'native_response_gain_8']
    queries, rows = first['queries'], first['rows']
    assert x.shape == (4, len(methods), 15, 128)
    for method in ['none', 'source']:
        assert all(np.array_equal(x[t, methods.index(method)], x[0, methods.index(method)]) for t in range(1, 4))
    effects = x[:, methods.index('none'):methods.index('none')+1]-x
    source = effects[:, methods.index('source')]
    errors = effects-source[:, None]
    # Preserve the fixed row order: verb x noun x form. Query/form repeats are
    # averaged within each cell before crossed lexical and target resampling.
    verbs = list(dict.fromkeys(r['verb'] for r in rows))
    nouns = list(dict.fromkeys(r['noun'] for r in rows))
    assert [(r['verb'], r['noun'], r['form']) for r in rows] == [(v, n, f) for v in verbs for n in nouns for f in range(4)]
    esq = (errors[:, :, 3:]**2).reshape(4, len(methods), 12, 8, 4, 4).mean(axis=(2, 5))
    ssq = (source[:, 3:]**2).reshape(4, 12, 8, 4, 4).mean(axis=(1, 4))
    point = np.sqrt(esq.mean(axis=(0, 2, 3))/ssq.mean())
    rng = np.random.default_rng(20260916)
    bootstrap = []
    for _ in range(4000):
        a = np.bincount(rng.integers(4, size=4), minlength=4)
        b = np.bincount(rng.integers(8, size=8), minlength=8)
        c = np.bincount(rng.integers(4, size=4), minlength=4)
        weight = a[:, None, None]*b[None, :, None]*c[None, None, :]
        denominator = np.sum(weight*ssq)
        bootstrap.append(np.sqrt(np.einsum('tvn,tmvn->m', weight, esq)/denominator))
    bootstrap = np.stack(bootstrap)
    cells = {}
    roles = ['predicate', 'object']
    for mi, method in enumerate(methods):
        profile = [[float(effects[:, mi, qi, [r['role'] == role for r in rows]].mean()) for role in roles] for qi in range(3)]
        cells[method] = dict(new_request_nrmse=float(point[mi]),
            ci95=np.quantile(bootstrap[:, mi], [.025, .975]).tolist(),
            target_nrmse=np.sqrt(esq[:, mi].mean(axis=(1, 2))/ssq.mean(axis=(1, 2))).tolist(),
            role_profile=profile,
            all_request_nrmse=float(np.sqrt(np.mean(errors[:, mi]**2)/np.mean(source**2))))
    contrasts = []
    for proposed, controls in [
        ('native_response_relation', ['native_tangent_relation', 'native', 'geometry_gain', 'geometry', 'raw_reconstruction', 'native_reencode', 'native_response_gain']),
        ('native_response_relation_8', ['native_tangent_relation_8', 'native', 'native_response_gain_8']),
        ('native_tangent_relation_8', ['native', 'native_response_gain_8'])]:
        p = methods.index(proposed)
        for control in controls:
            c = methods.index(control)
            contrasts.append(dict(proposed=proposed, control=control,
                difference=float(point[p]-point[c]),
                ci95=np.quantile(bootstrap[:, p]-bootstrap[:, c], [.025, .975]).tolist()))
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        evidence_level='frozen confirmation on authored new lexical/form contexts and 12 unused member subsets',
        freeze=identity(OUT/'IR01_CONFIRMATION_FREEZE.json'), analyzer=identity(Path(__file__).resolve()),
        methods=methods, queries=queries, primary_queries=queries[3:], context_roles=roles,
        cases=128, targets=4, verbs=verbs, nouns=nouns,
        statistics='4000 crossed target/verb/noun bootstrap draws; query/form repeats paired; fixed source and templates.',
        source_and_clean_replays_bitwise_equal=True, source_effect_rms=float(np.sqrt(ssq.mean())),
        results=cells, contrasts=contrasts, writing=writes, inventory=inventory, numerical_corrections=corrections,
        supplementary_same_source_control=supplements,
        supplementary_scope='Fixed response directions with exact live source amplitudes, added after original confirmation; fits use original natural states, never this panel.')
    (OUT/'IR01_CONFIRMATION_ANALYSIS.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(OUT/'IR01_CONFIRMATION_ARRAYS.npz', effects=effects, nrmse_bootstrap=bootstrap)
    print(json.dumps(dict(results=cells, contrasts=contrasts), indent=2))


if __name__ == '__main__':
    main()
