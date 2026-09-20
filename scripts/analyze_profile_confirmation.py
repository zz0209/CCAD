from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/final_science_20260920'
BULK = Path('D:/CCAD_Storage/runs/final_science_20260920')
REPS = 2000
PROFILE = 'initial_refined_source_metric'
TASKS = ['composer_surgeon_orientation0', 'composer_surgeon_orientation1',
         'model_software_engineer_orientation0', 'model_software_engineer_orientation1']


def draws(rng, count):
    return rng.multinomial(count, np.full(count, 1 / count), size=REPS)


def query_order(run, dataset):
    records = json.loads((run/'execution_diagnostics.json').read_text())
    records = [r for r in records if r['dataset']==dataset]
    names = [r['query'] for r in records if r['method']=='none']
    assert names and len(set(names))==len(names)
    for method in {r['method'] for r in records}:
        assert [r['query'] for r in records if r['method']==method]==names
    return names


def families(names, human):
    recorded = json.loads((ART / 'PROFILE_CONFIRMATION_REQUESTS.json').read_text())
    mapping = recorded['human_families' if human else 'grammar_families']
    assert set(mapping) == set(names)
    result = {f: [i for i, n in enumerate(names) if mapping[n] == f] for f in sorted(set(mapping.values()))}
    assert len(result['participation']) == 8
    return result


def infer(nums, den, doc_draws, requests, seeds, rng):
    # 每个方法共用seed、文本和请求的抽样，保留配对。
    sw = draws(rng, len(seeds)) / len(seeds)
    qw = {f: np.tile(ix, (REPS, 1)) if f in ['endpoints', 'center']
          else rng.choice(ix, (REPS, len(ix))) for f, ix in requests.items()}
    summaries, samples = {}, {}
    for method, num in nums.items():
        assert num.shape == (len(seeds), *den.shape)
        pq = np.sqrt(num.mean(-1) / den.mean(-1)[None])
        assert np.isfinite(pq).all()
        b = np.empty((len(seeds), den.shape[0], REPS, den.shape[1]))
        for h, dw in enumerate(doc_draws):
            denominator = dw @ den[h].T
            for s in range(len(seeds)):
                ratio = np.full_like(denominator, np.nan)
                np.divide(dw @ num[s, h].T, denominator, out=ratio, where=denominator > 0)
                b[s, h] = np.sqrt(ratio)
        b = b.mean(1)
        summaries[method] = {}
        for family, indices in requests.items():
            values = np.stack([np.take_along_axis(b[s], qw[family], axis=1).mean(1)
                               for s in range(len(seeds))], axis=1)
            sample = (values * sw).sum(1)
            valid = np.isfinite(sample)
            assert valid.any(), (method, family)
            samples[method, family] = sample
            summaries[method][family] = dict(nrmse=float(pq[:, :, indices].mean()),
                interval=np.quantile(sample[valid], [.025, .975]).tolist(),
                valid_bootstrap_draws=int(valid.sum()), undefined_zero_source_effect_draws=int((~valid).sum()),
                by_seed=pq[:, :, indices].mean((1, 2)).tolist(),
                by_head=pq[:, :, indices].mean((0, 2)).tolist())
    contrasts = []
    for method in nums:
        if method == PROFILE:
            continue
        for family in requests:
            difference = samples[method, family] - samples[PROFILE, family]
            assert np.array_equal(np.isfinite(samples[method, family]), np.isfinite(samples[PROFILE, family]))
            valid = np.isfinite(difference)
            contrasts.append(dict(reference=method, method=PROFILE, family=family,
                reduction=summaries[method][family]['nrmse'] - summaries[PROFILE][family]['nrmse'],
                interval=np.quantile(difference[valid], [.025, .975]).tolist(),
                valid_bootstrap_draws=int(valid.sum())))
    return dict(seeds=seeds, summary=summaries, contrasts=contrasts,
                source_effect_rms=np.sqrt(den.mean(-1)).tolist())


def human_arrays(runs, members):
    arrays = [dict(np.load(r / 'pooled.npz')) for r in runs]
    source, clean = arrays[0]['source'].astype(float), arrays[0]['none'].astype(float)
    for a in arrays[1:]:
        assert np.array_equal(source, a['source']) and np.array_equal(clean, a['none'])
    rows = members['human_rows']
    masks = [np.array([r['profession'] in ps for r in rows]) for ps in [(5, 25)] * 2 + [(12, 24)] * 2]
    weights, biases, paths = [], [], []
    for task in TASKS:
        path = ROOT / 'runs/IR04_shift_consumer_seed2_v1_20260916' / f'none__full__{task}__probe42.npz'
        probe = np.load(path)
        weights.append(probe['weight'].astype(float).ravel())
        biases.append(float(probe['bias'].item()))
        paths.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    den = np.stack([((source[:, mask] - clean[:, mask]) @ weight) ** 2 for mask, weight in zip(masks, weights)])
    assert (den.mean(-1) > 1e-12).all()
    rng = np.random.default_rng(20260920011)
    dw = []
    for h in [0, 2]:
        cohort = [r for r, selected in zip(rows, masks[h]) if selected]
        count = np.zeros((REPS, len(cohort)), dtype=int)
        for profession in sorted({r['profession'] for r in cohort}):
            for gender in [0, 1]:
                ix = [i for i, r in enumerate(cohort) if r['profession'] == profession and r['gender'] == gender]
                assert len(ix) == 16
                count[:, ix] = draws(rng, len(ix))
        dw.extend([count, count])
    nums, decisions = {}, {}
    common = set.intersection(*[set(a) for a in arrays]) - {'none', 'source'}
    for method in sorted(common):
        nums[method] = np.stack([np.stack([((a[method][:, mask].astype(float) - source[:, mask]) @ weight) ** 2
                                          for mask, weight in zip(masks, weights)]) for a in arrays])
        decisions[method] = []
        for a in arrays:
            heads = []
            for h, (mask, weight, bias) in enumerate(zip(masks, weights, biases)):
                target_label = a[method][:, mask].astype(float) @ weight + bias > 0
                source_label = source[:, mask] @ weight + bias > 0
                label = np.array([r['profession'] == (25 if h < 2 else 24) for r, selected in zip(rows, mask) if selected])
                heads.append(dict(source_agreement=float((target_label == source_label).mean()),
                                  task_accuracy=float((target_label == label).mean())))
            decisions[method].append(heads)
    return nums, den, dw, rng, dict(heads=paths, decisions_descriptive=decisions)


def scalar_arrays(runs, members, dataset):
    arrays = [dict(np.load(r / 'responses.npz')) for r in runs]
    prefix = dataset + '__'
    source, clean = arrays[0][prefix + 'source'].astype(float), arrays[0][prefix + 'none'].astype(float)
    for a in arrays[1:]:
        assert np.array_equal(source, a[prefix + 'source']) and np.array_equal(clean, a[prefix + 'none'])
    common = set.intersection(*[set(a) for a in arrays]) - {prefix + 'none', prefix + 'source'}
    nums = {m[len(prefix):]: np.stack([((a[m].astype(float) - source) ** 2)[None] for a in arrays])
            for m in sorted(common) if m.startswith(prefix)}
    den = ((source - clean) ** 2)[None]
    assert (den.mean(-1) > 1e-12).all()
    rng = np.random.default_rng(20260920012 if dataset == 'grammar' else 20260920013)
    rows = members[dataset + '_rows']
    if dataset == 'grammar':
        verbs, nouns = sorted({r['verb'] for r in rows}), sorted({r['noun'] for r in rows})
        vi = [verbs.index(r['verb']) for r in rows]
        ni = [nouns.index(r['noun']) for r in rows]
        dw = [draws(rng, len(verbs))[:, vi] * draws(rng, len(nouns))[:, ni]]
        extra = dict(verbs=verbs, nouns=nouns)
    else:
        count = np.zeros((REPS, len(rows)), dtype=int)
        for profession in sorted({r['profession'] for r in rows}):
            for gender in [0, 1]:
                ix = [i for i, r in enumerate(rows) if r['profession'] == profession and r['gender'] == gender]
                count[:, ix] = draws(rng, len(ix))
        dw, extra = [count], {}
    return nums, den, dw, rng, extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    freeze_path = ART / 'PROFILE_CONFIRMATION_FREEZE.json'
    freeze = json.loads(freeze_path.read_text())
    for item in freeze['inputs']:
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    runs = [BULK / f'PROFILE_CONFIRMATION_T{s}_20260920' for s in args.seeds]
    for run in runs:
        assert json.loads((run / 'status.json').read_text())['status'] == 'PASS', run
        for name, digest in freeze['code_hashes'].items():
            if name in ['scripts/train_intervention_changes.py', 'src/ccad/intervention_transport.py']:
                assert hashlib.sha256((run / 'source_snapshot' / name).read_bytes()).hexdigest() == digest
    members = [json.loads((run / 'membership.json').read_text()) for run in runs]
    assert all(m == members[0] for m in members)
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), runs=list(map(str, runs)),
                  freeze_sha256=hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
                  analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  complete_frozen_cohort=args.seeds == [1, 2, 3, 4, 5], statistics=freeze['statistics'])
    result['array_order_evidence'] = [dict(path=str(run/'execution_diagnostics.json'),
        sha256=hashlib.sha256((run/'execution_diagnostics.json').read_bytes()).hexdigest()) for run in runs]
    result['normalization_note'] = ('A bootstrap draw with exactly zero source effect for a sampled request has undefined nRMSE. '
        'Its whole family draw is omitted jointly for every method, with the count reported. Intervals for an affected '
        'family condition on a nonzero source effect; primary-family validity is reported separately. Original data require positive source energy for every request.')
    for setting in ['human_later_heads', 'grammar', 'human_original_head']:
        human = setting != 'grammar'
        dataset = 'human' if human else 'grammar'
        names = query_order(runs[0], dataset)
        assert set(names)==set(members[0][dataset+'_queries'])
        assert all(query_order(run, dataset)==names for run in runs)
        request_families = families(names, human)
        build = human_arrays if setting == 'human_later_heads' else lambda rr, mm: scalar_arrays(rr, mm, dataset)
        nums, den, dw, rng, extra = build(runs, members[0])
        output = infer(nums, den, dw, request_families, args.seeds, rng)
        output.update(extra, queries=names, request_families=request_families, contexts=len(members[0][dataset + '_rows']))
        result[setting] = output
        trained_seeds = [s for s in args.seeds if s in [2, 3, 4, 5]]
        if trained_seeds:
            trained_runs = [runs[args.seeds.index(s)] for s in trained_seeds]
            ns, dd, ww, rg, _ = build(trained_runs, members[0])
            ns = {m: v for m, v in ns.items() if m in [PROFILE, 'task_adapted_tangent']}
            assert len(ns) == 2
            output['trained_reference'] = infer(ns, dd, ww, request_families, trained_seeds, rg)
        print(setting, {m: round(v['participation']['nrmse'], 6) for m, v in output['summary'].items()})
        print([c for c in output['contrasts'] if c['family'] == 'participation'])
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
