from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BULK = Path('D:/CCAD_Storage/runs/science_upgrade_20260919')
ASSETS = ROOT / 'artifacts/science_upgrade_20260919'


def human():
    seeds = [2, 3, 4, 5]
    runs = [BULK / f'SCIENCE04_shift_t{s}_v1_20260919' for s in seeds]
    panel = json.loads((ASSETS / 'ROUND04_REQUESTS.json').read_text())
    names = panel['queries']
    rows = json.loads((runs[0] / 'evaluation_membership.json').read_text())['rows']
    tasks = ['composer_surgeon_orientation0', 'composer_surgeon_orientation1',
             'model_software_engineer_orientation0', 'model_software_engineer_orientation1']
    heads = [np.load(ROOT / 'runs/IR04_shift_consumer_seed2_v1_20260916' /
                     f'none__full__{t}__probe42.npz')['weight'].ravel() for t in tasks]
    cohorts = [np.array([r['profession'] in p for r in rows])
               for p in [(5, 25), (5, 25), (12, 24), (12, 24)]]

    def project(a):
        return np.stack([a[..., keep, :] @ w for keep, w in zip(cohorts, heads)])

    clean = project(np.load(runs[0] / 'none__full__pooled.npy').astype(float))[:, None]
    source = project(np.stack([np.load(runs[0] / f'source__{q}__pooled.npy') for q in names]).astype(float))
    methods = ['initial', 'head_mixed', 'input_initial', 'tangent_gain', 'tangent_mixed', 'raw_reconstruction']
    values = []
    for method in methods:
        current = []
        for run in runs:
            assert json.loads((run / 'status.json').read_text())['status'] == 'PASS'
            assert json.loads((run / 'evaluation_membership.json').read_text())['rows'] == rows
            current.append(project(np.stack([np.load(run / f'{method}__{q}__pooled.npy') for q in names]).astype(float)))
        values.append(np.stack(current))
        print(f'human loaded {method}', flush=True)
    return dict(target=np.stack(values), source=source, clean=clean,
                methods=methods, seeds=seeds, queries=names,
                families=[panel['families'][q] for q in names],
                runs=list(map(str, runs)), heads=tasks)


def infinitive():
    seeds = [1, 3, 4, 5]
    runs = [BULK / f'SCIENCE04_infinitive_t{s}_v{2 if s == 1 else 1}_20260919' for s in seeds]
    indices = [json.loads((r / 'INDEX.json').read_text()) for r in runs]
    index = indices[0]
    panel = json.loads((ASSETS / 'ROUND04_INFINITIVE_PANEL.json').read_text())
    methods = ['native', 'mixed', 'native_tangent_relation_8', 'tangent_gain', 'tangent_mixed', 'raw_reconstruction']
    values = [dict(np.load(r / 'responses.npz')) for r in runs]
    for run, ind, value in zip(runs, indices, values):
        assert json.loads((run / 'status.json').read_text())['status'] == 'PASS'
        assert ind['rows'] == index['rows'] and ind['query_masks'] == index['query_masks']
        assert np.array_equal(value['source'], values[0]['source'])
        assert np.array_equal(value['none'], values[0]['none'])
    return dict(target=np.stack([np.stack([v[m][None] for v in values]) for m in methods]).astype(float),
                source=values[0]['source'][None].astype(float),
                clean=values[0]['none'][None].astype(float),
                methods=methods, seeds=seeds, queries=index['queries'],
                families=[panel['families'][q] for q in index['queries']],
                runs=list(map(str, runs)), heads=['next_token_to'])


def analyze(data):
    target, source, clean = (data[k] for k in ['target', 'source', 'clean'])
    assert target.ndim == 5 and source.shape == target.shape[2:]
    assert np.isfinite(target).all() and np.isfinite(source).all()
    result = {}
    all_errors = target - source[None, None]
    for mi, method in enumerate(data['methods']):
        error = all_errors[mi]
        result[method] = {}
        for family in ['all', 'held_requests', 'endpoints', 'member_subsets']:
            ix = [i for i, f in enumerate(data['families']) if family == 'all' or
                  (family == 'held_requests' and f in ['interior', 'boundary']) or f == family]
            e = error[:, :, ix]
            source_energy = float(np.mean(((source-clean)[:, ix]) ** 2))
            total = float(np.mean(e ** 2))
            common = float(np.mean(e.mean(0) ** 2))
            variation = float(np.mean((e-e.mean(0)) ** 2))
            assert np.isclose(total, common+variation, rtol=1e-12, atol=1e-12)
            # 留出目标的作用误差由其余目标估计，同一请求和文本的旧响应可用。
            predictions = (e.sum(0)[None]-e)/(len(e)-1)
            prediction_error = float(np.mean((e-predictions) ** 2))
            pair_inner = (len(e)*common-total)/(len(e)-1)
            query_mse = np.mean(e ** 2, axis=3)
            peer_mse = np.mean(predictions ** 2, axis=3)
            peer_variance = np.stack([np.var(np.delete(e, s, axis=0), axis=0).mean(-1) for s in range(len(e))])
            retain = max(1, len(ix)//2)
            pick = np.argsort(peer_mse, axis=-1)[..., :retain]
            spread_pick = np.argsort(peer_variance, axis=-1)[..., :retain]
            result[method][family] = dict(
                source_normalized_rmse=float(np.sqrt(total/source_energy)),
                common_fraction=common/total,
                pair_cross_moment_fraction=pair_inner/total,
                peer_prediction_mse_ratio=prediction_error/total,
                peer_error_screen_mse=float(np.take_along_axis(query_mse, pick, axis=-1).mean()),
                peer_spread_screen_mse=float(np.take_along_axis(query_mse, spread_pick, axis=-1).mean()),
                all_query_mse=float(query_mse.mean()),
                target_error_mse=total, shared_error_mse=common, seed_variance_mse=variation,
                target_leave_one_out_ratios=[float(np.mean((e[s]-predictions[s])**2)/np.mean(e[s]**2)) for s in range(len(e))])
    return result, all_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('setting', choices=['human', 'infinitive'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dest = args.output
    if dest.exists() or dest.with_suffix('.npz').exists():
        raise FileExistsError(dest)
    data = human() if args.setting == 'human' else infinitive()
    result, errors = analyze(data)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest.with_suffix('.npz'), error=errors, source=data['source'], clean=data['clean'])
    metadata = {k: v for k, v in data.items() if k not in ['target', 'source', 'clean']}
    dest.write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        evidence='Development analysis of previously exposed panels; fixed source, methods and four-target pools.',
        information='Other targets have responses for the same queries and contexts. Held target responses are used only for scoring.',
        setting=args.setting, shape=list(errors.shape), summary=result, **metadata,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n', encoding='utf-8')
    print(json.dumps({m: s['held_requests'] for m, s in result.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
