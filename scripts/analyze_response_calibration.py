from pathlib import Path
from datetime import datetime, timezone
import argparse
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/final_science_20260921_round04'
OLD = ROOT / 'artifacts/science_upgrade_20260919'


def query_coordinates(setting, names):
    if setting == 'human':
        panel = json.loads((OLD / 'ROUND04_REQUESTS.json').read_text())
        groups = ['pronouns', 'names', 'associated_words']
        return np.array([[panel['dose_queries'][q][g] for g in groups] if q in panel['dose_queries']
                         else [float(q == 'full' or g in q.split('+')) for g in groups]
                         for q in names])
    panel = json.loads((OLD / 'ROUND04_INFINITIVE_PANEL.json').read_text())
    return np.array([[panel['queries'][q][0], panel['queries'][q][-1]] for q in names])


def layout(x, budget):
    d = np.sum((x[:, None]-x[None])**2, axis=-1)
    chosen = []
    nearest = np.full(len(x), np.inf)
    for _ in range(budget):
        i = min((j for j in range(len(x)) if j not in chosen),
                key=lambda j: (np.minimum(nearest, d[:, j]).max(), np.minimum(nearest, d[:, j]).mean(), j))
        chosen.append(i)
        nearest = np.minimum(nearest, d[:, i])
    return np.array(chosen)


def features(effect, x, kind):
    coordinates = np.broadcast_to(x[:, None, :], (*effect.shape, x.shape[-1]))
    polynomial = [coordinates, coordinates**2]
    polynomial += [(coordinates[..., i]*coordinates[..., j])[..., None]
                   for i in range(x.shape[-1]) for j in range(i)]
    poly = np.concatenate(polynomial, axis=-1)
    if kind == 'direct':
        return poly
    if kind == 'gain':
        return effect[..., None]
    return np.concatenate([effect[..., None], poly, effect[..., None]*coordinates], axis=-1)


def fit_predict(a, y, b):
    a = a.reshape(-1, a.shape[-1])
    scale = np.sqrt(np.mean(a*a, axis=0)).clip(1e-8)
    a = a/scale
    coefficient = np.linalg.solve(a.T@a+1e-2*len(a)*np.eye(a.shape[-1]), a.T@y.ravel())
    return (b/scale)@coefficient


def main():
    p = argparse.ArgumentParser()
    p.add_argument('setting', choices=['human', 'infinitive'])
    args = p.parse_args()
    dest = OUT / f'RESPONSE_CALIBRATION_{args.setting.upper()}_DEV.json'
    if dest.exists():
        raise FileExistsError(dest)
    path = OUT / f'SHARED_ERROR_{args.setting.upper()}_DEV'
    metadata = json.loads(path.with_suffix('.json').read_text())
    data = np.load(path.with_suffix('.npz'))
    keep = np.array([i for i, f in enumerate(metadata['families']) if f != 'member_subsets'])
    names = [metadata['queries'][i] for i in keep]
    x = query_coordinates(args.setting, names)
    error = data['error'][:, :, :, keep]
    effect = (data['source']-data['clean'])[:, keep]
    # 人工文本按独立文档划分；语法按verb整体划分，保留配对形式。
    if args.setting == 'human':
        cal = np.arange(0, error.shape[-1], 2)
        test = np.arange(1, error.shape[-1], 2)
    else:
        rows = json.loads((OLD / 'ROUND04_INFINITIVE_PANEL.json').read_text())['rows']
        verbs = sorted({r['verb'] for r in rows})
        cal = np.array([i for i, r in enumerate(rows) if r['verb'] in verbs[::2]])
        test = np.array([i for i, r in enumerate(rows) if r['verb'] in verbs[1::2]])
    records = []
    for budget in [1, 3, 7]:
        selected = layout(x, budget)
        evaluation = np.array([i for i in range(len(x)) if i not in selected])
        actual = (effect[None, None]+error)[:, :, :, evaluation][..., test]
        denominator = float(np.mean(effect[:, evaluation][..., test]**2))
        for mi, method in enumerate(metadata['methods']):
            predictions = {key: [] for key in ['source', 'gain', 'direct', 'residual']}
            for si in range(error.shape[1]):
                by_head = {key: [] for key in predictions}
                for hi in range(error.shape[2]):
                    base = effect[hi]
                    observed_error = error[mi, si, hi][selected][:, cal]
                    by_head['source'].append(base[evaluation][:, test])
                    for kind in ['gain', 'direct', 'residual']:
                        f = features(base, x, kind)
                        response = observed_error if kind == 'residual' else observed_error+base[selected][:, cal]
                        pred = fit_predict(f[selected][:, cal], response, f[evaluation][:, test])
                        if kind == 'residual':
                            pred = pred+base[evaluation][:, test]
                        by_head[kind].append(pred)
                for key in predictions:
                    predictions[key].append(np.stack(by_head[key]))
            risk = {k: float(np.sqrt(np.mean((np.stack(v)-actual[mi])**2)/denominator))
                    for k, v in predictions.items()}
            records.append(dict(method=method, budget=budget, calibration_queries=[names[i] for i in selected],
                                evaluation_queries=[names[i] for i in evaluation], response_prediction_nrmse=risk))
        print(f'{args.setting} budget {budget} complete', flush=True)
    dest.write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        evidence='Development only; preexisting source responses for every requested input, target outcomes restricted to calibration queries and contexts.',
        calibration_documents=cal.tolist(), evaluation_documents=test.tolist(),
        ridge=1e-2, summary=records), indent=2)+'\n', encoding='utf-8')
    print(json.dumps([r for r in records if r['budget'] == 3], indent=2))


if __name__ == '__main__':
    main()
