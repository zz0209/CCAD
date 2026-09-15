"""Source-fit-only carry readouts; cross-validation is development, not causality."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/correspondence_reform_20260913'
PARENT = ROOT / 'runs/REFORM_R51_code_range_source_v1_20260915'
OLD = ROOT / 'runs/REFORM_R32_qwen_counterfactual_source_v1_20260914'


def fit_reader(x, y, strength):
    mean = x.mean(0)
    # One global scale preserves SAE coefficient geometry and avoids amplifying rare codes.
    scale = np.sqrt(np.mean((x - mean) ** 2))
    xx = (x - mean) / scale
    gram = xx @ xx.T
    ridge = strength * np.trace(gram) / len(x)
    dual = np.linalg.solve(gram + ridge * np.eye(len(x)), y - y.mean())
    w = xx.T @ dual / scale
    return w, float(y.mean() - mean @ w)


def load_fit():
    panel = json.loads((PARENT / 'SOURCE_FIT_PANEL.json').read_text())
    rows = panel['rows']
    arrays = {}
    for name, files, key in [('hidden', ['states.npz', 'states.npz'], 'hidden'),
                             ('codes', ['source_seed1.npz', 'evaluation_seed1.npz'], 'codes')]:
        pieces = []
        for parent, indices, filename in zip([OLD, PARENT],
                [panel['original_indices'], panel['new_indices']], files):
            with np.load(parent / filename) as d:
                pieces.append(d[key][indices, 0].astype(np.float64))
        arrays[name] = np.concatenate(pieces)
    with np.load(PARENT / 'rule_members_seed1.npz') as d:
        members = d['code_scalar_64']
    assert len(rows) == 405 and all(r['split'] == 'fit' for r in rows)
    return rows, arrays, members


def main():
    rows, arrays, members = load_fit()
    arity = np.array([int('c' in r) for r in rows])
    y = np.array([r['carry'] for r in rows], dtype=float)
    folds = np.array([int(hashlib.sha256(str(tuple(r[k] for k in ['a','b','c'] if k in r)).encode()).hexdigest()[:8],16)%5 for r in rows])
    result = []; payload = {'members': members, 'carry': y, 'arity': arity, 'folds': folds}
    for tag, x in [('members', arrays['codes'][:, members]), ('full_codes', arrays['codes']), ('raw', arrays['hidden'])]:
        for strength in [.001, .01, .1, 1.]:
            pred = np.zeros(len(y)); weights = []; biases = []
            for r in range(2):
                use = arity == r
                for fold in range(5):
                    tr = use & (folds != fold); te = use & (folds == fold)
                    w, b = fit_reader(x[tr], y[tr], strength)
                    pred[te] = x[te] @ w + b
                w,b = fit_reader(x[use],y[use],strength)
                weights.append(w);biases.append(b)
            name=f'{tag}_{strength:g}'
            payload[name+'_weights']=np.array(weights);payload[name+'_bias']=np.array(biases)
            payload[name+'_cv_prediction']=pred
            cells = []
            for r in range(2):
                use=arity==r
                cells.append({'arity':r+2,'n':int(use.sum()),'rmse':float(np.sqrt(np.mean((pred[use]-y[use])**2))),
                              'rounded_accuracy':float(np.mean(np.rint(pred[use]).clip(0,r+1)==y[use]))})
            result.append({'name':name,'cv_mse':float(np.mean((pred-y)**2)),'cells':cells})
    selected={tag:min([r for r in result if r['name'].startswith(tag+'_')],key=lambda r:r['cv_mse'])['name'] for tag in ['members','full_codes','raw']}
    np.savez_compressed(ART/'R52_SOURCE_READOUTS.npz',**payload)
    report={'written_at_utc':datetime.now(timezone.utc).isoformat(),'source_fit_rows':405,'folds':5,
            'scope':'Within-fit cross-validation to choose source readout shrinkage; no intervention or held-out behavior claim.',
            'results':result,'selected':selected}
    (ART/'R52_SOURCE_READOUTS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'selected':selected,'results':[r for r in result if r['name'] in selected.values()]}))


if __name__ == '__main__':
    main()
