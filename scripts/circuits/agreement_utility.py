import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DESCRIPTION = ('Does agreement with a second dictionary help find the units of a CLT graph whose later writes the '
               'model does not produce? A unit is unconfirmed when removing its input takes out less than --cut of its '
               'claimed later writes. Single predictors (agreement, read layer, share of the claimed write in later '
               'layers, log norm of the claimed write, activation, and how many graphs of the run list the same '
               'feature) are scored by AUROC on held-out prompts, and logistic models with the baselines and with the '
               'baselines plus agreement are fitted on one prompt set and scored on another, with graph-bootstrap '
               'intervals for the difference.')


def later(r, key):
    c = np.array(r['claim_energy'][1:])
    return float(np.sum(r[key][1:]) / c.sum())


def table(depth_path, reverse_path):
    rev = {(r['graph'], r['unit']): r for r in json.loads(Path(reverse_path).read_text())['rows'] if r['features'] == 1}
    rows = [r for r in json.loads(Path(depth_path).read_text())['rows']
            if len(r['claim_energy']) > 1 and sum(r['claim_energy'][1:]) > 0 and (r['graph'], r['unit']) in rev]
    feat_graphs = {}
    for r in rows:
        feat_graphs.setdefault(r['unit'].split('P')[0], set()).add(r['graph'])
    n_graphs = len({r['graph'] for r in rows})
    out = []
    for r in rows:
        v = rev[(r['graph'], r['unit'])]
        own = [res for k, res, _ in v['residual_by_offset'] if k == 0]
        if not own:
            continue
        c = np.array(r['claim_energy'])
        out.append(dict(graph=r['graph'], agree=1 - own[0], layer=r['layer'], later_share=float(c[1:].sum() / c.sum()),
                        log_norm=float(np.log(c.sum())), log_act=float(np.log(max(r['activation'], 1e-6))),
                        recurrence=len(feat_graphs[r['unit'].split('P')[0]]) / n_graphs, confirm=later(r, 'removed_u')))
    return out


BASE = ['layer', 'later_share', 'log_norm', 'log_act', 'recurrence']


def design(units, cols):
    X = np.array([[u[c] for c in cols] for u in units], dtype=float)
    return X


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--train', required=True, help='DEPTH_REFEREE_ROWS.json,REVERSE_ROWS.json')
    parser.add_argument('--test', required=True, help='DEPTH_REFEREE_ROWS.json,REVERSE_ROWS.json')
    parser.add_argument('--cut', type=float, default=.2)
    parser.add_argument('--name', default='')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    tr, te = table(*args.train.split(',')), table(*args.test.split(','))
    ytr = np.array([u['confirm'] < args.cut for u in tr])
    yte = np.array([u['confirm'] < args.cut for u in te])
    res = dict(train_units=len(tr), test_units=len(te), test_unconfirmed=float(yte.mean()))
    # Single predictors on the test set, oriented so that higher means more likely unconfirmed.
    single = {}
    for c in ['agree'] + BASE:
        x = np.array([u[c] for u in te], dtype=float)
        auc = roc_auc_score(yte, x)
        single[c] = max(auc, 1 - auc)
    res['single_auroc'] = single
    # Logistic models fitted on the train set.
    mu, sd = None, None
    preds = {}
    for name, cols in [('baselines', BASE), ('baselines+agreement', BASE + ['agree']), ('agreement', ['agree'])]:
        Xtr, Xte = design(tr, cols), design(te, cols)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        m = LogisticRegression(max_iter=2000).fit((Xtr - mu) / sd, ytr)
        preds[name] = m.predict_proba((Xte - mu) / sd)[:, 1]
        res[f'auroc_{name}'] = float(roc_auc_score(yte, preds[name]))
    # Graph bootstrap of the gain from adding agreement, and precision among the 20% of units ranked most likely
    # unconfirmed.
    rng = np.random.default_rng(0)
    graphs = sorted({u['graph'] for u in te})
    idx = {g: np.flatnonzero(np.array([u['graph'] for u in te]) == g) for g in graphs}
    gains = []
    for _ in range(2000):
        pick = np.concatenate([idx[graphs[i]] for i in rng.choice(len(graphs), len(graphs))])
        if len(set(yte[pick])) < 2:
            continue
        gains.append(roc_auc_score(yte[pick], preds['baselines+agreement'][pick]) - roc_auc_score(yte[pick], preds['baselines'][pick]))
    res['auroc_gain_from_agreement'] = [float(np.mean(gains)), float(np.quantile(gains, .025)), float(np.quantile(gains, .975))]
    k = max(1, int(.2 * len(te)))
    for name, p in preds.items():
        top = np.argsort(-p)[:k]
        res[f'precision_top20pct_{name}'] = float(yte[top].mean())
    print(args.name, json.dumps(res))
    if args.out is not None:
        args.out.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
