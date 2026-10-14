import argparse
import json
from pathlib import Path

import numpy as np

DESCRIPTION = ('Adoption test for graphs filtered by the unit test (criteria written before the runs). Per prompt and '
               'node list, the intervention prediction error E = sum_s (actual_s - predicted_s)^2 / sum_s actual_s^2 '
               'over the node subsets, for resampling (rs, primary) and input removal (rm, second). Per model: medians, '
               'the best baseline among all, random, layer and plt by median E, and the paired difference test minus '
               'that baseline over prompts with a 95% bootstrap interval of its median. A model passes when, for both rs '
               'and rm, the test list has median E at most 0.8 times the best baseline\'s and the interval lies below '
               'zero, and the median absolute model change for its first 10 units under rs is at least 0.9 times the '
               'larger of the layer and plt lists\' (coverage). The filter is adopted when at least two models pass.')
BASELINES = ['all', 'random', 'layer', 'plt']


def errors(lists, kind):
    out = {}
    for name, v in lists.items():
        a = np.array([s[f'actual_{kind}'] for s in v['subsets']])
        p = np.array([s[f'pred_{kind}'] for s in v['subsets']])
        if len(a) and (a ** 2).sum() > 0:
            out[name] = float(((a - p) ** 2).sum() / (a ** 2).sum())
    return out


def top10(lists):
    # Absolute model change under resampling for the subset of the first 10 units (or all units when fewer).
    out = {}
    for name, v in lists.items():
        m = len(v['units'])
        for s in v['subsets']:
            if s['units'] == [u['node'] for u in v['units'][:min(10, m)]]:
                out[name] = abs(s['actual_rs'])
                break
    return out


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--models', nargs='+', required=True, help='name=CLT_ROWS.json,PLT_ROWS.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    summary = {}
    for spec in args.models:
        name, paths = spec.split('=')
        clt_path, plt_path = paths.split(',')
        clt = {r['graph']: r for r in json.loads(Path(clt_path).read_text())['rows']}
        plt = {r['graph']: r for r in json.loads(Path(plt_path).read_text())['rows']}
        graphs = [g for g in clt if g in plt]
        per = {g: {**clt[g]['lists'], **plt[g]['lists']} for g in graphs}
        d = dict(prompts=len(graphs), test_size=float(np.median([len(per[g]['test']['units']) for g in graphs])),
                 pool_failing=float(np.median([sum(not u['passes'] for u in clt[g]['pool']) for g in graphs])))
        ok = True
        for kind in ('rs', 'rm'):
            e = {g: errors(per[g], kind) for g in graphs}
            names = ['test'] + BASELINES
            med = {n: float(np.median([e[g][n] for g in graphs if n in e[g]])) for n in names}
            best = min(BASELINES, key=lambda n: med[n])
            diff = np.array([e[g]['test'] - e[g][best] for g in graphs if 'test' in e[g] and best in e[g]])
            boot = [np.median(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(2000)]
            ci = [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
            passed = med['test'] <= .8 * med[best] and ci[1] < 0
            d[kind] = dict(median=med, best=best, diff=float(np.median(diff)), diff_ci=ci, passed=bool(passed))
            ok = ok and passed
        cov = {g: top10(per[g]) for g in graphs}
        cmed = {n: float(np.median([cov[g][n] for g in graphs if n in cov[g]])) for n in ['test'] + BASELINES}
        d['coverage'] = dict(median=cmed, passed=bool(cmed['test'] >= .9 * max(cmed['layer'], cmed['plt'])))
        d['passed'] = bool(ok and d['coverage']['passed'])
        summary[name] = d
        print(name, json.dumps(d))
    adopted = sum(d['passed'] for d in summary.values()) >= 2
    print(json.dumps(dict(adopted=adopted)))
    args.out.write_text(json.dumps(dict(description=DESCRIPTION, models=summary, adopted=adopted), indent=1))


if __name__ == '__main__':
    main()
