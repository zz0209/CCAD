from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np

DESCRIPTION = 'Complete-edit versus part-edit fidelity of feature-matching and learned correspondences for the SHIFT explanation.'
PARTS = ['pronouns', 'names', 'associated_words']
UNIONS = ['pronouns+names', 'pronouns+associated_words', 'names+associated_words']
FAMILIES = dict(full=['full'], parts=PARTS, unions=UNIONS)


def load(run):
    logits, meta = {}, {}
    with (run / 'metrics.raw.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r.get('kind') != 'classification':
                continue
            logits.setdefault((r['method'], r['operation']), {})[r['component']] = r['logit']
            meta[r['component']] = (r['label'], r['gender'])
    docs = sorted(meta)
    values = {key: np.array([v[d] for d in docs]) for key, v in logits.items()}
    labels = np.array([meta[d][0] for d in docs])
    genders = np.array([meta[d][1] for d in docs])
    return docs, values, labels, genders


def cell_metrics(v, ref, base):
    changed = (ref > 0) != (base > 0)
    agree = (v > 0) == (ref > 0)
    balanced = .5 * (agree[changed].mean() + agree[~changed].mean()) if changed.any() and (~changed).any() else np.nan
    effect = ref - base
    return dict(balanced_agreement=float(balanced), nrmse_sq_num=float(((v - ref) ** 2).sum()),
                nrmse_sq_den=float((effect ** 2).sum()), changed=int(changed.sum()),
                effect_sq=float(((v - base) ** 2).sum()))


def bootstrap(runs_data, methods, draws, seed):
    # Joint resampling of target dictionaries and of biographies within label-by-gender strata,
    # shared by every method and query so that differences are paired.
    rng = np.random.default_rng(seed)
    docs0, _, labels, genders = runs_data[0]
    for docs, _, lab, gen in runs_data:
        assert docs == docs0 and np.array_equal(lab, labels) and np.array_equal(gen, genders)
    strata = [np.flatnonzero((labels == y) & (genders == g)) for y in [0, 1] for g in [0, 1]]
    out = {m: {f: np.empty(draws) for f in FAMILIES} for m in methods}
    for b in range(draws):
        targets = rng.integers(len(runs_data), size=len(runs_data))
        idx = np.concatenate([rng.choice(s, size=len(s)) for s in strata])
        for m in methods:
            for name, queries in FAMILIES.items():
                vals = []
                for t in targets:
                    values = runs_data[t][1]
                    base = values[('none', 'full')][idx]
                    num = sum(((values[(m, q)][idx] - values[('source', q)][idx]) ** 2).sum() for q in queries)
                    den = sum(((values[('source', q)][idx] - base) ** 2).sum() for q in queries)
                    vals.append(np.sqrt(num / den))
                out[m][name][b] = np.mean(vals)
    return out


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=0)
    parser.add_argument('--contrasts', nargs='*', default=[], help='pairs written as method_a-method_b')
    args = parser.parse_args()
    per_target, methods, runs_data = {}, None, []
    for run in args.runs:
        cfg = json.loads((run / 'config.resolved.json').read_text())
        assert json.loads((run / 'status.json').read_text())['status'] == 'PASS'
        docs, values, labels, genders = load(run)
        runs_data.append((docs, values, labels, genders))
        methods = [m for m in cfg['methods'] if m not in ('none', 'source')]
        base = values[('none', 'full')]
        out = {}
        for m in methods:
            fam = {}
            for name, queries in FAMILIES.items():
                cells = [cell_metrics(values[(m, q)], values[('source', q)], base) for q in queries]
                fam[name] = dict(balanced_agreement=float(np.mean([c['balanced_agreement'] for c in cells])),
                                 nrmse=float(np.sqrt(sum(c['nrmse_sq_num'] for c in cells) / sum(c['nrmse_sq_den'] for c in cells))),
                                 per_query_nrmse=[float(np.sqrt(c['nrmse_sq_num'] / c['nrmse_sq_den'])) for c in cells],
                                 effect_ratio=float(np.sqrt(sum(c['effect_sq'] for c in cells) / sum(c['nrmse_sq_den'] for c in cells))))
            pron = (values[(m, 'pronouns')] > 0) == labels
            name = (values[(m, 'names')] > 0) == labels
            fam['pronoun_minus_name_accuracy'] = float(pron.mean() - name.mean())
            out[m] = fam
        src_pron = ((values[('source', 'pronouns')] > 0) == labels).mean() - ((values[('source', 'names')] > 0) == labels).mean()
        per_target[run.name] = dict(target_seed=cfg['target_seed'], panel=cfg['evaluation_panel'], documents=len(docs),
                                    source_pronoun_minus_name_accuracy=float(src_pron), methods=out)
    summary = {}
    for m in methods:
        summary[m] = {}
        for name in FAMILIES:
            for metric in ['balanced_agreement', 'nrmse', 'effect_ratio']:
                vals = [t['methods'][m][name][metric] for t in per_target.values()]
                summary[m][f'{name}_{metric}'] = float(np.mean(vals))
        summary[m]['pronoun_minus_name_accuracy'] = float(np.mean([t['methods'][m]['pronoun_minus_name_accuracy'] for t in per_target.values()]))
    intervals, contrasts = {}, {}
    if args.draws:
        draws = bootstrap(runs_data, methods, args.draws, 20261010)
        intervals = {m: {f: np.quantile(draws[m][f], [.025, .975]).tolist() for f in FAMILIES} for m in methods}
        for pair in args.contrasts:
            a, b = pair.split('-')
            contrasts[pair] = {f: dict(point=summary[a][f'{f}_nrmse'] - summary[b][f'{f}_nrmse'],
                                       interval=np.quantile(draws[a][f] - draws[b][f], [.025, .975]).tolist())
                               for f in FAMILIES}
    record = dict(written_utc=datetime.now(timezone.utc).isoformat(), runs=[str(r) for r in args.runs],
                  summary=summary, intervals=intervals, contrasts=contrasts, draws=args.draws,
                  bootstrap='target dictionaries and biographies within label-by-gender strata, shared across methods',
                  per_target=per_target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1))
    width = max(len(m) for m in methods)
    print(f"{'method':<{width}}  full_agr  part_agr  union_agr  full_nrmse  part_nrmse  union_nrmse  pron-name")
    for m in methods:
        s = summary[m]
        print(f"{m:<{width}}  {s['full_balanced_agreement']:.4f}    {s['parts_balanced_agreement']:.4f}    {s['unions_balanced_agreement']:.4f}     "
              f"{s['full_nrmse']:.4f}      {s['parts_nrmse']:.4f}      {s['unions_nrmse']:.4f}      {s['pronoun_minus_name_accuracy']:+.4f}")


if __name__ == '__main__':
    main()
