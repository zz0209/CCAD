from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np

DESCRIPTION = ('Part-level SHIFT in Gemma-2-2B on the original classifier and on four later classifiers that never '
               'supervised a correspondence. For each method, part and classifier head, the balanced agreement of its '
               'answers with those after the source deletion and the deviation of its worst-group accuracy from the '
               'source, with bootstrap intervals over biographies within profession-by-gender strata and training seeds.')
LATER = ['composer_surgeon_o0', 'composer_surgeon_o1', 'model_software_engineer_o0', 'model_software_engineer_o1']


def group_of(method):
    return method.rsplit('_s', 1)[0] if '_sup_' in method else method


def worst_group(pred, true, spur, w):
    acc = []
    for t in (0, 1):
        for s in (0, 1):
            sel = (true == t) & (spur == s)
            acc.append((w[:, sel] * (pred[sel] == t)).sum(1) / w[:, sel].sum(1))
    return np.min(acc, axis=0)


def balanced_agreement(pred, ref, clean, w):
    changed = ref != clean
    agree = pred == ref
    a = (w[:, changed] * agree[changed]).sum(1) / w[:, changed].sum(1)
    b = (w[:, ~changed] * agree[~changed]).sum(1) / w[:, ~changed].sum(1)
    return .5 * (a + b)


def analyse(run, draws, rng):
    logits = np.load(run / 'task_logits.npz')
    labels = json.loads((run / 'task_labels.json').read_text())
    keys = [k.split('|') for k in logits.files]
    methods = sorted({m for _, part, m, _ in keys if part != 'none'})
    parts = sorted({part for _, part, _, _ in keys if part != 'none'})
    groups = {}
    for m in methods:
        groups.setdefault(group_of(m), []).append(m)
    weights = {}
    for task, lab in labels.items():
        true, spur = np.array(lab['true']), np.array(lab['spur'])
        w = np.zeros((draws, len(true)))
        for s in np.unique(true * 2 + spur):
            idx = np.flatnonzero(true * 2 + spur == s)
            picks = rng.choice(idx, size=(draws, len(idx)))
            for d in range(draws):
                np.add.at(w[d], picks[d], 1.)
        weights[task] = (true, spur, np.vstack([np.ones(len(true)), w]))
    seed_picks = {n: rng.integers(0, n, (draws + 1, n)) for n in {len(v) for v in groups.values()}}
    for n, p in seed_picks.items():
        p[0] = np.arange(n)
    out = {}
    for task, (true, spur, w) in weights.items():
        clean = logits[f'{task}|none|clean|original'] > 0
        out[task] = {}
        for part in parts:
            out[task][part] = {}
            for classifier in ['original', 'retrained']:
                # Answers of a retrained head are compared with the source after its own retraining.
                ref = logits[f'{task}|{part}|source|{classifier}'] > 0
                src_wg = worst_group(ref, true, spur, w)
                row = {}
                for g, members in groups.items():
                    agree = np.stack([balanced_agreement(logits[f'{task}|{part}|{m}|{classifier}'] > 0, ref, clean, w)
                                      for m in members])
                    dev = np.stack([worst_group(logits[f'{task}|{part}|{m}|{classifier}'] > 0, true, spur, w) - src_wg
                                    for m in members])
                    pick = seed_picks[len(members)]
                    row[g] = dict(agreement=agree[pick, np.arange(len(w))[:, None]].mean(1),
                                  wg_deviation=dev[pick, np.arange(len(w))[:, None]].mean(1),
                                  wg=(dev + src_wg)[pick, np.arange(len(w))[:, None]].mean(1))
                row['_source_wg'] = src_wg
                out[task][part][classifier] = row
    return out, groups


def summarize(out, groups, tasks):
    summary = {}
    for part in out[tasks[0]]:
        summary[part] = {}
        for classifier in ['original', 'retrained']:
            summary[part][classifier] = {}
            for g in groups:
                agree = np.mean([out[t][part][classifier][g]['agreement'] for t in tasks], axis=0)
                absdev = np.mean([np.abs(out[t][part][classifier][g]['wg_deviation']) for t in tasks], axis=0)
                wg = np.mean([out[t][part][classifier][g]['wg'] for t in tasks], axis=0)
                summary[part][classifier][g] = dict(
                    agreement=float(agree[0]), agreement_ci=np.quantile(agree[1:], [.025, .975]).tolist(),
                    abs_wg_deviation=float(absdev[0]), abs_wg_deviation_ci=np.quantile(absdev[1:], [.025, .975]).tolist(),
                    wg=float(wg[0]), _agree=agree, _absdev=absdev)
            src = np.mean([out[t][part][classifier]['_source_wg'] for t in tasks], axis=0)
            summary[part][classifier]['_source_wg'] = float(src[0])
    return summary


def contrasts(summary, pairs):
    res = {}
    for part, by_cls in summary.items():
        for classifier, rows in by_cls.items():
            for a, b in pairs:
                if a in rows and b in rows:
                    for metric in ['_agree', '_absdev']:
                        d = rows[a][metric] - rows[b][metric]
                        res[f'{part}|{classifier}|{metric[1:]}|{a}-{b}'] = dict(
                            point=float(d[0]), interval=np.quantile(d[1:], [.025, .975]).tolist())
    return res


def strip(summary):
    return {p: {c: {g: ({k: v for k, v in r.items() if not k.startswith('_')} if isinstance(r, dict) else r)
                    for g, r in rows.items()} for c, rows in by.items()} for p, by in summary.items()}


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261011)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    record = dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, draws=args.draws, runs={})
    pairs = [('part_sup_hungarian', 'complete_sup_hungarian'), ('part_sup_hungarian', 'random_sup_hungarian'),
             ('part_sup_hungarian', 'singleton_sup_hungarian'), ('random_sup_hungarian', 'complete_sup_hungarian'),
             ('part_sup_hungarian', 'hungarian_decoder'), ('part_sup_hungarian', 'readout')]
    for run in args.runs:
        assert json.loads((run / 'status.json').read_text())['status'] == 'PASS', run
        out, groups = analyse(run, args.draws, rng)
        later, shift = summarize(out, groups, LATER), summarize(out, groups, ['shift'])
        record['runs'][run.name] = dict(later=strip(later), shift=strip(shift), later_contrasts=contrasts(later, pairs),
                                        shift_contrasts=contrasts(shift, pairs),
                                        per_task={t: {p: {c: {g: dict(wg=float(r['wg'][0]), agreement=float(r['agreement'][0]))
                                                              for g, r in rows.items() if g != '_source_wg'}
                                                          for c, rows in by.items()} for p, by in parts.items()}
                                                  for t, parts in out.items()})
        print(run.name)
        for name, summ in [('later', later), ('shift', shift)]:
            for part in summ:
                for classifier in ['original', 'retrained']:
                    rows = summ[part][classifier]
                    print(f'  {name} {part[:7]} {classifier:<9} source wg {100 * rows["_source_wg"]:.1f}')
                    for g, r in rows.items():
                        if g.startswith('_'):
                            continue
                        print(f'    {g:<26} agree {100 * r["agreement"]:5.1f}  |dWG| {100 * r["abs_wg_deviation"]:5.1f}  '
                              f'WG {100 * r["wg"]:5.1f}')
        for k, v in record['runs'][run.name]['later_contrasts'].items():
            print(f'  later {k:<70} {v["point"]:+.4f} [{v["interval"][0]:+.4f}, {v["interval"][1]:+.4f}]')
    args.output.write_text(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
