from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np

from naacl_analyze_task_use import worst_group

RUNS = Path('D:/CCAD_Storage/runs/cross_seed_models')
DESCRIPTION = ('EXP-R (research lead decision 4): SHIFT on the target dictionary with the transferred explanation T '
               '(state-fit maps, three seeds) against the rediscovered explanation D, Hungarian matching and the source '
               'edit. Worst-group accuracy of the SHIFT task after each deletion, with the classifier head retrained on '
               'the edited data and kept original, and the part contrast (pronouns only minus descriptions of women '
               'only). Bootstrap over balanced test biographies within profession-by-gender strata and over T seeds.')
T = [f'subset_state_sup_hungarian_s{s}' for s in [0, 1, 2]]


def analyse(run, draws, rng):
    logits = np.load(run / 'task_logits.npz')
    lab = json.loads((run / 'task_labels.json').read_text())['shift']
    true, spur = np.array(lab['true']), np.array(lab['spur'])
    w = np.zeros((draws, len(true)))
    for s in np.unique(true * 2 + spur):
        idx = np.flatnonzero(true * 2 + spur == s)
        picks = rng.choice(idx, size=(draws, len(idx)))
        for d in range(draws):
            np.add.at(w[d], picks[d], 1.)
    w = np.vstack([np.ones(len(true)), w])
    pick = rng.integers(0, len(T), (draws + 1, len(T)))
    pick[0] = np.arange(len(T))
    out = {}
    for classifier in ['retrained', 'original']:
        wg = {}
        for part in ['complete', 'pronouns', 'female_descriptions']:
            for m in ['source', 'hungarian_decoder', 'rediscovered']:
                wg[part, m] = worst_group(logits[f'shift|{part}|{m}|{classifier}'] > 0, true, spur, w)
            per = np.stack([worst_group(logits[f'shift|{part}|{t}|{classifier}'] > 0, true, spur, w) for t in T])
            wg[part, 'transferred'] = per[pick, np.arange(len(w))[:, None]].mean(1)
        rec = {f'{p}|{m}': float(v[0]) for (p, m), v in wg.items()}
        ci = lambda v: np.quantile(v[1:], [.025, .975]).tolist()
        diff = wg['complete', 'transferred'] - wg['complete', 'rediscovered']
        contrast = {m: wg['pronouns', m] - wg['female_descriptions', m] for m in ['source', 'transferred', 'rediscovered', 'hungarian_decoder']}
        dist = {m: np.abs(contrast[m] - contrast['source']) for m in ['transferred', 'rediscovered', 'hungarian_decoder']}
        out[classifier] = dict(
            worst_group=rec,
            complete_transferred_minus_rediscovered=dict(point=float(diff[0]), interval=ci(diff)),
            contrast={m: dict(point=float(v[0]), interval=ci(v)) for m, v in contrast.items()},
            contrast_distance_to_source={m: dict(point=float(v[0]), interval=ci(v)) for m, v in dist.items()},
            transferred_closer_than_rediscovered=dict(point=float((dist['rediscovered'] - dist['transferred'])[0]),
                                                      interval=ci(dist['rediscovered'] - dist['transferred'])))
    return out


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(20261011)
    out = dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, draws=args.draws, runs={})
    for name in args.runs:
        run = RUNS / name
        assert json.loads((run / 'status.json').read_text())['status'] == 'PASS', run
        red = json.loads((RUNS / name.replace('EXPR', 'REDISCOVER') / 'REDISCOVERY.json').read_text())
        res = analyse(run, args.draws, rng)
        res['rediscovery'] = {k: red[k] for k in ['target_count', 'checked', 'accepted', 'per_part']}
        out['runs'][name] = res
        r = res['retrained']
        print(name, 'complete T-D', round(r['complete_transferred_minus_rediscovered']['point'], 3),
              [round(x, 3) for x in r['complete_transferred_minus_rediscovered']['interval']],
              'contrast distance D-T', round(r['transferred_closer_than_rediscovered']['point'], 3),
              [round(x, 3) for x in r['transferred_closer_than_rediscovered']['interval']])
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
