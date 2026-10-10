from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np

DESCRIPTION = 'Request-family errors of every executor in infinitive runs, with optional public-target metadata.'


def families(queries):
    return dict(complete=['full'], parts=['predicate', 'object'],
                participation=[q for q in queries if q.startswith(('interior_', 'boundary_'))],
                members=[q for q in queries if q.startswith('member_subset_')])


def summarize(run):
    index = json.loads((run / 'INDEX.json').read_text())
    values = np.load(run / 'responses.npz')
    qi = {q: i for i, q in enumerate(index['queries'])}
    fam = families(index['queries'])
    none, source = values['none'], values['source']
    out = {}
    for method in values.files:
        if method in ('none', 'source'):
            continue
        lp = values[method]
        row = {}
        for name, qs in fam.items():
            ix = [qi[q] for q in qs]
            err, eff, mine = lp[ix] - source[ix], source[ix] - none[ix], lp[ix] - none[ix]
            row[name] = dict(nrmse=float(np.sqrt((err ** 2).mean() / (eff ** 2).mean())),
                             effect_ratio=float(np.sqrt((mine ** 2).mean() / (eff ** 2).mean())))
        out[method] = row
    return out


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--targets-json', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads(args.targets_json.read_text())['targets'] if args.targets_json else {}
    per_run = {}
    for run in args.runs:
        cfg = json.loads((run / 'config.resolved.json').read_text())
        assert json.loads((run / 'status.json').read_text())['status'] == 'PASS', run
        per_run[run.name] = dict(target_seed=cfg['target_seed'], target=meta.get(str(cfg['target_seed'])), methods=summarize(run))
    methods = sorted(set.intersection(*(set(r['methods']) for r in per_run.values())))
    summary = {m: {f: float(np.mean([r['methods'][m][f]['nrmse'] for r in per_run.values()]))
                   for f in ['complete', 'parts', 'participation', 'members']} for m in methods}
    record = dict(written_utc=datetime.now(timezone.utc).isoformat(), runs=[str(r) for r in args.runs],
                  summary=summary, per_run=per_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1))
    width = max(len(m) for m in methods)
    print(f"{'method':<{width}}  complete   parts   particip  members")
    for m in methods:
        s = summary[m]
        print(f"{m:<{width}}  {s['complete']:.4f}   {s['parts']:.4f}   {s['participation']:.4f}   {s['members']:.4f}")


if __name__ == '__main__':
    main()
