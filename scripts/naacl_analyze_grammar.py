from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np

DESCRIPTION = 'Request-family errors for grammar runs, pooled within grammar and averaged over grammars and runs.'
FAMILIES = dict(full=['full'], parts=['verb', 'number', 'gender'],
                new=['verb_number', 'verb_gender', 'number_gender', 'participation_a', 'participation_b',
                     'participation_c', 'half'])


def summarize(run):
    cfg = json.loads((run / 'config.resolved.json').read_text())
    values = np.load(run / 'responses.npz')
    panel = json.loads((run / 'panel.json').read_text())
    queries = list(panel['query_order'])
    tasks = np.array([r['task'] for r in panel['rows']])
    qi = {q: i for i, q in enumerate(queries)}
    none, source = values['none'], values['source']
    out = {}
    for method in values.files:
        if method in ('none', 'source'):
            continue
        row = {}
        for fam, qs in FAMILIES.items():
            ix = [qi[q] for q in qs]
            per_task = []
            for t in sorted(set(tasks)):
                cols = tasks == t
                err = (values[method][ix][:, cols] - source[ix][:, cols]) ** 2
                eff = (source[ix][:, cols] - none[ix][:, cols]) ** 2
                per_task.append(np.sqrt(err.sum() / eff.sum()))
            row[fam] = float(np.mean(per_task))
        out[method] = row
    return cfg, out


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    per_run = {}
    for run in args.runs:
        assert json.loads((run / 'status.json').read_text())['status'] == 'PASS', run
        cfg, out = summarize(run)
        per_run[run.name] = dict(source_seed=cfg.get('source_seed'), target_seed=cfg['target_seed'], methods=out)
    methods = sorted(set.intersection(*(set(r['methods']) for r in per_run.values())))
    summary = {m: {f: float(np.mean([r['methods'][m][f] for r in per_run.values()])) for f in FAMILIES} for m in methods}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(),
                                           runs=[str(r) for r in args.runs], summary=summary, per_run=per_run), indent=1))
    width = max(len(m) for m in methods)
    print(f"{'method':<{width}}  full     parts    new")
    for m in methods:
        s = summary[m]
        print(f"{m:<{width}}  {s['full']:.4f}   {s['parts']:.4f}   {s['new']:.4f}")


if __name__ == '__main__':
    main()
