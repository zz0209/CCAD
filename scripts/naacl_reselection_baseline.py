from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RG09 = Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round09')
DESCRIPTION = ('Re-selection baseline for the GPT-2 grammatical explanations. For every direction from source seed s to '
               'target seed t, the explanation selected by the same source-gradient rule in the target SAE is compared '
               'with the source explanation on the same held-out pairs and requests, next to the transferred correspondences.')
FAMILIES = dict(complete=['full'], parts=['verb', 'number', 'gender'],
                new=['verb_number', 'verb_gender', 'number_gender', 'participation_a', 'participation_b', 'participation_c',
                     'half'])


def load(s, t):
    run = RG09 / f'RG09_GPT2_SOURCE_REUSE_CONFIRM_S{s}_T{t}_20260921'
    assert json.loads((run / 'status.json').read_text())['status'] == 'PASS', run
    panel = json.loads((run / 'panel.json').read_text())
    return dict(np.load(run / 'responses.npz')), panel


def nrmse(values, ref, none, panel):
    qi = {q: i for i, q in enumerate(panel['query_order'])}
    tasks = np.array([r['task'] for r in panel['rows']])
    out = {}
    for fam, qs in FAMILIES.items():
        ix = [qi[q] for q in qs]
        per_task = []
        for task in sorted(set(tasks)):
            cols = tasks == task
            err = ((values[ix][:, cols] - ref[ix][:, cols]) ** 2).sum()
            eff = ((ref[ix][:, cols] - none[ix][:, cols]) ** 2).sum()
            per_task.append(np.sqrt(err / eff))
        out[fam] = float(np.mean(per_task))
    return out


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'artifacts/naacl_sprint_20261010/e1_matching/GPT2_RESELECTION_BASELINE.json')
    args = parser.parse_args()
    seeds = [2, 3, 4, 5]
    own = {}
    for t in seeds:
        values, panel = load(t, next(x for x in seeds if x != t))
        own[t] = (values['source'], panel)
    rows = {}
    for s in seeds:
        for t in seeds:
            if s == t:
                continue
            values, panel = load(s, t)
            assert panel['rows'] == own[t][1]['rows'] and panel['query_order'] == own[t][1]['query_order']
            ref, none = values['source'], values['none']
            rows[f's{s}_t{t}'] = dict(
                reselected=nrmse(own[t][0], ref, none, panel),
                **{k: nrmse(values[k], ref, none, panel) for k in ['initial', 'readout_initial', 'reuse_whole', 'reuse_program']})
    methods = list(next(iter(rows.values())))
    summary = {m: {f: float(np.mean([r[m][f] for r in rows.values()])) for f in FAMILIES} for m in methods}
    args.output.write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(), summary=summary,
                                           directions=rows), indent=1))
    for m, v in summary.items():
        print(f"{m:<16} complete {v['complete']:.3f}  parts {v['parts']:.3f}  new {v['new']:.3f}")


if __name__ == '__main__':
    main()
