from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np

from naacl_supervision_table import SETTINGS, load

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION = ('Answer-level outcomes of part deletions in the grammar settings. For every method and part, the BLiMP '
               'accuracy of each grammatical function after the deletion, the drop on the deleted function relative to '
               'the other functions, and the balanced agreement of pair answers with the source edit.')
PARTS = ['verb', 'number', 'gender']


def analyse(setting):
    extras = setting.get('extra_runs', [None] * len(setting['runs']))
    loaded = [load(r, 'grammar', e) for r, e in zip(setting['runs'], extras)]
    out = {}
    for name, method in dict(source='source', none='none', **setting['methods']).items():
        acc, agree, selectivity = [], [], []
        for (values, _, groups), run in zip(loaded, setting['runs']):
            panel = json.loads((run / 'panel.json').read_text())
            qi = {q: i for i, q in enumerate(panel['query_order'])}
            tasks = sorted(set(groups))
            none = values['none'][qi['full']] > 0
            table = np.zeros((3, len(tasks)))
            agr = []
            for p, part in enumerate(PARTS):
                mine = values[method][qi[part]] > 0
                ref = values['source'][qi[part]] > 0
                for t in tasks:
                    table[p, t] = mine[groups == t].mean()
                changed = ref != none
                hits = mine == ref
                agr.append(.5 * (hits[changed].mean() + hits[~changed].mean()))
            acc.append(table)
            agree.append(np.mean(agr))
            # Accuracy drop on the deleted function minus the mean drop on the other two functions.
            base = np.array([(none[groups == t]).mean() for t in tasks])
            drop = base[None, :] - table
            selectivity.append(np.mean([drop[p, p] - np.mean(np.delete(drop[p], p)) for p in range(3)]))
        out[name] = dict(accuracy=np.mean(acc, 0).tolist(), answer_agreement=float(np.mean(agree)),
                         selectivity=float(np.mean(selectivity)))
    return out


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/naacl_sprint_20261010/e2_supervision/GRAMMAR_ANSWERS.json')
    args = parser.parse_args()
    record = dict(written_utc=datetime.now(timezone.utc).isoformat(), parts=PARTS, settings={})
    for setting in SETTINGS:
        if setting['kind'] != 'grammar':
            continue
        record['settings'][setting['key']] = res = analyse(setting)
        print(setting['key'])
        for m, v in res.items():
            print(f"  {m:<22} agreement {v['answer_agreement']:.3f}  selectivity {v['selectivity']:+.3f}  "
                  f"acc diag {np.round(np.diag(np.array(v['accuracy'])), 3)}")
    args.output.write_text(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
