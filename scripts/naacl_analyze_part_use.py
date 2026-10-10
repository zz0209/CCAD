from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np

DESCRIPTION = ('Part-level SHIFT outcomes in Gemma-2-2B. For each method and part, profession, gender and worst-group '
               'accuracy of the retrained classifier, averaged over training seeds, and the comparison between the two '
               'parts that the original explanation supports.')
METRICS = ['profession', 'gender', 'worst_group']


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = dict(written_utc=datetime.now(timezone.utc).isoformat(), runs={})
    for run in args.runs:
        assert json.loads((run / 'status.json').read_text())['status'] == 'PASS', run
        res = json.loads((run / 'PART_USE.json').read_text())
        groups = {}
        for key, v in res.items():
            part, method = key.split(':')
            name = method.rsplit('_s', 1)[0] if '_sup_' in method else method
            groups.setdefault(name, {}).setdefault(part, []).append(v)
        out = {}
        for name, parts in groups.items():
            out[name] = {}
            for classifier in ['original', 'retrained']:
                row = {}
                for part, vals in parts.items():
                    vals = [x[classifier] for x in vals]
                    row[part] = {m: float(np.mean([x[m] for x in vals])) for m in METRICS}
                    row[part]['seed_range'] = {m: [float(min(x[m] for x in vals)), float(max(x[m] for x in vals))]
                                               for m in METRICS}
                # Difference between deleting pronouns and deleting descriptions of women, the comparison the parts support.
                row['pronouns_minus_women'] = {m: row['pronouns'][m] - row['female_descriptions'][m] for m in METRICS}
                out[name][classifier] = row
        for name in out:
            for classifier in ['original', 'retrained']:
                src, row = out['source'][classifier], out[name][classifier]
                row['deviation_from_source'] = {f'{p}_{m}': row[p][m] - src[p][m]
                                                for p in ['pronouns', 'female_descriptions'] for m in METRICS}
        record['runs'][run.name] = out
        print(run.name)
        for classifier in ['original', 'retrained']:
            print(' ', classifier)
            for name, rows in out.items():
                row = rows[classifier]
                print(f"    {name:<26}" + '  '.join(f"{p[:5]} prof {100 * row[p]['profession']:.1f} gen "
                                                  f"{100 * row[p]['gender']:.1f} wg {100 * row[p]['worst_group']:.1f}"
                                                  for p in ['pronouns', 'female_descriptions'])
                      + f"  diff wg {100 * row['pronouns_minus_women']['worst_group']:+.1f}")
    args.output.write_text(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
