import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from itertools import combinations_with_replacement

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=32)
    parser.add_argument('--arity', type=int, choices=[2, 3], default=2)
    args = parser.parse_args()
    assert not args.output.exists()
    history = []
    exposed = set()
    for path in sorted(Path('runs').glob('*/panel.json')):
        data = json.loads(path.read_text())
        rows = data.get('rows', []) if isinstance(data, dict) else []
        names = ['a', 'b', 'c'][:args.arity]
        found = {tuple(sorted(r[k] for k in names)) for r in rows
                 if isinstance(r, dict) and all(k in r for k in names) and (args.arity == 3 or 'c' not in r)}
        if found:
            exposed.update(found)
            history.append(dict(path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), questions=len(found)))
    candidates = [values for values in combinations_with_replacement(range(10, 90 if args.arity == 2 else 40), args.arity)
                  if sum(values) < 100 and values not in exposed]
    rng = np.random.default_rng(2026092003)
    order = rng.permutation(len(candidates))
    used = set()
    selected = []
    for index in order:
        if index in used:
            continue
        current = candidates[index]
        partners = [j for j in order if j not in used and j != index
                    and (sum(candidates[j]) % 10 != sum(current) % 10)
                    and (sum(candidates[j]) // 10 != sum(current) // 10)
                    and not set(candidates[j]) & set(current)]
        if not partners:
            continue
        j = partners[0]
        selected.append((candidates[index], candidates[j]))
        used.update([index, j])
        if len(selected) == args.pairs:
            break
    assert len(selected) == args.pairs, (len(candidates), len(selected))
    templates = ['16+27=43\n38+25=63\n{a}+{b}=',
                 'Q: Add 16 and 27. A: 43\nQ: Add 38 and 25. A: 63\nQ: Add {a} and {b}. A:']
    if args.arity == 3:
        templates = ['12+16+15=43\n23+18+22=63\n{a}+{b}+{c}=',
                     'Q: Add 12, 16 and 15. A: 43\nQ: Add 23, 18 and 22. A: 63\nQ: Add {a}, {b} and {c}. A:']
    rows, pairs = [], []
    for template_id, template in enumerate(templates):
        for cluster, (recipient, donor) in enumerate(selected):
            row_ids = []
            for values in [recipient, donor]:
                operands = dict(zip(names, values))
                row_ids.append(len(rows))
                total = sum(values)
                rows.append(dict(**operands, template=template_id, total=total, unit=total%10, tens=total//10,
                    carry=sum(v%10 for v in values)>=10, split='confirmation', prompt=template.format(**operands)))
            r, d = [rows[i] for i in row_ids]
            pairs.append(dict(recipient=row_ids[0], donor=row_ids[1], template=template_id, cluster=cluster,
                base_answer=r['total'], donor_answer=d['total'], unit_answer=10*r['tens']+d['unit'],
                tens_answer=10*d['tens']+r['unit'], recipient_carry=r['carry'], donor_carry=d['carry']))
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), rows=rows, pairs=pairs,
        history=history, exposed_questions=len(exposed), available_questions=len(candidates),
        arity=args.arity,
        scope='New unordered operand questions absent from retained arithmetic panels of the same arity; paired forms share each question cluster')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['rows', 'pairs', 'history']}))


if __name__ == '__main__':
    main()
