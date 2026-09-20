import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def analyze(run):
    assert json.loads((run/'status.json').read_text())['status'] == 'PASS'
    records = [json.loads(line) for line in (run/'metrics.raw.jsonl').read_text().splitlines()]
    base = {r['row_id']: r['answer'] for r in records if r['kind'] == 'base'}
    source = {(r['operation'], r['mode'], r['row_id']): r['answer'] for r in records if r.get('method') == 'source'}
    outcomes = {}
    for row in records:
        if row['kind'] != 'source_field' or row['method'] == 'source':
            continue
        key = row['operation'], row['mode'], row['row_id']
        value = source[key]
        group = row['target_seed'], row['operation'], 'full' if row['mode'] == 'query_0' else 'parts', row['method']
        valid = value is not None and 10 <= value < 100
        outcomes.setdefault(group, []).append((row['answer'] == value and valid, value != base[row['row_id']],
                                              row['exact_hybrid'], row['row_id'], int(row['mode'].split('_')[-1])))
    cells = []
    for key, values in outcomes.items():
        a = np.asarray(values)
        changed = a[:, 1].astype(bool)
        c = float(a[changed, 0].mean()) if changed.any() else None
        u = float(a[~changed, 0].mean()) if (~changed).any() else None
        cells.append(dict(target=key[0], operation=key[1], family=key[2], method=key[3], n=len(a),
            exact_agreement=float(a[:, 0].mean()), changed_agreement=c, unchanged_agreement=u,
            balanced_agreement=None if c is None or u is None else (c+u)/2, hybrid_success=float(a[:, 2].mean()),
            source_changed=int(changed.sum())))
    return dict(run=run.as_posix(), cells=cells,
        source_full_hybrid=float(np.mean([r['exact_hybrid'] for r in records if r.get('method') == 'source' and r['mode'] == 'query_0'])),
        raw_sha256=hashlib.sha256((run/'metrics.raw.jsonl').read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), results=[analyze(run) for run in args.runs])
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
