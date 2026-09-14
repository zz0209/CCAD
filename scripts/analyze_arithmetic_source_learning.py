"""Describe source-learning interventions with their prompt/seed structure intact."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--pool-transfer-sources', action='store_true')
    args = ap.parse_args()
    status = json.loads((args.run / 'status.json').read_text())
    assert status['status'] == 'PASS'
    cfg = json.loads((args.run / 'config.resolved.json').read_text())
    panel = json.loads((args.run / 'panel.json').read_text())
    raw = args.run / 'metrics.raw.jsonl'
    records = [json.loads(line) for line in raw.read_text().splitlines()]
    groups = defaultdict(list)
    for row in records:
        if row['kind'] == 'source_patch' and row['seed'] != 0:
            method = row['method']
            if args.pool_transfer_sources and method.startswith('transfer_s'):
                method = 'transfer_' + method.split('_', 2)[2]
            if args.pool_transfer_sources and method.startswith('response_s'):
                method = 'response_' + method.split('_', 2)[2]
            if args.pool_transfer_sources and method.startswith('position_s'):
                method = 'position_' + method.split('_', 2)[2]
            groups[method, row['operation']].append(row)
    cells = []
    for (method, operation), rows in sorted(groups.items()):
        assert len(rows) == len(cfg['seeds']) * len(panel['pairs'])
        assert len({(r['seed'], r['row_id']) for r in rows}) == len(rows)
        cells.append(dict(method=method, operation=operation, n=len(rows),
                          **{key: float(np.mean([r[key] for r in rows]))
                             for key in ['exact_hybrid', 'target_digit_success', 'preserve_digit_success', 'edit_norm']},
                          seed_hybrid={str(s): float(np.mean([r['exact_hybrid'] for r in rows if r['seed'] == s])) for s in cfg['seeds']},
                          prompt_hybrid={str(t): float(np.mean([r['exact_hybrid'] for r in rows if r['task'] == f'template_{t}'])) for t in range(len(cfg['templates']))}))
    means = {m: float(np.mean([c['exact_hybrid'] for c in cells if c['method'] == m])) for m in sorted({c['method'] for c in cells})}
    fits = []
    for p in sorted(args.run.glob('counterfactual_seed*_k*_*.json')):
        d = json.loads(p.read_text())
        fits.append(dict(file=p.name, seed=d['seed'], method=d['initialization'],
                         gradient=d['gradient'], steps=d['updates'], latent_weight=d.get('latent_weight', 0),
                         first=d['trace'][0], last=d['trace'][-1], backward_budget=d['backward_budget']))
    # All variants start with the same pair-generation seed and sample schedule.
    schedules_equal = None
    if cfg.get('source_refit'):
        arrays = []
        for seed in cfg['seeds']:
            for name in cfg['source_refit']['objectives']:
                with np.load(args.run / f'counterfactual_seed{seed}_k{cfg["members"][0]}_{name}.npz') as z:
                    arrays.append(z['fit_schedule'])
        schedules_equal = all(np.array_equal(arrays[0], a) for a in arrays[1:])
        assert schedules_equal
    out = dict(written_utc=datetime.now(timezone.utc).isoformat(), run=str(args.run), status=status,
               scope=cfg['scope'], inference='Descriptive development; no independent confirmation claimed.',
               raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(), cells=cells,
               mean_hybrid=means, fits=fits, all_fit_schedules_equal=schedules_equal,
               base_accuracy={str(t): float(np.mean([r['correct'] for r in records if r['kind'] == 'base' and r['task'] == f'template_{t}'])) for t in range(len(cfg['templates']))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(dict(mean_hybrid=means, cells=cells), ensure_ascii=False))


if __name__ == '__main__':
    main()
