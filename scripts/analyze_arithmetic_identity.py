"""Compare functional identity and compute-matched attribution on existing pairs.

Descriptive development analysis; preserves operand-pair, prompt and seed axes.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import collections
import hashlib
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--prior', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    panels = [json.loads((p / 'panel.json').read_text()) for p in [args.prior, args.run]]
    assert panels[0]['rows'] == panels[1]['rows'] and panels[0]['pairs'] == panels[1]['pairs']
    panel = panels[0]
    rr, provenance, costs = [], [], []
    for run in [args.prior, args.run]:
        status = json.loads((run / 'status.json').read_text())
        assert status['status'] == 'PASS'
        raw = run / 'metrics.raw.jsonl'
        provenance.append(dict(run=str(run), status=status,
                               raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest()))
        rr.extend(r for r in map(json.loads, raw.read_text().splitlines())
                  if r['kind'] == 'source_patch' and r['method'].startswith('adapt_'))
        for path in sorted(run.glob('counterfactual_seed*_k64_*.json')):
            fit = json.loads(path.read_text())
            costs.append(dict(run=str(run), file=path.name, seed=fit['seed'],
                              method=fit['initialization'],
                              backward_budget=fit.get('backward_budget', dict(selection=0,
                                  optimization=fit['updates'], total=fit['updates'], numerical_check=1)),
                              fit_elapsed_seconds=fit['trace'][-1]['fit_elapsed_seconds']))
    lookup = {}
    for r in rr:
        key = r['method'], r['seed'], r['operation'], r['row_id']
        assert key not in lookup, key
        lookup[key] = r
    methods = sorted({r['method'] for r in rr})
    cells = []
    for method in methods:
        for op in ['unit', 'tens']:
            rows = [r for r in rr if r['method'] == method and r['operation'] == op]
            assert len(rows) == 5 * len(panel['pairs'])
            cats = collections.Counter()
            seed_means = {}
            for r in rows:
                pair = panel['pairs'][r['row_id']]
                category = next((name for name in ['base', 'unit', 'tens', 'donor']
                                 if r['answer'] == pair[name + '_answer']), 'other')
                cats[category] += 1
            for seed in range(1, 6):
                ss = [r for r in rows if r['seed'] == seed]
                seed_means[seed] = float(np.mean([r['exact_hybrid'] for r in ss]))
            cells.append(dict(method=method, operation=op, n=len(rows),
                              hybrid=float(np.mean([r['exact_hybrid'] for r in rows])),
                              target=float(np.mean([r['target_digit_success'] for r in rows])),
                              preserve=float(np.mean([r['preserve_digit_success'] for r in rows])),
                              norm=float(np.mean([r['edit_norm'] for r in rows])),
                              seed_hybrid=seed_means,
                              outcomes={c: cats[c] / len(rows) for c in ['base','unit','tens','donor','other']}))
    examples = []
    # Show both directions of disagreement using the first pair in the saved order.
    for other in ['target_gradient', 'target_integrated_gradient', 'clean_swapped', 'clean_fixed']:
        for op in ['unit', 'tens']:
            for clean_wins in [True, False]:
                found = None
                for pair_id, pair in enumerate(panel['pairs']):
                    for seed in range(1, 6):
                        a = lookup['adapt_clean_u64', seed, op, pair_id]
                        b = lookup[f'adapt_{other}_u64', seed, op, pair_id]
                        if a['exact_hybrid'] == clean_wins and b['exact_hybrid'] != clean_wins:
                            found = dict(comparator=other, operation=op, clean_wins=clean_wins,
                                         seed=seed, pair_id=pair_id,
                                         recipient=panel['rows'][pair['recipient']],
                                         donor=panel['rows'][pair['donor']],
                                         desired=pair[op+'_answer'],
                                         clean_answer=a['answer'], comparator_answer=b['answer'])
                            break
                    if found is not None:
                        break
                if found is not None:
                    examples.append(found)
    checks = []
    for seed in range(1, 6):
        schedules = []
        for name in ['target_gradient','target_integrated_gradient','clean_swapped','clean_fixed']:
            with np.load(args.run / f'counterfactual_seed{seed}_k64_{name}.npz') as data:
                schedules.append(data['fit_schedule'])
        assert all(np.array_equal(schedules[0], s) for s in schedules[1:])
        checks.append(dict(seed=seed, same_fit_schedule=True))
    out = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
               scope='Descriptive development; already exposed R32 pairs. Repeated prompts and shared seeds are not independent replications.',
               provenance=provenance, cells=cells, costs=costs,
               focused_checks=checks, examples=examples)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'identity.json').write_text(json.dumps(out, indent=2)+'\n')
    for c in cells:
        if c['method'].endswith('u64'):
            print(json.dumps({k:c[k] for k in ['method','operation','hybrid','target','preserve','seed_hybrid']}))


if __name__ == '__main__':
    main()
