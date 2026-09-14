"""Execute the frozen paired analysis of arithmetic component reuse."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    run = args.run
    assert json.loads((run / 'status.json').read_text())['status'] == 'PASS'
    cfg = json.loads((run / 'config.resolved.json').read_text())
    spec = cfg['confirmation']
    panel = json.loads((run / 'panel.json').read_text())
    raw = run / 'metrics.raw.jsonl'
    raw_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
    assert raw_hash == json.loads((run / 'metrics.summary.json').read_text())['metrics_raw_sha256']
    methods = ['clean'] + spec['comparators']
    updates = cfg['frozen_adaptation']['updates']
    seeds = cfg['seeds']
    metrics = ['exact_hybrid', 'target_digit_success', 'preserve_digit_success']

    def cluster(pair):
        return tuple(panel['rows'][i][k] for i in [pair['recipient'], pair['donor']] for k in ['a', 'b'])

    identities = sorted({cluster(p) for p in panel['pairs']})
    assert len(identities) == cfg['pairs_per_template']
    index = {key: i for i, key in enumerate(identities)}
    shape = (len(methods), len(updates), len(identities), 2, 2, len(seeds), len(metrics))
    outcomes = np.full(shape, np.nan)
    loaded = 0
    for line in raw.read_text().splitlines():
        r = json.loads(line)
        if r['kind'] != 'source_patch' or not r['method'].startswith('adapt_'):
            continue
        method, count = r['method'][6:].rsplit('_u', 1)
        pair = panel['pairs'][r['row_id']]
        key = (methods.index(method), updates.index(int(count)), index[cluster(pair)],
               ['unit', 'tens'].index(r['operation']), pair['template'], seeds.index(r['seed']))
        assert np.isnan(outcomes[key]).all(), ('duplicate', key)
        outcomes[key] = [float(r[m]) for m in metrics]
        loaded += 1
    assert np.isfinite(outcomes).all()
    assert loaded == int(np.prod(shape[:-1]))

    # All sources, requests and prompt renderings remain in the same resample.
    rng = np.random.default_rng(spec['bootstrap_seed'])
    draws = rng.integers(len(identities), size=(spec['bootstrap_replicates'], len(identities)))
    primary = []
    u = updates.index(spec['primary_updates'])
    for comparator in spec['comparators']:
        diff = (outcomes[0, u, ..., 0] - outcomes[methods.index(comparator), u, ..., 0]).mean(axis=(1, 2, 3))
        boot = diff[draws].mean(1)
        lo, hi = np.quantile(boot, [.025, .975])
        primary.append(dict(comparator=comparator, difference_points=100 * float(diff.mean()),
                            interval_points=[100 * float(lo), 100 * float(hi)],
                            positive_lower_bound=bool(lo > 0)))
    cells = []
    for mi, method in enumerate(methods):
        for ui, count in enumerate(updates):
            for oi, operation in enumerate(['unit', 'tens']):
                a = outcomes[mi, ui, :, oi]
                cells.append(dict(initialization=method, updates=count, operation=operation,
                                  **{m: float(a[..., k].mean()) for k, m in enumerate(metrics)},
                                  per_seed={str(s): float(a[:, :, si, 0].mean()) for si, s in enumerate(seeds)}))
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), run=str(run.resolve()),
                  raw_sha256=raw_hash, config_sha256=hashlib.sha256((run / 'config.resolved.json').read_bytes()).hexdigest(),
                  distinct_operand_pair_clusters=len(identities), analyzed_rows=loaded,
                  spec=spec, primary=primary,
                  both_primary_comparisons_positive=all(x['positive_lower_bound'] for x in primary),
                  cells=cells, scope=cfg['scope'])
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'cluster_outcomes.npz', outcomes=outcomes, operand_pairs=np.array(identities),
                        methods=np.array(methods), updates=np.array(updates), seeds=np.array(seeds), metrics=np.array(metrics))
    (args.output / 'confirmation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['distinct_operand_pair_clusters', 'analyzed_rows', 'primary', 'both_primary_comparisons_positive']}))


if __name__ == '__main__':
    main()
