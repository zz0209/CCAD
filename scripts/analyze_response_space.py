"""Measure intervention reuse across request and readout families.

This analysis uses existing exposed development contexts. Future-task heads
are evaluation-only and never enter the dictionary or relation fitting.
"""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference-run', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = json.loads((args.run/'config.resolved.json').read_text())
    inputs = []

    def read_array(path, key=None):
        path = Path(path)
        inputs.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        arr = np.load(path)
        return np.asarray(arr if key is None else arr[key], dtype=np.float64)

    old = read_array(Path(cfg['frozen_source_run'])/'probe.npz', 'weight').ravel()
    head_dir = Path('runs/IR04_shift_consumer_seed2_v1_20260916')
    heads = {'old': old}
    for task in ['composer_surgeon_orientation0', 'composer_surgeon_orientation1',
                 'model_software_engineer_orientation0', 'model_software_engineer_orientation1']:
        heads[task] = read_array(head_dir/f'none__full__{task}__probe42.npz', 'weight').ravel()
    weights = np.stack(list(heads.values()), axis=1)
    clean = read_array(args.run/'none__full__pooled.npy')
    queries = cfg['queries']
    families = {
        'single_parts': ['pronouns', 'names', 'associated_words'],
        'full_and_pairs': [q for q in queries if q == 'full' or '+' in q],
        'fractional_requests': list(cfg['dose_queries']),
    }
    methods = list(dict.fromkeys(['initial','head_parts']+cfg['variants']+cfg.get('evaluate_baselines',[])))
    cells, arrays = [], {}
    for query in queries:
        source = read_array(args.run/f'source__{query}__pooled.npy')
        effect = source-clean
        denominator = np.mean((effect@weights)**2, axis=0)
        assert np.all(denominator > 1e-12)
        for method in methods:
            target_path=args.run/f'{method}__{query}__pooled.npy'
            if not target_path.exists() and method=='head_parts' and args.reference_run:
                ref_cfg=json.loads((args.reference_run/'config.resolved.json').read_text())
                assert ref_cfg['target_seed']==cfg['target_seed']
                reference_membership=json.loads((args.reference_run/'evaluation_membership.json').read_text())
                current_membership=json.loads((args.run/'evaluation_membership.json').read_text())
                assert reference_membership['rows']==current_membership['rows']
                reference_source=read_array(args.reference_run/f'source__{query}__pooled.npy')
                assert np.allclose(reference_source,source,atol=2e-5,rtol=1e-5)
                target_path=args.reference_run/f'{method}__{query}__pooled.npy'
            target = read_array(target_path)
            error = target-source
            values = np.sqrt(np.mean((error@weights)**2, axis=0)/denominator)
            for index, head in enumerate(heads):
                cells.append(dict(method=method, request=query, head=head,
                                  response_nrmse=float(values[index]),
                                  source_rms=float(np.sqrt(denominator[index]))))
            cells.append(dict(method=method, request=query, head='pooled',
                              response_nrmse=float(np.linalg.norm(error)/np.linalg.norm(effect)),
                              source_rms=float(np.sqrt(np.mean(effect**2)))))
            arrays[method, query] = ((error@weights)**2, (effect@weights)**2)
    summary = []
    for method in methods:
        for family, requested in families.items():
            for head_set, selected in [('old', ['old']), ('future_heads', list(heads)[1:]), ('pooled', ['pooled'])]:
                values = [r['response_nrmse'] for r in cells if r['method'] == method
                          and r['request'] in requested and r['head'] in selected]
                summary.append(dict(method=method, requests=family, readouts=head_set,
                                    mean_nrmse=float(np.mean(values))))
    # Paired context resampling. Shared heads and requests stay grouped;
    # this interval describes these fixed heads and this one development seed.
    rng = np.random.default_rng(20260919)
    bootstrap = rng.integers(0, len(clean), size=(2000, len(clean)))
    comparisons = []
    for method in [m for m in cfg['variants'] if m!='head_parts']:
        for family, requested in families.items():
            for label, indices in [('old', [0]), ('future_heads', [1, 2, 3, 4])]:
                replicate = []
                for m in [method, 'head_parts']:
                    scores = []
                    for q in requested:
                        numerator, denominator = arrays[m, q]
                        scores.append(np.sqrt(numerator[bootstrap][:, :, indices].mean(1)/
                                              denominator[bootstrap][:, :, indices].mean(1)).mean(1))
                    replicate.append(np.mean(scores, axis=0))
                difference = replicate[0]-replicate[1]
                comparisons.append(dict(method=method, reference='head_parts', requests=family,
                                        readouts=label, delta=float(np.mean(difference)),
                                        interval=np.quantile(difference, [.025, .975]).tolist()))
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
                  evidence='Exposed development; fixed source, one target seed and four evaluation-only task heads',
                  intervals='2000 paired context resamples; no task or seed population inference',
                  inputs=inputs, contexts=len(clean), cells=cells, summary=summary, comparisons=comparisons)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
