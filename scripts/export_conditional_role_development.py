import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    args = parser.parse_args()
    names = ['HELD_DEVELOPMENT_ANALYSIS.json', 'HELD_MEMBERS_DEVELOPMENT_ANALYSIS.json',
             'HELD_CONTRAST_DEVELOPMENT_ANALYSIS.json', 'HELD_PROGRAM_DEVELOPMENT_ANALYSIS.json']
    results, seen = [], set()
    reference_index, reference_values = None, None
    for name in names:
        analysis = json.loads((args.artifacts/name).read_text())
        assert len(analysis['questions']) == 99
        for target in analysis['targets']:
            run = Path(target['run'])
            index = json.loads((run/'BANK_INDEX.json').read_text())
            values = np.load(run/'responses.npz')['logits']
            common = np.stack([values[index['methods'].index(k)] for k in ['none', 'source']])
            if reference_values is None:
                reference_values, reference_index = common, index
            assert index['queries'] == reference_index['queries']
            assert index['documents'] == reference_index['documents']
            np.testing.assert_array_equal(common, reference_values)
            for method, metric in target['methods'].items():
                key = (str(run), method)
                if key in seen:
                    continue
                seen.add(key)
                results.append(dict(run=run.name, method=method,
                    response_nrmse=metric['response_nrmse'], rank_mean=metric['rank_mean'],
                    conditional_utility=metric['policies']['conditional']['mean_utility'],
                    global_utility=metric['policies']['global']['mean_utility'],
                    standalone_utility=metric['policies']['standalone']['mean_utility'],
                    oracle_utility=metric['oracle_utility'], source_file=name))
    with (args.artifacts/'HELD_COMPARISON.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    inventory = []
    for run in sorted(p for p in args.runs.iterdir() if p.is_dir()):
        status = json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
        summary = json.loads((run/'metrics.summary.json').read_text()) if (run/'metrics.summary.json').exists() else {}
        progress = json.loads((run/'progress.json').read_text()) if (run/'progress.json').exists() else {}
        effective = status.get('status', 'FAIL_INITIALIZATION')
        if effective == 'RUNNING':
            effective = 'FAIL_FINALIZATION'
        inventory.append(dict(run=run.name, recorded_status=status.get('status'), status=effective,
            started_utc=status.get('started_at_utc'), ended_utc=status.get('ended_at_utc'),
            driver_seconds=summary.get('wall_seconds'), partial_elapsed_seconds=progress.get('elapsed'),
            peak_allocated_bytes=summary.get('peak_allocated_bytes'),
            bytes=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())))
    gating_run = args.runs/'RG05_gating_development_seed2_v1_20260921'
    gating = json.loads((gating_run/'GATING_ANALYSIS.json').read_text())
    source = next(r for r in results if r['method'] == 'source')
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        evidence='Exposed development; one target,99 held pairs,32 source/evaluation documents.',
        equal_documents_queries_clean_source=True, comparisons=results,
        standalone_fraction_of_source_oracle=source['standalone_utility']/source['oracle_utility'],
        gating_actual_nrmse=gating['actual_change_nrmse'],
        gating_fixed_mask_nrmse=gating['fixed_gate_change_nrmse'],
        gating_squared_error_reduction=1-gating['sums']['fixed_gate_error']/gating['sums']['actual_error'],
        runs=inventory, known_completed_driver_seconds=sum(r['driver_seconds'] or 0 for r in inventory),
        unfinished_receipt_elapsed_seconds=sum(r['partial_elapsed_seconds'] or 0 for r in inventory
                                              if r['driver_seconds'] is None),
        bulk_bytes=sum(r['bytes'] for r in inventory))
    with (args.artifacts/'DEVELOPMENT_SUMMARY.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k not in ['comparisons', 'runs']}))
    print(json.dumps({r['run']: r['status'] for r in inventory}))


if __name__ == '__main__':
    main()
