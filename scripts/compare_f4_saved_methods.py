"""Match an added method to saved causal records without new model forwards."""
import argparse
import csv
import json
from pathlib import Path

from compare_f4_single_atom import digest, key, median, read_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--previous-method-run', type=Path)
    args = parser.parse_args()
    run = args.run_dir
    cfg = json.loads((run / 'config.resolved.json').read_text())
    other = run.parent / cfg['comparison_run']
    for folder in (run, other):
        assert json.loads((folder / 'status.json').read_text())['status'] == 'PASS'
    assert json.loads((run / 'selection.json').read_text()) == json.loads((other / 'selection.json').read_text())
    added = read_rows(run / 'metrics.raw.jsonl')
    saved = read_rows(other / 'metrics.raw.jsonl')
    method_names = sorted({r['method'] for r in added})
    assert len(method_names) == 1, 'This comparison expects one added method'
    method = method_names[0]
    targets = {key(r): r for r in saved if r['method'] == 'target'}
    assert len(targets) == len(added) == len({key(r) for r in added})
    assert method not in {r['method'] for r in saved}
    for a in added:
        d = targets[key(a)]
        for field in ('common_source_dose_scale', 'source_natural_hook_energy',
                      'recipient_masked_hook_norm', 'source_hook_fraction',
                      'intervention_positions', 'donor_sequence', 'donor_positions',
                      'document_ids', 'donor_document_ids'):
            assert a[field] == d[field], (key(a), field)
        assert a['hook']['source_energy'] == d['hook']['source_energy']
        for endpoint in a['endpoints']:
            for field in ('source_energy', 'source_rms'):
                assert a['endpoints'][endpoint][field] == d['endpoints'][endpoint][field], (key(a), endpoint, field)
    methods = sorted({r['method'] for r in saved + added})
    for name in methods:
        rows = [r for r in saved + added if r['method'] == name]
        assert len(rows) == len(targets) == len({key(r) for r in rows})
        assert {key(r) for r in rows} == set(targets)
    tables = {}
    for level, fields in [('target', ('condition', 'source_seed', 'source_atom', 'target_seed', 'rank')),
                          ('query', ('condition', 'source_seed', 'source_atom', 'rank'))]:
        groups = {}
        for row in saved + added:
            for endpoint in ('next_state', 'centered_logits'):
                group = tuple(row[k] for k in fields) + (endpoint,)
                groups.setdefault(group, {}).setdefault(row['method'], []).append(row['endpoints'][endpoint]['normalized_error'])
        table = []
        for group, values in sorted(groups.items()):
            medians = {name: median(values[name]) for name in methods}
            record = dict(zip(fields + ('endpoint',), group))
            record.update({name + '_error': medians[name] for name in methods})
            for name in methods:
                record[name + '_valid_pairs'] = sum(v is not None for v in values[name])
            record['requested_pairs_per_method'] = len(values[method])
            dist, comparator = medians['target'], medians[method]
            record['distributed_better_than_added'] = dist < comparator if dist is not None and comparator is not None else None
            record['relative_error_reduction_vs_added'] = 1 - dist / comparator if dist is not None and comparator is not None and comparator > 0 else None
            table.append(record)
        tables[level] = table
        with (run / f'method_comparison_{level}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    summary = {}
    for condition in ('positive', 'negative'):
        for endpoint in ('next_state', 'centered_logits'):
            rows = [r for r in tables['query'] if r['condition'] == condition and r['endpoint'] == endpoint]
            summary[f'{condition}/{endpoint}'] = {
                'requested_queries': len(rows),
                'valid_queries': sum(r['distributed_better_than_added'] is not None for r in rows),
                'method_query_medians': {name: median(r[name + '_error'] for r in rows) for name in methods},
                'distributed_better_than_added_queries': sum(r['distributed_better_than_added'] is True for r in rows),
                'below_zero_queries': {name: sum(r[name + '_error'] is not None and r[name + '_error'] < 1 for r in rows) for name in methods}}
    report = {
        'added_method': method, 'matched_rows': len(added),
        'source_replay': 'EXACT selection, documents, donor positions, source dose/energy and endpoint energy/RMS',
        'summary': summary, 'query_table': tables['query'],
        'source_artifacts': {str(p): digest(p) for p in [run / 'metrics.raw.jsonl', other / 'metrics.raw.jsonl', run / 'selection.json', run / 'config.resolved.json', run / 'ot_fits.json', run / 'ot_fits.npz']},
        'analysis_code': {str(p): digest(p) for p in [Path(__file__), Path(__file__).with_name('compare_f4_single_atom.py')]},
        'scope': 'All8 development queries and all4 dependent target seeds retained, query medians pool target/document rows; no independent confidence interval. Same source rank and dose, not candidate-energy-matched or behavior-trained OT. Not Semantic OT, SCOTM or MAS replication. No native-deletion or uniqueness claim.'}
    if (run / 'ot_tuning.json').exists():
        report['source_artifacts'][str(run / 'ot_tuning.json')] = digest(run / 'ot_tuning.json')
    if args.previous_method_run:
        previous = args.previous_method_run
        assert json.loads((previous / 'status.json').read_text())['status'] == 'PASS'
        assert json.loads((previous / 'selection.json').read_text()) == json.loads((run / 'selection.json').read_text())
        old_rows = read_rows(previous / 'metrics.raw.jsonl')
        old = {key(r): r for r in old_rows if r['method'] == method}
        assert len(old) == len(old_rows) == len(added)
        changes = {}
        for row in added:
            before = old[key(row)]
            for field in ('common_source_dose_scale', 'source_natural_hook_energy', 'donor_positions', 'intervention_positions'):
                assert before[field] == row[field]
            for endpoint in row['endpoints']:
                assert before['endpoints'][endpoint]['source_energy'] == row['endpoints'][endpoint]['source_energy']
            for endpoint in ('next_state', 'centered_logits'):
                group = tuple(row[k] for k in ('condition', 'source_seed', 'source_atom', 'rank')) + (endpoint,)
                changes.setdefault(group, []).append((before['endpoints'][endpoint]['normalized_error'], row['endpoints'][endpoint]['normalized_error']))
        change_table = []
        for group, values in sorted(changes.items()):
            before = median(v[0] for v in values); after = median(v[1] for v in values)
            change_table.append({**dict(zip(('condition', 'source_seed', 'source_atom', 'rank', 'endpoint'), group)),
                'previous_error': before, 'current_error': after,
                'current_better': after < before if before is not None and after is not None else None,
                'relative_error_reduction': 1 - after / before if before is not None and before > 0 and after is not None else None,
                'valid_pairs': sum(a is not None and b is not None for a, b in values), 'requested_pairs': len(values)})
        with (run / 'method_change_query.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(change_table[0])); writer.writeheader(); writer.writerows(change_table)
        report['previous_method_comparison'] = change_table
        report['source_artifacts'][str(previous / 'metrics.raw.jsonl')] = digest(previous / 'metrics.raw.jsonl')
    (run / 'method_comparison.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'matched_rows': len(added), 'summary': summary, 'query_table': tables['query']}, indent=2))


if __name__ == '__main__':
    main()
