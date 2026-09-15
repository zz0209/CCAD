"""Export both operation endpoints for absolute and state-dependent components."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/correspondence_reform_20260913'
PAPER = ROOT / 'paper'


def identity(path, base=ROOT):
    return dict(path=path.relative_to(base).as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    analyses = {name: json.loads((ART / f'r48_{name}_pilot.json').read_text())
                for name in ['absolute_membership', 'state_components']}
    costs = []
    for analysis in analyses.values():
        run = ROOT / analysis['run']
        assert analysis['status']['status'] == 'PASS'
        assert identity(run / 'metrics.raw.jsonl')['sha256'] == analysis['raw_sha256']
        summary = json.loads((run / 'metrics.summary.json').read_text())
        costs.append(dict(run=analysis['run'], status=analysis['status'], **{
            k: summary[k] for k in ['wall_seconds', 'process_cpu_seconds',
                'peak_allocated_bytes', 'sequence_forwards', 'token_forwards', 'rows']}))
    static = analyses['absolute_membership']; dynamic = analyses['state_components']
    families = [('Source', 'source', static), ('Initial membership', 'initial', static),
                ('Difference fit', 'difference', static), ('Absolute fit', 'absolute', static),
                ('Joint fit', 'joint', static), ('State component, exact source', 'anchor_exact', dynamic),
                ('State component, code reader', 'anchor_code', dynamic),
                ('State component, raw reader', 'anchor_raw', dynamic),
                ('Code reader, source field', 'readout_code', dynamic),
                ('Raw reader, source field', 'readout_raw', dynamic),
                ('Selected gain', 'gain', static), ('Direct target selection', 'direct', static),
                ('Assignment', 'assignment', static)]
    table = []
    for label, family, analysis in families:
        cells = {op: next(c for c in analysis['cells'] if c['family'] == family
                          and c['kind'] == op and c['sites'] == 'all')
                 for op in ['replace', 'delete']}
        table.append(dict(method=label, family=family, run=analysis['run'],
            replacement_complete=100*cells['replace']['complete_expected_answer_pair']['mean'],
            deletion_source_answer=100*cells['delete']['requested_source_answer']['mean'],
            deletion_original_answer=100*cells['delete']['requested_expected_answer']['mean'],
            deletion_protected_answer=100*cells['delete']['protected_expected_answer']['mean']))
    # Independently run common methods must retain identical discrete endpoints.
    for family in ['source', 'absolute', 'direct', 'gain']:
        for op in ['replace', 'delete']:
            cc = [next(c for c in x['cells'] if c['family'] == family and c['kind'] == op
                       and c['sites'] == 'all') for x in [static, dynamic]]
            for metric in ['complete_expected_answer_pair', 'requested_source_answer']:
                assert cc[0][metric]['mean'] == cc[1][metric]['mean'], (family, op, metric)
    result = dict(analyses=analyses, table=table, costs=costs,
                  total_driver_seconds=sum(c['wall_seconds'] for c in costs),
                  total_process_cpu_seconds=sum(c['process_cpu_seconds'] for c in costs),
                  scope='Two exposed development pilots; same 64 contexts, two forms, source1 to target2. No independent confirmation. Static loss objectives and dynamic components use source-fit states only. Native variants use target amplitudes plus retained source queries; exact-source and gain-deletion comparators receive exact source amplitudes.')
    for path in [ART / 'R48_COMBINED_RESULTS.json', PAPER / 'data/binding_component_objectives.json']:
        path.write_text(json.dumps(result, indent=2)+'\n')
    with (PAPER / 'data/binding_component_objectives.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    tex = [r'\begin{anchoredtable}\centering\small', r'\begin{tabular}{lrrrr}', r'\toprule',
           r' & Replacement & \multicolumn{3}{c}{Deletion} \\',
           r'Method & Complete pair & Source answer & Original answer & Protected answer \\', r'\midrule']
    for row in table:
        values = ' & '.join(f'{row[k]:.2f}' for k in ['replacement_complete',
                            'deletion_source_answer', 'deletion_original_answer', 'deletion_protected_answer'])
        tex.append(row['method']+' & '+values+r' \\')
    tex += [r'\bottomrule\end{tabular}',
            r'\caption{Fitting a component for replacement and deletion. All values are percentages on the exposed 64-context panel, with two forms and one source--target direction. Replacement requires both expected city answers; deletion agreement compares requested answers with the actual source ablation. Original and protected answers describe the deletion effect. Direct selection uses the full target dictionary; the learned native components use a fixed 256-member bank. Every native edit changes at most 64 codes. Source-field readers are unrestricted. The source reference and common discrete endpoints agree across both pilots.}',
            r'\label{tab:binding_component_objectives}\end{anchoredtable}']
    (PAPER / 'tables/binding_component_objectives.tex').write_text('\n'.join(tex)+'\n')
    print(json.dumps(dict(table=table, total_driver_seconds=result['total_driver_seconds'])))


if __name__ == '__main__':
    main()
