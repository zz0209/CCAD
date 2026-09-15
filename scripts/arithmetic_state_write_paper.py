"""Export exposed and frozen state-writer evidence without pooling populations."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/correspondence_reform_20260913'
CONDITIONS = ['same_answer_opposite_carry', 'same_carry_different_answer']


def main():
    analyses = {key: json.loads((ART / name).read_text()) for key, name in [
        ('development2', 'r54_state_2.json'), ('development3', 'r54_state_3.json'),
        ('confirmation3', 'r54_state_confirmation.json')]}
    costs = []; seen = set()
    for data in analyses.values():
        run = ROOT / data['run']
        assert hashlib.sha256((run / 'metrics.raw.jsonl').read_bytes()).hexdigest() == data['raw_sha256']
        if run in seen:
            continue
        seen.add(run)
        summary = json.loads((run / 'metrics.summary.json').read_text())
        assert summary['status'] == 'PASS'
        costs.append(dict(run=run.relative_to(ROOT).as_posix(),
            status=json.loads((run / 'status.json').read_text()),
            **{k: summary[k] for k in ['wall_seconds', 'process_cpu_seconds', 'peak_allocated_bytes',
                                     'rows', 'sequence_forwards', 'token_forwards']}))
    source = ROOT / analyses['development3']['run']
    confirm = ROOT / analyses['confirmation3']['run']
    def reference_records(run, method):
        records = [json.loads(line) for line in (run / 'metrics.raw.jsonl').read_text().splitlines()]
        fields = ['recipient_question', 'donor_question', 'expected', 'answer', 'correct', 'unit_preserved']
        return {(r['operation'], tuple(r['recipient_question']), tuple(r['donor_question']), r['task']):
                [r[k] for k in fields] for r in records
                if r['kind'] == 'rule_intervention' and r['method'] == method}
    old = ROOT / 'runs/REFORM_R53_teacher_write_source_v1_20260915'
    for method in ['readwrite_code_64', 'code_read_raw_write']:
        records = reference_records(source, method)
        assert len(records) == 512 and records == reference_records(old, method)
    panel = json.loads((confirm / 'panel.json').read_text())
    triples = {tuple(r[k] for k in ['a', 'b', 'c']) for r in panel['rows']}
    fit = json.loads((source / 'SOURCE_FIT_PANEL.json').read_text())
    fit_triples = {tuple(r[k] for k in ['a', 'b', 'c']) for r in fit['rows'] if 'c' in r}
    assert len(triples) == 256 and not triples & fit_triples
    fits = [json.loads(p.read_text()) for p in sorted(source.glob('state_*_fit.json'))]
    assert len(fits) == 6 and all(d['steps'] == 256 for d in fits)
    names = [('noop', 'No edit'), ('readwrite_code_64', 'Prior native 64'),
             ('code_read_raw_write', 'Prior code/raw')]
    names += [(f'state_{mode}_{space}', f'{space.capitalize()} {mode}')
              for space in ['code', 'raw'] for mode in ['constant', 'scalar', 'direction']]
    rows = []
    for method, label in names:
        row = dict(method=method, label=label)
        for key, data in analyses.items():
            for condition, suffix in zip(CONDITIONS, ['change', 'preserve']):
                cell = next((c for c in data['cells'] if c['method'] == method and c['condition'] == condition), None)
                row[key + '_' + suffix] = 100 * cell['success']['mean'] if cell else None
        rows.append(row)
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), table=rows, analyses=analyses,
        fits=fits, costs=costs, new_triples=len(triples), repeated_operand_triples=sum(len(set(q)) < 3 for q in triples),
        source_fit_triples=len(fit_triples), source_fit_overlap=len(triples & fit_triples),
        reference_replay='Both prior native and code/raw request and discrete outcome records match all 512 exposed R53 requests.',
        total_driver_seconds=sum(c['wall_seconds'] for c in costs),
        total_process_cpu_seconds=sum(c['process_cpu_seconds'] for c in costs),
        scope='Source1; exposed development and frozen new questions with expanded operand/multiplicity range are separate populations. No new cross-seed result.')
    for path in [ART / 'R54_COMBINED_RESULTS.json', ROOT / 'paper/data/arithmetic_state_write.json']:
        path.write_text(json.dumps(result, indent=2) + '\n')
    with (ROOT / 'paper/data/arithmetic_state_write.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    tex = [r'\begin{table}[t]', r'\centering\small', r'\begin{tabular}{lrrrrrr}', r'\toprule',
           r'& \multicolumn{2}{c}{Two: exposed} & \multicolumn{2}{c}{Three: exposed} & \multicolumn{2}{c}{Three: new} \\',
           r'Source writer & Change & Preserve & Change & Preserve & Change & Preserve \\', r'\midrule']
    for i, row in enumerate(rows):
        if i in [3, 6]: tex.append(r'\midrule')
        values = [row[key + '_' + suffix] for key in analyses for suffix in ['change', 'preserve']]
        tex.append(row['label'] + ' & ' + ' & '.join('--' if v is None else f'{v:.2f}' for v in values) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}',
        r'\caption{State-conditioned source writing, complete-answer success (\%). Six new writers share the frozen 64-code reader, eight state coordinates and 256 additional source-fit batches. Native writes use the same 64 members; raw writes use 1,536 coordinates. New three-operand questions include a wider operand range and repeated operands; methods are frozen before evaluation. All requests remain in the denominator.}',
        r'\label{tab:state_write}', r'\end{table}']
    (ROOT / 'paper/tables/arithmetic_state_write.tex').write_text('\n'.join(tex) + '\n')
    print(json.dumps({k: result[k] for k in ['costs', 'new_triples', 'repeated_operand_triples', 'source_fit_overlap', 'total_driver_seconds', 'total_process_cpu_seconds']}))


if __name__ == '__main__':
    main()
