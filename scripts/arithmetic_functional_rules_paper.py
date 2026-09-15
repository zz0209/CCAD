"""Export source-rule selection and cross-seed reuse with their strong controls."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/correspondence_reform_20260913'
PAPER = ROOT / 'paper'
CONDITIONS = ['same_answer_opposite_carry', 'same_carry_different_answer']
LABELS = {
    'noop': 'No edit', 'whole_state': 'Whole state',
    'raw_carry_direction': 'Raw carry direction',
    'conditional_carry_64': 'Conditional association, 64',
    'conditional_carry_256': 'Conditional association, 256',
    'unconditional_carry_64': 'Unconditional association, 64',
    'answer_tens_64': 'Answer-tens association, 64',
    'random_active_64': 'Random active members, 64',
    'field_ranked_binary_64': 'Field score, 64',
    'field_fitted_weighted_64': 'Field fit, weighted',
    'field_fitted_binary_64': 'Field fit, binary support',
    'assignment_64': 'Source assignment',
    'transferred_fitted_weighted_64': 'Source-field transfer, weighted',
    'transferred_fitted_binary_64': 'Source-field transfer, binary support',
}


def identity(path):
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    analyses = {name: json.loads((ART / f'r49_{name}.json').read_text())
                for name in ['carry_source', 'carry_participation', 'carry_relation']}
    costs = []
    for analysis in analyses.values():
        run = ROOT / analysis['run']
        assert analysis['status']['status'] == 'PASS'
        assert identity(run / 'metrics.raw.jsonl')['sha256'] == analysis['raw_sha256']
        summary = json.loads((run / 'metrics.summary.json').read_text())
        costs.append(dict(run=analysis['run'], status=analysis['status'], **{
            k: summary[k] for k in ['wall_seconds', 'process_cpu_seconds',
                'peak_allocated_bytes', 'sequence_forwards', 'token_forwards', 'rows']}))
    # The second source experiment repeats every original comparison exactly.
    for cell in analyses['carry_source']['cells']:
        repeated = next(c for c in analyses['carry_participation']['cells']
                        if (c['method'], c['condition']) == (cell['method'], cell['condition']))
        for metric in ['success', 'unit_preserved', 'original_retained']:
            assert cell[metric] == repeated[metric]
    table = []
    for stage, name in [('Source', 'carry_participation'), ('Target mean', 'carry_relation')]:
        data = analyses[name]
        for method, label in LABELS.items():
            cells = {c['condition']: c for c in data['cells'] if c['method'] == method}
            if not cells:
                continue
            table.append(dict(stage=stage, method=method, label=label,
                carry_change=100*cells[CONDITIONS[0]]['success']['mean'],
                same_carry_preservation=100*cells[CONDITIONS[1]]['success']['mean'],
                changing_carry_units=100*cells[CONDITIONS[0]]['unit_preserved']['mean'],
                same_carry_units=100*cells[CONDITIONS[1]]['unit_preserved']['mean']))
    result = dict(analyses=analyses, table=table, costs=costs,
        total_driver_seconds=sum(c['wall_seconds'] for c in costs),
        total_process_cpu_seconds=sum(c['process_cpu_seconds'] for c in costs),
        scope='Exposed development, adaptively selected source1 binary support then frozen for target seeds2-5. All directions share source1 and 64 canonical pairs. No independent confirmation.',
        source_selection_clarification='RULE_RESULTS source_selection describes the initial contrast only. The fitted support additionally uses source-fit field optimization; the final binary source was selected after source-development interventions, before target fitting. No target outcomes selected it.')
    for path in [ART / 'R49_COMBINED_RESULTS.json', PAPER / 'data/arithmetic_functional_rules.json']:
        path.write_text(json.dumps(result, indent=2)+'\n')
    with (PAPER / 'data/arithmetic_functional_rules.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    tex = [r'\begin{anchoredtable}\centering\small', r'\begin{tabular}{llrr}', r'\toprule',
           r'Stage & Method & Carry change & Same-carry preservation \\', r'\midrule']
    previous = None
    for row in table:
        if previous and previous != row['stage']:
            tex.append(r'\midrule')
        tex.append(f"{row['stage'] if previous != row['stage'] else ''} & {row['label']} & "
                   f"{row['carry_change']:.2f} & {row['same_carry_preservation']:.2f}"+r' \\')
        previous = row['stage']
    tex += [r'\bottomrule\end{tabular}',
        r'\caption{A functional carry rule separates an internal intervention from answer copying. Percentages on two development conditions, each with 32 canonical pairs, both orientations and two prompt forms. Carry change requires the predicted complete answer $s+10(c_d-c_r)$ when donor and recipient have the same sum. Same-carry preservation requires retaining the recipient answer when the sums differ. Source rows edit SAE1; target means edit SAEs2--5 using the frozen source group where indicated. Raw and direct field fits remain strong controls.}',
        r'\label{tab:arithmetic_carry_rule}\end{anchoredtable}']
    (PAPER / 'tables/arithmetic_functional_rules.tex').write_text('\n'.join(tex)+'\n')
    tex = [r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrrrr}',r'\toprule',
           r'Method & Target2 & Target3 & Target4 & Target5 \\',r'\midrule']
    for method in ['assignment_64', 'field_fitted_binary_64', 'transferred_fitted_weighted_64',
                   'transferred_fitted_binary_64']:
        values = []
        for seed in [2, 3, 4, 5]:
            cells = {c['condition']: c for c in analyses['carry_relation']['per_seed']
                     if c['method'] == method and c['seed'] == seed}
            values.append('/'.join(f"{100*cells[c]['success']:.2f}" for c in CONDITIONS))
        tex.append(LABELS[method]+' & '+' & '.join(values)+r' \\')
    tex += [r'\bottomrule\end{tabular}',
        r'\caption{Carry-change success / same-carry preservation (percent) for each target SAE. These directions share one source and the same questions; they are kept together in the paired bootstrap. All listed methods use 64 members.}',
        r'\label{tab:arithmetic_carry_seeds}\end{anchoredtable}']
    (PAPER / 'tables/arithmetic_functional_rules_seeds.tex').write_text('\n'.join(tex)+'\n')
    print(json.dumps(dict(table=table, costs=costs)))


if __name__ == '__main__':
    main()
