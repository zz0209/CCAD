"""Export carry-rule generalization, source refitting and inherited outcomes."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/correspondence_reform_20260913'
PAPER = ROOT / 'paper'
CONDITIONS = ['same_answer_opposite_carry', 'same_carry_different_answer']
LABELS = {'noop': 'No edit', 'raw_carry_direction': 'Raw direction',
    'field_fitted_binary_64': 'Field fit, binary', 'field_fitted_weighted_64': 'Field fit, weighted',
    'function_ce_binary_64': 'Answer CE, binary', 'function_ce_weighted_64': 'Answer CE, weighted',
    'assignment_64': 'Assignment', 'transferred_fitted_binary_64': 'Source transfer, binary',
    'transferred_fitted_weighted_64': 'Source transfer, weighted'}


def identity(p):
    return dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def analyze_outcomes(run):
    raw=[json.loads(s) for s in (run/'metrics.raw.jsonl').read_text().splitlines()]
    base={r['row_id']:r['correct'] for r in raw if r['kind']=='base'}
    source={r['row_id']:r for r in raw if r['kind']=='rule_intervention'
            and r['seed']==1 and r['method']=='field_fitted_binary_64'}
    outcome=[]
    for condition in CONDITIONS:
        target=[r for r in raw if r['kind']=='rule_intervention' and r['seed'] in [2,3,4,5]
                and r['method']=='transferred_fitted_binary_64' and r['operation']==condition]
        counts=dict(source_target_same_correct=0,source_target_same_wrong=0,
                    source_correct_target_wrong=0,source_wrong_target_correct=0,both_wrong_different=0)
        for r in target:
            s=source[r['row_id']]
            if s['answer']==r['answer']:
                key='source_target_same_correct' if r['correct'] else 'source_target_same_wrong'
            elif s['correct']:key='source_correct_target_wrong'
            elif r['correct']:key='source_wrong_target_correct'
            else:key='both_wrong_different'
            counts[key]+=1
        outcome.append(dict(condition=condition,rows=len(target),counts=counts))
    cases=[]
    for r in source.values():
        if sorted([r['recipient_carry'],r['donor_carry']])==[0,2] and base[r['row_id']] and base[r['donor_row']]:
            direction=1 if r['donor_carry']>r['recipient_carry'] else -1
            cases.append(dict(row_id=r['row_id'],donor_row=r['donor_row'],task=r['task'],
                recipient=r['recipient_question'],donor=r['donor_question'],
                original=r['original_answer'],expected=r['expected'],answer=r['answer'],correct=r['correct'],
                signed_change=None if r['answer'] is None else direction*(r['answer']-r['original_answer'])))
    diagnostic=dict(scope='Post-hoc source diagnostic restricted to base-correct donor and recipient; primary analyses retain every row.',
        rows=len(cases),predicted_two_tens=sum(c['correct'] for c in cases),
        observed_one_ten=sum(c['signed_change']==10 for c in cases),cases=cases)
    assert len(cases)==50 and diagnostic['predicted_two_tens']==2 and diagnostic['observed_one_ten']==39
    (ART/'R50_SOURCE_TARGET_OUTCOMES.json').write_text(json.dumps(outcome,indent=2)+'\n')
    (ART/'R50_CASES.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    return outcome,diagnostic


def main():
    names = ['frozen_source', 'frozen_targets', 'source_function_ce', 'source_function_ce_two_operand']
    analyses = {name: json.loads((ART / f'r50_{name}.json').read_text()) for name in names}
    costs = []
    for name in dict.fromkeys(a['run'] for a in analyses.values()):
        run = ROOT / name
        raw_hash = identity(run / 'metrics.raw.jsonl')['sha256']
        assert all(a['raw_sha256'] == raw_hash for a in analyses.values() if a['run'] == name)
        summary = json.loads((run / 'metrics.summary.json').read_text())
        costs.append(dict(run=name, **{k: summary[k] for k in ['status', 'wall_seconds',
            'process_cpu_seconds', 'peak_allocated_bytes', 'sequence_forwards', 'token_forwards', 'rows']}))
    failed = ROOT / 'runs/REFORM_R50_function_ce_carry_two_operand_eval_v1_20260915'
    summary = json.loads((failed / 'metrics.summary.json').read_text())
    costs.append(dict(run=failed.relative_to(ROOT).as_posix(), **{k: summary[k] for k in
        ['status', 'wall_seconds', 'process_cpu_seconds', 'peak_allocated_bytes', 'sequence_forwards', 'token_forwards', 'rows', 'error']}))
    # Refitting must not change any repeated baseline or old source result.
    lookup = {(c['method'], c['condition']): c for c in analyses['frozen_source']['cells']}
    for cell in analyses['source_function_ce']['cells']:
        if (cell['method'], cell['condition']) in lookup:
            old = lookup[cell['method'], cell['condition']]
            for metric in ['success', 'unit_preserved', 'original_retained']:
                assert cell[metric] == old[metric]
    prior=json.loads((ART/'r49_carry_participation.json').read_text())
    lookup={(c['method'],c['condition']):c for c in prior['cells']}
    for cell in analyses['source_function_ce_two_operand']['cells']:
        if (cell['method'],cell['condition']) in lookup:
            for metric in ['success','unit_preserved','original_retained']:
                assert cell[metric] == lookup[cell['method'],cell['condition']][metric]
    rows = []
    for name, analysis in analyses.items():
        for c in analysis['cells']:
            rows.append(dict(panel=name, method=c['method'], condition=c['condition'],
                success=100*c['success']['mean'], lower=100*c['success']['interval'][0],
                upper=100*c['success']['interval'][1], unit_preserved=100*c['unit_preserved']['mean']))
    outcome,diagnostic = analyze_outcomes(ROOT/analyses['frozen_source']['run'])
    result = dict(written_at_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        analyses=analyses, source_target_outcomes=outcome, source_diagnostic=diagnostic, costs=costs,
        total_driver_seconds=sum(c['wall_seconds'] for c in costs),
        total_process_cpu_seconds=sum(c['process_cpu_seconds'] for c in costs),
        scope='Frozen new-arity test precedes source refitting. Refit evaluations are exposed development. Source/target agreement and correctness are separate. Failed metadata-only evaluation retained.')
    for p in [ART/'R50_COMBINED_RESULTS.json', PAPER/'data/arithmetic_carry_generalization.json']:
        p.write_text(json.dumps(result, indent=2)+'\n')
    with (PAPER/'data/arithmetic_carry_generalization.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    source = {name:{(c['method'],c['condition']):c for c in analyses[name]['cells']}
              for name in ['source_function_ce_two_operand','source_function_ce']}
    tex=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrrrr}',r'\toprule',
        r'& \multicolumn{2}{c}{Two operands} & \multicolumn{2}{c}{Three operands} \\',
        r'Method & Carry change & Preserve & Carry change & Preserve \\',r'\midrule']
    for method in list(LABELS)[:6]:
        values=[100*source[name][method,condition]['success']['mean']
                for name in source for condition in CONDITIONS]
        tex.append(LABELS[method]+' & '+' & '.join(f'{v:.2f}' for v in values)+r' \\')
    tex += [r'\bottomrule\end{tabular}',
        r'\caption{Source-function learning and generalization. Complete predicted-answer success (percent); preservation is the same-carry condition. All methods use one source SAE and at most 64 changed members, except the unrestricted raw direction. Answer CE uses only the original two-operand fit split and a fixed final checkpoint. Both evaluation panels are exposed development for this refit. Two operands have 32 canonical pairs per condition; three operands have 48 changing-carry and 16 preservation pairs. Each pair retains both orientations and two forms.}',
        r'\label{tab:carry_source_generalization}\end{anchoredtable}']
    (PAPER/'tables/arithmetic_carry_source_generalization.tex').write_text('\n'.join(tex)+'\n')
    target=analyses['frozen_targets'];cells={(c['method'],c['condition']):c for c in target['cells']}
    strata={(c['method'],c['stratum']):c for c in target['per_stratum']}
    tex=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrrrrr}',r'\toprule',
        r'& \multicolumn{4}{c}{Carry change} & Same carry \\',
        r'Method & Pooled & $0\leftrightarrow1$ & $1\leftrightarrow2$ & $0\leftrightarrow2$ & Preserve \\',r'\midrule']
    for method in ['noop','raw_carry_direction','assignment_64','field_fitted_binary_64',
                   'transferred_fitted_binary_64','transferred_fitted_weighted_64']:
        values=[cells[method,CONDITIONS[0]]['success']['mean']]+[strata[method,s]['success']
            for s in ['carry_0_1','carry_1_2','carry_0_2']]+[cells[method,CONDITIONS[1]]['success']['mean']]
        tex.append(LABELS[method]+' & '+' & '.join(f'{100*v:.2f}' for v in values)+r' \\')
    tex += [r'\bottomrule\end{tabular}',
        r'\caption{Frozen three-operand rule test, averaged over target SAEs 2--5. All source and target memberships were fixed using two-operand data before any three-operand output was observed. Every predetermined question is retained. There are 16 canonical pairs in each changing-carry stratum and 16 preservation pairs. Raw is shared across the cohort; directions are not independent replicates. Field fit is direct target re-identification; source transfer uses the frozen source-group field.}',
        r'\label{tab:carry_frozen_targets}\end{anchoredtable}']
    (PAPER/'tables/arithmetic_carry_frozen_targets.tex').write_text('\n'.join(tex)+'\n')
    print(json.dumps(dict(costs=costs, source_table=[r for r in rows if r['panel'].startswith('source_function')]),indent=2))


if __name__ == '__main__':main()
