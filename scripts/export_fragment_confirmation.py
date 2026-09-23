from pathlib import Path
import argparse
import hashlib
import json


def export(results, paper):
    results, paper = Path(results), Path(paper)
    data = json.loads(results.read_text())
    assert data['split'] == 'confirmation' and data['target_seeds'] == [4, 5]
    methods = [
        ('Target carrier whole', 'F_carrier4_whole', 'L_carrier4_whole', 'equal_target'),
        ('PW carrier whole', 'PW_F_carrier4_whole', 'PW_L_carrier4_whole', 'equal_target'),
        ('Whole-energy four', 'whole_energy4_for_F', 'whole_energy4_for_L', 'equal_target'),
        ('Source carrier whole', 'F_carrier4_whole', 'L_carrier4_whole', 'source_direct_execution'),
        ('Target full projection', 'F', 'L', 'equal_target'),
        ('Raw full projection', 'raw_F', 'raw_L', 'equal_target'),
    ]
    rows = []
    for label, first, last, group in methods:
        values = [data[group][operation][metric]['value'] for operation in (first, last)
                  for metric in ('requested_damage', 'protected_absolute_change')]
        rows.append(dict(method=label, first_requested=values[0], first_protected=values[1],
                         last_requested=values[2], last_protected=values[3]))
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Operation & \multicolumn{2}{c}{First letter} & \multicolumn{2}{c}{Last letter} \\',
             r' & Requested & Protected & Requested & Protected \\', r'\midrule']
    for row in rows:
        values = [row[key] for key in ('first_requested', 'first_protected', 'last_requested', 'last_protected')]
        lines.append(row['method']+' & '+' & '.join(f'{value:.4f}' for value in values)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (paper/'tables/fragment_confirmation.tex').write_text('\n'.join(lines)+'\n')
    values = dict(FragWords=str(data['words']),
                  FragCleanFirst=f"{100*data['equal_target']['clean']['first_full_vocab_accuracy']['value']:.2f}",
                  FragCleanLast=f"{100*data['equal_target']['clean']['last_full_vocab_accuracy']['value']:.2f}")
    for role, prefix in [('F', 'FragFirst'), ('L', 'FragLast')]:
        contrast = data['paired_contrasts'][f'{role}_carrier4_whole_minus_PW_{role}_carrier4_whole']
        for metric, suffix in [('requested_damage', 'DamageDifference'), ('protected_absolute_change', 'ProtectedDifference')]:
            stat = contrast[metric]
            values[prefix+suffix] = f"{stat['value']:.3f}"
            values[prefix+suffix+'Lower'] = f"{stat['ci95'][0]:.3f}"
            values[prefix+suffix+'Upper'] = f"{stat['ci95'][1]:.3f}"
    (paper/'tables/fragment_confirmation_values.tex').write_text('\n'.join('\\newcommand{\\'+key+'}{'+value+'}' for key,value in values.items())+'\n')
    receipt = dict(input_path=str(results.resolve()), input_sha256=hashlib.sha256(results.read_bytes()).hexdigest(),
                   analyzer=data['analyzer'], words=data['words'], target_seeds=data['target_seeds'],
                   bootstrap=data['bootstrap'], table_rows=rows, macros=values)
    (paper/'data/fragment_confirmation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(words=data['words'], rows=len(rows), input_sha256=receipt['input_sha256'])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--paper', type=Path, default=Path('paper'))
    args = parser.parse_args()
    export(args.results, args.paper)


if __name__ == '__main__':
    main()
