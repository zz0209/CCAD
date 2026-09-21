from pathlib import Path
import csv
import json


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/final_science_20260921_round05'


def main():
    human=json.loads((OUT/'HUMAN_REQUEST_DEVELOPMENT.json').read_text())
    grammar=json.loads((OUT/'INFINITIVE_REQUEST_DEVELOPMENT.json').read_text())
    generic=json.loads((OUT/'GENERIC_PROGRAM_GENERALIZATION.json').read_text())
    cross=json.loads((OUT/'CROSS_PROGRAM_GENERALIZATION.json').read_text())
    rows=[]
    table=[r'\begin{tabular}{lrrrrrr}',r'\toprule',
        r'& \multicolumn{3}{c}{Human explanation} & \multicolumn{3}{c}{Infinitive} \\',
        r'Training requests & Parts & Participation & Members & Parts & Participation & Members \\',r'\midrule']
    choices=[('semantic','semantic','Semantic groups'),('balanced','balanced','Fixed group mass'),
        ('independent','independent','Independent members'),('input_initial','native_tangent_relation_8','Initial encoder columns'),
        ('raw_reconstruction','raw_reconstruction','Source readout')]
    for h,g,label in choices:
        values=[]
        for setting,study,key in [('human',human,h),('infinitive',grammar,g)]:
            for family in ['endpoints','participation','member_subsets']:
                value=study['summary'][key][family]['nrmse']; values.append(value)
                rows.append(dict(panel='requests',setting=setting,method=label,family=family,nrmse=value))
        table.append(label+' & '+' & '.join(f'{v:.3f}' for v in values)+r' \\')
    table += [r'\bottomrule',r'\end{tabular}']
    (ROOT/'paper/tables/request_training_development.tex').write_text('\n'.join(table)+'\n')
    table=[r'\begin{tabular}{lrrrr}',r'\toprule',
        r'& \multicolumn{2}{c}{Infinitive} & \multicolumn{2}{c}{Subject number} \\',
        r'Training information & Participation & Members & Participation & Members \\',r'\midrule']
    for key,label in [('initial_tangent','Initial encoder columns'),('local','Other members, local state'),
                      ('downstream','Other members, final state'),('distribution','Other members, vocabulary'),
                      ('infinitive','Infinitive explanation'),('initial_raw_readout','Source readout')]:
        values=[]
        for setting in ['infinitive','agreement']:
            if key=='infinitive':
                result=(grammar['summary']['semantic'] if setting=='infinitive' else
                        cross['records']['semantic']['methods']['task_adapted_tangent']['families'])
                vals=([result['participation']['nrmse'],result['member_subsets']['nrmse']] if setting=='infinitive' else
                      [result['participation'],result['members']])
            else:
                entry=generic['records']['local' if key.startswith('initial') else key][setting]
                method=key if key.startswith('initial') else ('generic_program_512' if setting=='infinitive' else 'task_adapted_tangent')
                result=entry['methods'][method]['families']; vals=[result['participation'],result['members']]
            values.extend(vals)
            for family,value in zip(['participation','members'],vals):
                rows.append(dict(panel='transfer',setting=setting,method=label,family=family,nrmse=value))
        table.append(label+' & '+' & '.join(f'{v:.3f}' for v in values)+r' \\')
    table += [r'\bottomrule',r'\end{tabular}']
    (ROOT/'paper/tables/program_transfer_development.tex').write_text('\n'.join(table)+'\n')
    with (ROOT/'paper/data/request_transfer_development.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f'Exported {len(rows)} source-linked values.')


if __name__=='__main__':
    main()
