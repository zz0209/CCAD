from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import numpy as np


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/reuse_generalization_20260921_round04'
BULK=Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round04')
LABELS=[('Local actions','local_action'),('State feedback','state_feedback'),
        ('Source readout','raw_readout'),('Fixed counterparts','counterpart_action_span_64'),
        ('Exchanged counterparts','counterpart_swapped_action_span_64')]


def main():
    result=json.loads((OUT/'NUMBER_CONFIRMATION_ANALYSIS.json').read_text())
    rows=[]
    for study in result['studies']:
        for family,methods in study['metrics'].items():
            rows.extend(dict(target=study['target_seed'],family=family,method=m,nrmse=v) for m,v in methods.items())
    data=ROOT/'paper/data/program_counterparts.csv'
    with data.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=[r'\begin{tabular}{lrrr}',r'\toprule',r'Execution & Parts & Union & All \\',r'\midrule']
    for label,method in LABELS:
        values=[result['mean'][family][method] for family in ['parts','union','all']]
        lines.append(label+' & '+' & '.join(f'{v:.3f}' for v in values)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (ROOT/'paper/tables/program_counterparts.tex').write_text('\n'.join(lines)+'\n')
    support_rows=[];costs=[]
    for study in result['studies']:
        run=Path(study['run']);config=json.loads((run/'config.resolved.json').read_text())
        bank=json.loads(Path(config['counterpart_path']).read_text())['supports']['action_span_64']
        for site in bank['singular']:
            a,b=set(bank['singular'][site]),set(bank['plural'][site])
            support_rows.append(dict(target=study['target_seed'],site=site,singular=len(a),plural=len(b),
                                     union=len(a|b),shared=len(a&b),exchangeable=len(a)==len(b)))
        execution=json.loads((run/'target_analysis.json').read_text())['execution']
        for method,counts in execution.items():
            costs.append(dict(target=study['target_seed'],method=method,
                              changed_per_state=counts['changed']/counts['states'],**counts))
    human=json.loads((OUT/'HUMAN_COUNTERPART_SMOKE_ANALYSIS.json').read_text())['studies'][0]
    propagation=json.loads((OUT/'HUMAN_PROPAGATED_SMOKE_ANALYSIS.json').read_text())['studies'][0]
    replay={}
    first=Path(human['run']);second=Path(propagation['run'])
    for file,prefix in [('responses.npz','human__'),('pooled.npz','')]:
        with np.load(first/file) as a,np.load(second/file) as b:
            for method in ['none','source','initial_trajectory_feedback']:
                key=prefix+method
                replay[file+'__'+method]=bool(np.array_equal(a[key],b[key]))
    if not all(replay.values()):
        raise ValueError('Shared human predictions changed')
    human_methods=[('State feedback',human,'initial_trajectory_feedback'),
                   ('Local-action counterparts',human,'initial_counterpart_action_span_64'),
                   ('Propagated-effect counterparts',propagation,'initial_counterpart_propagated_span_64')]
    lines=[r'\begin{tabular}{lrrr}',r'\toprule',r'Source information & Parts & Full & All \\',r'\midrule']
    for label,study,method in human_methods:
        values=[study['metrics'][method]['later'][f] for f in ['parts','full','all']]
        lines.append(label+' & '+' & '.join(f'{v:.3f}' for v in values)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (ROOT/'paper/tables/human_counterpart_development.tex').write_text('\n'.join(lines)+'\n')
    confirmation=json.loads((OUT/'HUMAN_COUNTERPART_CONFIRMATION_ANALYSIS.json').read_text())
    human_labels=[('State feedback','initial_trajectory_feedback'),
                  ('Source readout','initial_raw_readout'),
                  ('Local-action pools','initial_counterpart_action_span_64'),
                  ('Site-count control','initial_counterpart_filled_action_span_64'),
                  ('Propagated-effect pools','initial_counterpart_propagated_span_64')]
    available=set(confirmation['mean'])
    if any(method not in available for _,method in human_labels):
        raise ValueError(available)
    lines=[r'\begin{tabular}{lrrr}',r'\toprule',r'Execution & Parts & Full & All \\',r'\midrule']
    for label,method in human_labels:
        values=[confirmation['mean'][method][family] for family in ['parts','full','all']]
        lines.append(label+' & '+' & '.join(f'{v:.3f}' for v in values)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (ROOT/'paper/tables/human_counterpart_confirmation.tex').write_text('\n'.join(lines)+'\n')
    human_rows=[]
    human_supports=[]
    for study in confirmation['studies']:
        config=json.loads((Path(study['run'])/'config.resolved.json').read_text())
        supports=json.loads(Path(config['counterpart_path']).read_text())['supports']
        parts=['pronouns','names','associated_words']
        for part in parts:
            for site,ids in supports['propagated_span_64'][part].items():
                if len(ids)!=len(supports['filled_action_span_64'][part][site]):
                    raise ValueError((config['target_seed'],part,site))
        for query in study['queries']:
            selected=parts if query in ['full','center'] else query.split('+')
            for site in supports['propagated_span_64'][parts[0]]:
                for scheme,bank in supports.items():
                    members=set().union(*(set(bank[part][site]) for part in selected))
                    human_supports.append(dict(target=config['target_seed'],query=query,
                                               site=site,scheme=scheme,candidates=len(members)))
        for method,metrics in study['metrics'].items():
            for endpoint,values in metrics.items():
                if endpoint in ['later','original']:
                    human_rows.extend(dict(target=config['target_seed'],endpoint=endpoint,
                                           family=family,method=method,nrmse=value)
                                      for family,value in values.items())
    with (ROOT/'paper/data/human_counterparts.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(human_rows[0]))
        writer.writeheader();writer.writerows(human_rows)
    output=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),number_supports=support_rows,
                number_execution=costs,human_replay=replay,human_local=human['metrics'],
                human_propagated=propagation['metrics'],
                human_confirmation=confirmation['mean'],
                human_confirmation_comparisons=confirmation['comparisons'],
                human_supports=human_supports,
                human_confirmation_sha256=hashlib.sha256((OUT/'HUMAN_COUNTERPART_CONFIRMATION_ANALYSIS.json').read_bytes()).hexdigest(),
                confirmation_sha256=hashlib.sha256((OUT/'NUMBER_CONFIRMATION_ANALYSIS.json').read_bytes()).hexdigest())
    (OUT/'COUNTERPART_EXPORT.json').write_text(json.dumps(output,indent=2)+'\n')
    index_path=ROOT/'paper/EVIDENCE_INDEX.json'
    index=json.loads(index_path.read_text())
    index['program_counterparts']={
        'number_analysis':str((OUT/'NUMBER_CONFIRMATION_ANALYSIS.json').relative_to(ROOT)),
        'number_freeze':str((OUT/'NUMBER_CONFIRMATION_FREEZE.json').relative_to(ROOT)),
        'human_analysis':str((OUT/'HUMAN_COUNTERPART_CONFIRMATION_ANALYSIS.json').relative_to(ROOT)),
        'human_freeze':str((OUT/'HUMAN_COUNTERPART_FREEZE.json').relative_to(ROOT)),
        'number_sha256':output['confirmation_sha256'],
        'human_sha256':output['human_confirmation_sha256'],
        'tables':['paper/tables/program_counterparts.tex','paper/tables/human_counterpart_confirmation.tex'],
        'data':['paper/data/program_counterparts.csv','paper/data/human_counterparts.csv'],
        'evidence':'64 new number prefixes and32 new biographies, three fixed targets per study. Fixed source banks and candidate pools. Online source trajectory retained. Union pool sizes are reported separately from per-part/site pool counts.'}
    index_path.write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows),support_records=len(support_rows),human_replay=replay)))


if __name__=='__main__':
    main()
