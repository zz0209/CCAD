from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import numpy as np
import torch
from analyze_program_generalization import read_run


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/reuse_generalization_20260921_round01'
BULK=Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round01')


def main():
    torch.set_num_threads(2)
    a=json.loads((OUT/'LEARNING_ANALYSIS.json').read_text())
    checked=[]
    for objective,groups in a['records'].items():
        for name,record in groups.items():
            assert read_run(Path(record['run']))==record
            checked.append(record['run'])
    c=json.loads((ROOT/'configs/rg01_state_writer_development.json').read_text())
    unchanged=[]
    writer_run=BULK/'RG01_STATE_WRITER_DEVELOPMENT_20260921'
    for steps in [512,4096]:
        for site in c['train_sites']:
            name=f'{site}_seed3.pt'
            initial=torch.load(Path(c['target_directory'])/name,map_location='cpu',weights_only=True)
            trained=torch.load(writer_run/f'generic_program_{steps}'/name,map_location='cpu',weights_only=True)
            assert initial.keys()==trained.keys()
            assert all(torch.equal(initial[k],trained[k]) for k in initial)
            unchanged.append(f'{steps}/{site}')
    initial=torch.load(Path(c['target_directory'])/'resid_4_seed3.pt',map_location='cpu',weights_only=True)
    writer=torch.load(writer_run/'generic_program_4096/program_writers.pt',map_location='cpu',weights_only=True)['resid_4']
    delta=float((writer-initial['encoder.weight']).norm())
    assert delta>0
    support={name:read_run(BULK/f'RG01_ACTIVE_EXECUTION_{name.upper()}_DEVELOPMENT_20260921') for name in ['infinitive','agreement']}
    for name,r in support.items():
        reference=a['records']['local']['infinitive' if name=='infinitive' else 'agreement_4096']
        da=np.load(Path(r['run'])/'responses.npz'); db=np.load(Path(reference['run'])/'responses.npz')
        for key in ['grammar__none','grammar__source','grammar__initial_tangent','grammar__initial_raw_readout']:
            assert np.array_equal(da[key],db[key]),(name,key)
    rows=[]
    labels={'local':'Aggregate supervision','local_parts':'Column supervision','state_parts':'Active-state columns','state_writer':'Separate action writer'}
    for objective,groups in a['records'].items():
        quality=json.loads((Path(groups['infinitive']['run'])/'quality.json').read_text())
        for steps in [512,4096]:
            qi=next(r for r in quality if r['site']=='resid_4' and r['method']==f'generic_program_{steps}')
            inf=groups['infinitive']['methods'][f'generic_program_{steps}']['families']
            agr=groups[f'agreement_{steps}']['methods']['task_adapted_tangent']['families']
            rows.append(dict(objective=objective,label=labels[objective],steps=steps,infinitive=inf['participation'],agreement=agr['participation'],infinitive_members=inf['members'],agreement_members=agr['members'],fve=qi['fve']))
    with (ROOT/'paper/data/generic_execution_learning.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'& \multicolumn{2}{c}{Infinitive} & \multicolumn{2}{c}{Subject number} & Natural \\',r'Training rule & 512 & 4,096 & 512 & 4,096 & FVE \\',r'\midrule']
    for objective in labels:
        x,y=[r for r in rows if r['objective']==objective]
        lines.append(f"{x['label']} & {x['infinitive']:.3f} & {y['infinitive']:.3f} & {x['agreement']:.3f} & {y['agreement']:.3f} & {y['fve']:.3f} "+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (ROOT/'paper/tables/generic_execution_learning.tex').write_text('\n'.join(lines)+'\n')
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),evidence='One target, exposed development functions. Source identity is fixed, and ordinary actions share eight held text sequences.',
        rows=rows,support_comparison=support,checks=dict(result_collections=len(checked),all_recomputed_exactly=True,
        unchanged_writer_dictionary_checkpoints=unchanged,writer_parameter_change_norm=delta,
        support_controls_bitwise_equal=True,field_interface=json.loads((OUT/'FIELD_INTERFACE_CHECK.json').read_text())))
    dest=OUT/'RESULTS_AND_CHECKS.json';assert not dest.exists()
    dest.write_text(json.dumps(result,indent=2)+'\n')
    registry=ROOT/'paper/EVIDENCE_INDEX.json';index=json.loads(registry.read_text())
    index['generic_execution_learning_development']=dict(analysis=str(dest.relative_to(ROOT)),
        sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),table='paper/tables/generic_execution_learning.tex',
        data='paper/data/generic_execution_learning.csv',evidence=result['evidence'])
    registry.write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps(dict(rows=rows,support={n:{m:v['families'] for m,v in r['methods'].items()} for n,r in support.items()},checks=result['checks']),indent=2))


if __name__=='__main__':
    main()
