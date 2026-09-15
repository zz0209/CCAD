"""Export the two request-writer development pilots as one comparison."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/correspondence_reform_20260913'
PAPER=ROOT/'paper'


def main():
    a=json.loads((ART/'r46_request_writer_pilot.json').read_text())
    b=json.loads((ART/'r46_binding_request_writer_pilot_v3.json').read_text())
    mapping=[('Source','source','country64'),('Member relation','member','union_member_country'),
             ('Assignment','assignment','assignment_country'),('Code readout','full_activation_readout','code_readout_country'),
             ('Raw readout / initial raw','raw_readout','raw_readout_country'),
             ('Native initialization','learned_native_u0','learned_native_u0_country'),
             ('Learned native, 64','learned_native_u64','learned_native_u64_country'),
             ('Learned native, 128','learned_native_u128','learned_native_u128_country'),
             ('Learned raw, 64','learned_raw_u64','learned_raw_u64_country'),
             ('Learned raw, 128','learned_raw_u128','learned_raw_u128_country'),
             ('Per-request code solver',None,'synthesized_code_bank'),
             ('Direct country selection',None,'target_country')]
    table=[]
    for label,am,bm in mapping:
        part=next((r['balanced'] for r in a['fidelity'] if r['method']==am),None)
        full=[next((r['H'] for r in a['cells'] if r['method']==am and r['query']=='full' and r['operation']==op),None) for op in ['unit','tens']]
        binding=[100*next(r['complete_binding_accuracy'] for r in b['macro'] if r['method']==bm and r['operation']==op) for op in ['macro','both']]
        table.append(dict(method=label,arithmetic_parts=part,arithmetic_units=full[0],arithmetic_tens=full[1],binding_all=binding[0],binding_joint=binding[1]))
    arrays=np.load(ART/'r46_binding_request_writer_pilot_v3.npz')
    joint_index=arrays['operations'].tolist().index('both')
    weights=np.random.default_rng(946028).multinomial(64,np.full(64,1/64),10000)/64
    contrasts=[]
    for ref in ['synthesized_code_bank','union_member_country','code_readout_country','raw_readout_country','learned_raw_u64_country','target_country']:
        left=arrays['layer13|learned_native_u128_country|1|2'][...,0].prod(-1)
        right=arrays[f'layer13|{ref}|1|2'][...,0].prod(-1)
        for op in ['all','both']:
            d=(left-right).mean((1,2)) if op=='all' else (left-right)[:,:,joint_index].mean(1)
            label='macro' if op=='all' else 'both'
            expected=100*(next(r['complete_binding_accuracy'] for r in b['macro'] if r['method']=='learned_native_u128_country' and r['operation']==label)-next(r['complete_binding_accuracy'] for r in b['macro'] if r['method']==ref and r['operation']==label))
            assert abs(100*d.mean()-expected)<1e-8
            contrasts.append(dict(reference=ref,operation=op,points=float(100*d.mean()),interval=(100*np.quantile(weights@d,[.025,.975])).tolist()))
    run_names=[a['run']]+['runs/REFORM_R46_binding_request_writer_pilot_v'+str(v)+'_20260915' for v in [1,2,3]]
    costs=[]
    for run in run_names:
        summary=json.loads((ROOT/run/'metrics.summary.json').read_text())
        costs.append(dict(run=run,status=json.loads((ROOT/run/'status.json').read_text()),
                          **{k:summary[k] for k in ['wall_seconds','process_cpu_seconds','peak_allocated_bytes','sequence_forwards','token_forwards','rows']}))
    data=dict(arithmetic=a,binding=b,table=table,contrasts=contrasts,costs=costs,
              total_driver_seconds=sum(r['wall_seconds'] for r in costs),
              scope='Two exposed development panels; one source-target direction. Checkpoints retained. No independent-confirmation claim.')
    (ART/'R46_COMBINED_RESULTS.json').write_text(json.dumps(data,indent=2)+'\n')
    (PAPER/'data/request_writer_development.json').write_text(json.dumps(data,indent=2)+'\n')
    with (PAPER/'data/request_writer_development.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    tex=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrrrrr}',r'\toprule',
         r' & \multicolumn{3}{c}{Arithmetic} & \multicolumn{2}{c}{Binding} \\',
         r'Method & Part agreement & Full units & Full tens & All requests & Joint request \\',r'\midrule']
    for row in table:
        tex.append(row['method']+' & '+' & '.join('--' if row[k] is None else f'{row[k]:.2f}' for k in list(row)[1:])+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\caption{Reusable response-trained relations on two development panels. All entries are percentages. Arithmetic: eight question clusters, two forms; binding: 64 contexts, two forms. Both use source seed 1 and target seed 2. Updates count per digit in arithmetic and per matrix in binding. Raw initialization equals the retained raw readout. Native matrices are frozen at evaluation; the binding joint operation was absent from their response fitting.}',r'\label{tab:request_writer_development}',r'\end{anchoredtable}']
    (PAPER/'tables/request_writer_development.tex').write_text('\n'.join(tex)+'\n')
    print(json.dumps(dict(table=table,contrasts=contrasts,total_driver_seconds=data['total_driver_seconds'])))


if __name__=='__main__':main()
