"""Export operation-specific binding evidence and source-anchored comparisons."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/correspondence_reform_20260913'
PAPER=ROOT/'paper'


def main():
    replacement=json.loads((ART/'r47_binding_mixing_vs_gain.json').read_text())
    deletion=json.loads((ART/'r47_binding_deletion.json').read_text())
    run=ROOT/'runs/REFORM_R47_binding_mixing_vs_gain_v1_20260915'
    selection=json.loads((run/'WRITER_SELECTION_s1.json').read_text())
    names=[('Source','country64'),('Code readout','code_readout_country'),
           ('Raw readout','raw_readout_country'),('Direct target selection','target_country'),
           ('Selected native gain','selected_native_gain'),('Selected native mixing','selected_native_mix'),
           ('Selected raw mixing','selected_raw_mix')]
    table=[]
    for label,method in names:
        record=dict(method=label)
        for op,field in [('macro','all_requests'),('both','joint_request')]:
            record[field]=100*next(c['complete_binding_accuracy'] for c in replacement['macro'] if c['method']==method and c['operation']==op)
        family={'selected_native_gain':'native_gain','selected_native_mix':'native_mix','selected_raw_mix':'raw_mix'}.get(method)
        record['validation_kl']=selection['best'][family]['loss'] if family else None
        record['selected_update']=selection['best'][family]['update'] if family else None
        table.append(record)
    arrays=np.load(ART/'r47_binding_mixing_vs_gain.npz');oi=arrays['operations'].tolist().index('both')
    weights=np.random.default_rng(947015).multinomial(64,np.full(64,1/64),10000)/64
    contrasts=[]
    for reference in ['selected_native_gain','selected_raw_mix','code_readout_country','target_country']:
        a=arrays['layer13|selected_native_mix|1|2'][...,0].prod(-1)
        b=arrays[f'layer13|{reference}|1|2'][...,0].prod(-1)
        for label,d in [('all',(a-b).mean((1,2))),('both',(a-b)[:,:,oi].mean(1))]:
            contrasts.append(dict(method='selected_native_mix',reference=reference,operation=label,
                                  difference_points=float(100*d.mean()),interval_points=(100*np.quantile(weights@d,[.025,.975])).tolist()))
    dtable=[]
    for method,label in [('source_delete','Source deletion'),('general_geometry','Geometry, general codes'),
                         ('general_gain','Gain, general codes'),('general_mix','Mixing, general codes'),
                         ('deletion_geometry','Geometry, attenuation'),('deletion_gain','Gain, attenuation'),
                         ('deletion_mix','Mixing, attenuation'),('projected_deletion','Projected attenuation')]:
        c=next(c for c in deletion['cells'] if c['method']==method and c['operation']=='all')
        dtable.append(dict(method=label,source_answer=100*c['requested_source_answer']['mean'],
                           original_city=100*c['requested_original_city_retention']['mean'],
                           protected_city=100*c['protected_original_city_retention']['mean'],
                           source_logp_mae=c['requested_original_city_logp_mae']['mean']))
    costs=[]
    for name in ['mixing_vs_gain','unfitted_deletion','projected_deletion']:
        p=ROOT/f'runs/REFORM_R47_binding_{name}_v1_20260915';s=json.loads((p/'metrics.summary.json').read_text())
        costs.append(dict(run=p.relative_to(ROOT).as_posix(),status=json.loads((p/'status.json').read_text()),
                          **{k:s[k] for k in ['wall_seconds','process_cpu_seconds','peak_allocated_bytes','sequence_forwards','token_forwards','rows']}))
    out=dict(replacement=replacement,selection=selection,replacement_table=table,replacement_contrasts=contrasts,
             deletion=deletion,deletion_table=dtable,costs=costs,total_driver_seconds=sum(c['wall_seconds'] for c in costs),
             scope='Exposed64-context development panel; source1 to target2. Deletion consumes exact source amplitudes. No independent confirmation.')
    for p in [ART/'R47_COMBINED_RESULTS.json',PAPER/'data/binding_operation_development.json']:
        p.write_text(json.dumps(out,indent=2)+'\n')
    for name,rows in [('binding_writer_selection',table),('binding_unfitted_deletion',dtable)]:
        with (PAPER/f'data/{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tex=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrrrr}',r'\toprule',
         r'Method & All requests & Joint request & Validation KL & Update \\',r'\midrule']
    for r in table:
        kl='--' if r['validation_kl'] is None else f"{r['validation_kl']:.4f}"
        update='--' if r['selected_update'] is None else str(r['selected_update'])
        tex.append(f"{r['method']} & {r['all_requests']:.2f} & {r['joint_request']:.2f} & {kl} & {update}"+r' \\')
    tex += [r'\bottomrule\end{tabular}',r'\caption{Binding replacement after source-fit model selection. Accuracy is complete city-pair success (\%). Each learned family receives the same three learning rates and four checkpoint choices. KL uses source-fit validation responses; accuracy uses the exposed development panel.}',r'\label{tab:binding_writer_selection}\end{anchoredtable}']
    (PAPER/'tables/binding_writer_selection.tex').write_text('\n'.join(tex)+'\n')
    tex=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrrrr}',r'\toprule',
         r' & Source answer & Original city & Protected city & Original-city \\',
         r'Method & agreement (\%) & retained (\%) & retained (\%) & log-prob. MAE \\',r'\midrule']
    for r in dtable:
        tex.append(f"{r['method']} & {r['source_answer']:.2f} & {r['original_city']:.2f} & {r['protected_city']:.2f} & {r['source_logp_mae']:.3f}"+r' \\')
    tex += [r'\bottomrule\end{tabular}',r'\caption{Unfitted deletion of country-contrast source members. All methods receive the exact source amplitudes. Agreement concerns the requested entity; protected-city retention uses singleton requests. Source deletion changes 418 of 512 requested answers. Original-city retention is an effect description, not replacement accuracy. General codes permit increases; attenuation only reduces existing codes. The same exposed 64 contexts and source1--target2 direction are used.}',r'\label{tab:binding_unfitted_deletion}\end{anchoredtable}']
    (PAPER/'tables/binding_unfitted_deletion.tex').write_text('\n'.join(tex)+'\n')
    print(json.dumps(dict(replacement_contrasts=contrasts,total_driver_seconds=out['total_driver_seconds'])))


if __name__=='__main__':main()
