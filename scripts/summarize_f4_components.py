"""All-case controls and nonlinear component composition from saved real forwards."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from summarize_f4_probability import ROOT,key,median,fmt,jsonl,write,sha256,METRICS

METHODS=['target','raw','global_rows','single_atom_dynamic','uot_default','uot_discovery_tuned',
         'wrong_query_matched_energy','native_top16_difference_matched','readout_top16','readout_tail16','readout_random16']
SCOPES=['intervention_positions','same_document_downstream']
ERRORS=['normalized_kl_error','normalized_nll_delta_squared_error']


def ratio(x,y):return float(x/y) if y>1e-12 else None


def main():
    output=ROOT/'runs/F4_component_probability_expanded_v1_20260905'
    if (output/'component_summary.json').exists():raise ValueError('Component summary exists')
    all_rows=[];cases=[];compositions=[];checks={};inputs=[];forwards=0;seconds=0
    for panel in ['original','expanded']:
        run=ROOT/'runs'/f'F4_component_probability_{panel}_v1_20260905'
        old=ROOT/'runs'/f'F4_probability_confirmation_{panel}_v1_20260905'
        cfg=json.loads((run/'config.resolved.json').read_text())
        assert json.loads((run/'status.json').read_text())['status']=='PASS'
        summary=json.loads((run/'metrics.summary.json').read_text());forwards+=summary['model_forwards'];seconds+=summary['wall_seconds']
        rows=jsonl(run/'metrics.raw.jsonl');index={key(r):r for r in rows};previous=jsonl(old/'metrics.raw.jsonl')
        oldindex={key(r):r for r in previous};anchors=0
        for r in rows:
            if r['method'] in ('target','readout_top16'):
                before=oldindex[key(r)]
                for f in ('endpoints','probability_endpoints','hook','common_source_dose_scale'):assert r[f]==before[f],f
                anchors+=1
        assert json.loads((run/'selection.json').read_text())==json.loads((old/'selection.json').read_text())
        assert json.loads((run/'source_scope_selection.json').read_text())==json.loads((old/'source_scope_selection.json').read_text())
        rows+= [r for r in previous if r['method'] in ('raw','global_rows')]
        for r in rows:
            for scope,p in r['probability_endpoints'].items():
                all_rows.append(dict(panel=panel,source_seed=r['source_seed'],source_atom=r['source_atom'],condition=r['condition'],
                    sequence=r['sequence'],target_seed=r['target_seed'],method=r['method'],scope=scope,**{k:p[k] for k in METRICS},
                    hook_error=r['hook']['normalized_error'],candidate_hook_fraction=r['candidate_hook_fraction'],
                    centered_logit_error=r['endpoints']['centered_logits']['normalized_error'],next_state_error=r['endpoints']['next_state']['normalized_error']))
        matching=json.loads((ROOT/cfg['case_replay']['path']).read_text());selected=0
        groupmap={(r['source_seed'],r['source_atom'],r['target_seed']):r for r in json.loads((run/'component_group_selection.json').read_text())['groups']}
        for c in matching['choices']:
            s,a,cond=c['source_seed'],c['source_atom'],c['condition'];entry=c['entry']
            chosen=bool(entry and c['source_scope']['selected']);record=dict(panel=panel,source_seed=s,source_atom=a,condition=cond,
                selected=chosen,status='SELECTED' if chosen else c['matching_status'] if not entry else 'SOURCE_NOT_SELECTED',entry=entry,scopes={})
            if chosen:
                selected+=1
                group=[r for r in rows if (r['source_seed'],r['source_atom'],r['condition'])==(s,a,cond)]
                assert len(group)==4*len(METHODS)
                for scope in SCOPES:
                    record['scopes'][scope]={method:{m:median([r['probability_endpoints'][scope][m] for r in group if r['method']==method]) for m in METRICS} for method in METHODS}
                    src=group[0]['probability_endpoints'][scope]
                    assert all(r['probability_endpoints'][scope]['source_nll_deltas']==src['source_nll_deltas'] for r in group)
                for target in sorted({r['target_seed'] for r in group}):
                    g={r['method']:r for r in group if r['target_seed']==target};coords=g['target']['component_coordinates']
                    assert all(r['component_coordinates']==coords for r in g.values() if 'component_coordinates' in r)
                    full,head,tail=(np.array(coords[k]) for k in ['full','top','tail'])
                    np.testing.assert_allclose(head+tail,full,atol=1e-12,rtol=1e-12)
                    rec=dict(panel=panel,source_seed=s,source_atom=a,condition=cond,target_seed=target,
                        group=groupmap[s,a,target],coordinates=coords,head_energy_over_full=ratio(head@head,full@full),
                        tail_energy_over_full=ratio(tail@tail,full@full),head_tail_cross_over_full=ratio(2*head@tail,full@full),scopes={})
                    for scope in SCOPES:
                        fs,hs,ts=(g[m]['probability_endpoints'][scope] for m in ('target','readout_top16','readout_tail16'))
                        f,h,t=(np.array(v['candidate_nll_deltas']) for v in (fs,hs,ts))
                        residual=f-h-t
                        rec['scopes'][scope]=dict(full_nll_delta=f.tolist(),head_nll_delta=h.tolist(),tail_nll_delta=t.tolist(),
                            composition_residual=residual.tolist(),composition_nll_relative_squared_error=ratio(residual@residual,f@f),
                            composition_nll_rmse=float(np.sqrt(np.mean(residual**2))),
                            tail_minus_full_error={m:ts[m]-fs[m] for m in ERRORS},head_minus_full_error={m:hs[m]-fs[m] for m in ERRORS})
                    compositions.append(rec)
            cases.append(record)
        assert len(index)==selected*4*9
        checks[panel]=dict(selected_cases=selected,requested_cases=len(matching['choices']),new_rows=len(index),exact_old_anchor_rows=anchors,
            source_selection_and_scope_exact=True,component_coordinates_additive=True)
        for p in (run/'config.resolved.json',run/'metrics.raw.jsonl',run/'metrics.summary.json',run/'component_group_selection.json',old/'metrics.raw.jsonl',ROOT/cfg['case_replay']['path']):
            inputs.append(dict(path=str(p),sha256=sha256(p)))
    aggregates=[];comp_aggregates=[]
    for panel in ['original','expanded']:
        pc=[c for c in cases if c['panel']==panel and c['selected']]
        for scope in SCOPES:
            for method in METHODS:
                obs=[r for r in all_rows if r['panel']==panel and r['scope']==scope and r['method']==method]
                medians={k:median([r['scopes'][scope][method][k] for r in pc]) for k in METRICS}
                aggregates.append(dict(panel=panel,scope=scope,method=method,cases=len(pc),medians=medians,
                    full_wins={k:sum(c['scopes'][scope]['target'][k]<c['scopes'][scope][method][k] for c in pc) for k in ERRORS},
                    target_rows=len(obs),target_both_below_zero=sum(all(r[k] is not None and r[k]<1 for k in ERRORS) for r in obs)))
            vals=[median([r['scopes'][scope]['composition_nll_relative_squared_error'] for r in compositions if (r['panel'],r['source_seed'],r['source_atom'],r['condition'])==(panel,c['source_seed'],c['source_atom'],c['condition'])]) for c in pc]
            comp_aggregates.append(dict(panel=panel,scope=scope,cases=len(pc),median_case_composition_nll_relative_squared_error=median(vals)))
    write(output/'component_summary.json',dict(cases=cases,aggregates=aggregates,composition_aggregates=comp_aggregates,compositions=compositions,checks=checks,
        model_forwards=forwards,wall_seconds=seconds,inputs=inputs,generator_script_sha256=sha256(Path(__file__)),
        scope='Exposed-data development; case then target dependent; actual standalone head+tail NLL effects test nonlinear composition, not heldout semantic prediction. Raw/global reused with exact full/top16 anchors; complement unequal size.'))
    with (output/'component_rows.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    lines=['# Component probability: all requested cases','','|Panel|Scope|Method|Cases|KL|NLL|Full wins KL/NLL|Targets both <1|','|---|---|---|---:|---:|---:|---:|---:|']
    for r in aggregates:lines.append('|'+ '|'.join([r['panel'],r['scope'],r['method'],str(r['cases']),fmt(r['medians'][ERRORS[0]]),fmt(r['medians'][ERRORS[1]]),f"{r['full_wins'][ERRORS[0]]}/{r['full_wins'][ERRORS[1]]}",f"{r['target_both_below_zero']}/{r['target_rows']}"])+'|')
    lines+=['','|Case|Status|Full KL/NLL|Top16|Tail|Random16|Native16|Dynamic atom|UOT tuned|','|---|---|---|---|---|---|---|---|---|']
    for c in cases:
        label=f"{c['panel']} {c['source_seed']}:{c['source_atom']} {c['condition']}"
        values=[' / '.join(fmt(c['scopes'][SCOPES[0]][m][k]) for k in ERRORS) if c['selected'] else 'NA' for m in ('target','readout_top16','readout_tail16','readout_random16','native_top16_difference_matched','single_atom_dynamic','uot_discovery_tuned')]
        lines.append('|'+ '|'.join([label,c['status']]+values)+'|')
    lines+=['','Composition test: actual full NLL change versus sum of separately measured head and tail NLL changes. Hook additivity alone does not guarantee this. No semantic or independent-replicate claim.']
    (output/'COMPONENT_TABLE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(checks=checks,aggregates=aggregates,composition_aggregates=comp_aggregates,model_forwards=forwards,wall_seconds=seconds)))


if __name__=='__main__':main()
