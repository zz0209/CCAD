"""Complete conditional composition comparisons, separating role and transfer."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from run_r011s1_raw_hook_asset import ROOT,write_json as write,entry


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',required=True)
    args=ap.parse_args()
    run=ROOT/'runs'/args.run
    out=ROOT/'artifacts/calendar_composition_20260906'
    out.mkdir(exist_ok=True)
    load=lambda x:json.loads((run/x).read_text())
    if load('status.json')['status']!='PASS':
        raise ValueError('Composition run is incomplete or failed')
    prompts=load('prompts_and_pairs.json')['rows']
    base=load('baseline_results.json')['rows']
    rr=[json.loads(s) for s in (run/'metrics.raw.jsonl').read_text().splitlines()]
    rows=[r for r in rr if r.get('record_kind')=='conditional_composition']
    baseline=[]
    for template in dict.fromkeys(r['template'] for r in prompts):
        p=next(r for r in prompts if r['template']==template)
        a=[r for r in base if r['template']==template]
        baseline.append(dict(template=template,role=p['role'],phase=p['phase'],n=len(a),
            family_correct=sum(r['family_correct'] for r in a),full_correct=sum(r['full_correct'] for r in a),
            mean_family_mass=float(np.mean([r['family_mass'] for r in a]))))
    groups=defaultdict(list)
    for r in rows:
        for scope,key in [('role',r['role']),('template',r['template']),('all','all')]:
            groups[scope,key,r['operation'],r['source_seed'],r['target_seed'],r['method']].append(r)
    statistics=[]
    for (scope,key,operation,s,t,method),a in groups.items():
        get=lambda name:np.array([r[name] for r in a],dtype=float)
        denominator=float(get('reference_noop_kl').sum())
        pn=float(get('conditional_noop_kl').sum())
        rawenergy=float(get('raw_vector_energy').sum())
        sourceenergy=float(get('source_vector_energy').sum())
        v=dict(scope=scope,key=key,operation=operation,source=s,target=t,method=method,n=len(a),
            mean_conditional_kl=float(get('primary_conditional_kl').mean()),
            aggregate_conditional_kl_over_noop=float(get('primary_conditional_kl').sum()/pn) if pn>1e-15 else None,
            mean_reference_kl=float(get('reference_kl').mean()),
            aggregate_reference_kl_over_noop=float(get('reference_kl').sum()/denominator) if denominator>1e-15 else None,
            mean_intervention_kl=float(get('intervention_kl').mean()),
            raw_vector_relative_squared_error=float(get('vector_error_to_raw').sum()/rawenergy),
            candidate_over_raw_vector_energy=float(get('vector_energy').sum()/rawenergy),
            source_vector_relative_squared_error=float(get('vector_error_to_source').sum()/sourceenergy),
            donor_margin_change_median=float(np.median(get('donor_margin_change'))),
            raw_margin_change_median=float(np.median(get('raw_margin_change'))),
            raw_margin_relative_squared_error=float(np.sum((get('donor_margin_change')-get('raw_margin_change'))**2)/np.sum(get('raw_margin_change')**2)),
            margin_change_energy_over_raw=float(np.sum(get('donor_margin_change')**2)/np.sum(get('raw_margin_change')**2)),
            donor_family_correct=sum(r['donor_family_correct'] for r in a))
        statistics.append(v)
    # Summarize a fixed member display without using test outcomes.
    sources=load('composition_sources.json')['sources']
    instance=next(r for r in sources if r['source_seed']==1)
    ar=np.load(run/'codes_seed1.npz')
    slots=np.array([r['slot'] for r in prompts])
    z=ar['codes'][np.arange(len(prompts)),slots].astype(float)
    dec=ar['decoder'].astype(float)
    ids=np.array(instance['source_ids'])
    coef=np.array(instance['source_coefficients'])
    profiles={}
    for role in ['calendar','noncalendar']:
        a=np.array([np.mean(z[[r['id'] for r in prompts if r['phase']=='fit' and r['role']==role and r['value_index']==v]],axis=0) for v in range(12)])
        profiles[role]=a[:,ids]
    score=np.var(profiles['calendar'],axis=0)*coef**2*np.sum(dec[ids]**2,axis=1)
    order=np.argsort(-score,kind='stable')[:16]
    display=dict(source=1,rule='top16 source-fit calendar contribution variance among the already frozen64; no test outcomes',
        atoms=ids[order].tolist(),coefficients=coef[order].tolist(),
        calendar_activation=profiles['calendar'][:,order].T.tolist(),
        noncalendar_activation=profiles['noncalendar'][:,order].T.tolist(),
        labels=[r['value'] for r in prompts if r['template']=='fit_calendar_this'],
        positive_fit_variance=score[order].tolist())
    example_id=next(p['pair_id'] for p in load('prompts_and_pairs.json')['pairs'] if
        prompts[p['recipient']]['template']=='test_calendar_our' and prompts[p['recipient']]['value_index']==0 and p['offset']==1)
    example=[r for r in rows if r['pair_id']==example_id and r['source_seed']==1]
    summary=dict(run=args.run,baseline=baseline,statistics=statistics,source_groups=sources,display=display,
        fixed_example=example,cost=load('metrics.summary.json'),environment=load('environment.json'),
        scope='source supervised conditional masks; 8 new-prefix test templates; same12values; dependent five seeds and20directions',
        raw_baseline_dependence='For a fixed source, raw-linear map is identical across target seeds because shared hook inputs are identical',
        metric_definition='positive KL(raw patch||candidate); negative KL(base||candidate); transferred reference is actual source patch')
    write(out/'summary.json',summary)
    sourcepaths=[run/x for x in ['config.resolved.json','metrics.raw.jsonl','metrics.summary.json','environment.json',
        'composition_sources.json','composition_maps.json','composition_summary.json','prompts_and_pairs.json',
        'baseline_results.json','operation_checks.json','inputs.json','code_hashes.json']]
    write(out/'source_manifest.json',dict(files=[entry(p,'CCAD frozen conditional composition run','source') for p in sourcepaths],
        analysis_script=entry(Path(__file__),'CCAD','analysis')))
    small=[r for r in statistics if r['scope']=='role' and r['operation']=='source']
    aggregate={}
    for method in dict.fromkeys(r['method'] for r in small):
        b=[r for r in small if r['method']==method]
        aggregate[method]={role:{k:float(np.median([r[k] for r in b if r['key']==role])) for k in
            ['mean_conditional_kl','raw_vector_relative_squared_error','candidate_over_raw_vector_energy','raw_margin_relative_squared_error','margin_change_energy_over_raw']} for role in ['calendar','noncalendar']}
    write(out/'compact_summary.json',dict(baseline=baseline,source_seed_medians=aggregate,
        cross_seed_medians={method:float(np.median([r['aggregate_reference_kl_over_noop'] for r in statistics if r['scope']=='all' and r['operation']=='cross_seed' and r['method']==method])) for method in ['native64','best_native_atom','raw_linear','target_full']}))
    print(json.dumps(dict(baseline=baseline,source_seed_medians=aggregate,
        cross_seed_medians=json.loads((out/'compact_summary.json').read_text())['cross_seed_medians']),indent=2))


if __name__=='__main__':
    main()
