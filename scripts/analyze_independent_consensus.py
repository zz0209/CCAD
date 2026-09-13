"""Analyze frozen target choices and functional regions with paired units."""
import argparse,csv,json,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
from analyze_functional_reuse import analyze


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    base=analyze(a.run,a.out);cfg=json.loads((a.run/'config.resolved.json').read_text())
    frozen=json.loads((a.run/'SELECTION_FREEZE.json').read_text())['decisions']
    structs=json.loads((a.run/'STRUCTURE_FREEZE.json').read_text())['rows']
    primary=[r for r in frozen if r['validation_pairs_per_task']==cfg['primary_validation_budget']]
    chosen={(r['objective'],r['source'],r['operation'],r['family']):r['chosen_allowance'] for r in primary}
    ops=sorted(cfg['operations'],key=lambda x:cfg['operations'][x].index(1));families=cfg['methods'];seeds=cfg['seeds'];tasks=cfg['source_tasks']
    lo,hi=cfg['consumer_ranges']['evaluation'];n=hi-lo
    shape=(len(cfg['objectives']),len(families),len(seeds),3,3,n)
    errors=np.full(shape,np.nan);margins=np.full(shape,np.nan);regionraw=defaultdict(list)
    rawcount=0
    for line in (a.run/'metrics.raw.jsonl').open():
        r=json.loads(line);rawcount+=1
        if r['split']!='evaluation':continue
        key=(r['objective'],r['seed'],r['operation'],r['method'])
        if r['kind']=='functional_reuse' and chosen.get(key)==r['allowance']:
            idx=(cfg['objectives'].index(r['objective']),families.index(r['method']),seeds.index(r['seed']),ops.index(r['operation']),tasks.index(r['task']),r['row_id']-lo)
            assert np.isnan(errors[idx]);errors[idx]=int(r['introduced_error']);margins[idx]=r['clean_margin']-r['margin']
        elif r['kind'] in ['functional_region','functional_union']:
            regionraw[r['objective'],r['seed'],r['kind'].removeprefix('functional_'),r['method'],r['operation'],r['task']].append(r)
    assert np.isfinite(errors).all() and np.isfinite(margins).all()
    weights=np.full((3,3),-.5);np.fill_diagonal(weights,1.)
    u=(errors.mean(-1)*weights).sum(-1).mean(-1)
    # The same generated pairs are shared by every target/request/method.
    rng=np.random.default_rng(290913);replicates=4000
    contrasts=[]
    for oi,obj in enumerate(cfg['objectives']):
        for mi,method in enumerate(families):
            for comparator in ['cached64_full','shared_scalar_full']:
                if method==comparator:continue
                cj=families.index(comparator);diff=errors[oi,mi]-errors[oi,cj]
                seed_diff=u[oi,mi]-u[oi,cj];boot=[]
                for _ in range(replicates):
                    ss=rng.integers(0,len(seeds),len(seeds));v=diff[ss]
                    vv=np.stack([v[:,:,j,rng.integers(0,n,n)].mean(-1) for j in range(3)],-1)
                    boot.append(float((vv*weights).sum(-1).mean()))
                contrasts.append(dict(objective=obj,method=method,comparator=comparator,gain_points=float(seed_diff.mean()*100),target_seed_gains_points=(seed_diff*100).tolist(),paired_target_and_within_grammar_percentile95=(np.quantile(boot,[.025,.975])*100).tolist(),replicates=replicates))
    byfunction=[]
    for obj in cfg['objectives']:
        for family in families:
            for op in ops:
                rr=[r for r in base['selected'] if r['objective']==obj and r['method']==family and r['operation']==op and r['validation_pairs_per_task']==cfg['primary_validation_budget']]
                byfunction.append(dict(objective=obj,method=family,operation=op,selectivity=statistics.mean(r['selectivity'] for r in rr),requested_error_rate=statistics.mean(r['requested_error_rate'] for r in rr),collateral_error_rate=statistics.mean(r['collateral_error_rate'] for r in rr)))
    rebuilt=[];runtime=json.loads((a.run/'structure_results.json').read_text())['rows'];maxerror=0.
    for detail in structs:
        rates={};decrements={}
        for task in tasks:
            rr=regionraw[detail['objective'],detail['source'],detail['kind'],detail['family'],detail['operation'],task]
            assert len(rr)==n and len({r['row_id'] for r in rr})==n
            rates[task]=statistics.mean(r['introduced_error'] for r in rr)
            decrements[task]=statistics.mean(r['clean_margin']-r['margin'] for r in rr)
        own=[tasks[k] for k in detail['requested']];other=[t for t in tasks if t not in own]
        req=statistics.mean(rates[t] for t in own);col=statistics.mean(rates[t] for t in other) if other else 0.
        rp=statistics.mean(decrements[t] for t in own);cp=statistics.mean(max(decrements[t],0) for t in other) if other else 0.
        item=dict(**detail,requested_error_rate=req,collateral_error_rate=col,selectivity=req-col,task_rates=rates,task_margin_decrements=decrements,margin_selectivity=rp-cp)
        prior=next(r for r in runtime if all(r[k]==detail[k] for k in ['objective','source','kind','family','operation']))
        maxerror=max(maxerror,abs(item['selectivity']-prior['selectivity']),abs(item['margin_selectivity']-prior['margin_selectivity']))
        rebuilt.append(item)
    aggregate=[]
    for obj in cfg['objectives']:
        for family in cfg['structure_families']:
            for kind in ['region','union']:
                labels=sorted({r['operation'] for r in rebuilt if r['kind']==kind},key=int)
                for label in labels:
                    rr=[r for r in rebuilt if r['objective']==obj and r['family']==family and r['kind']==kind and r['operation']==label]
                    aggregate.append(dict(objective=obj,family=family,kind=kind,label=label,requested=rr[0]['requested'],mean_members=statistics.mean(r['actual_members'] for r in rr),nonempty_targets=sum(r['actual_members']>0 for r in rr),selectivity=statistics.mean(r['selectivity'] for r in rr),margin_selectivity=statistics.mean(r['margin_selectivity'] for r in rr),task_rates={t:statistics.mean(r['task_rates'][t] for r in rr) for t in tasks},task_margin_decrements={t:statistics.mean(r['task_margin_decrements'][t] for r in rr) for t in tasks}))
    empirical=json.loads((a.run/'consumer_results.json').read_text())['selected']
    for r in base['selected']:
        prev=next(x for x in empirical if x['objective']==r['objective'] and x['source']==r['seed'] and x['operation']==r['operation'] and x['family']==r['method'] and x['validation_pairs_per_task']==r['validation_pairs_per_task'])
        maxerror=max(maxerror,abs(prev['selectivity']-r['selectivity']))
    assert maxerror<1e-12,maxerror
    summary=dict(status=base['status'],raw_rows=rawcount,aggregates=base['aggregates'],by_function=byfunction,contrasts=contrasts,structures=rebuilt,structure_aggregates=aggregate,focused_check=dict(maximum_raw_recompute_difference=maxerror,complete_primary_tensor=True,structural_rows=len(rebuilt)),scope='Intervals resample target initializations and generated pairs independently within each fixed grammar, with each draw shared across methods/requests. Conditional on the five-source bank, training corpus/model/hook, three grammars, and realized calibration choices. No cross-grammar or source-bank population claim; no multiple-testing-adjusted claims. Primary shared path versus cached64 and identically aggregated scalar are fixed in the config.')
    (a.out/'independent_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(a.out/'primary_paired_outcomes.npz',errors=errors,margin_decrements=margins,objectives=np.asarray(cfg['objectives']),families=np.asarray(families),source_index=np.asarray(seeds),target_seeds=np.asarray(cfg['target_seeds']),operations=np.asarray(ops),tasks=np.asarray(tasks),row_ids=np.arange(lo,hi))
    for r in summary['aggregates']:
        if r['validation_pairs_per_task']==cfg['primary_validation_budget']:print(json.dumps(r))
    for r in contrasts:
        if r['method']=='shared_path_full':print(json.dumps(r))


if __name__=='__main__':main()
