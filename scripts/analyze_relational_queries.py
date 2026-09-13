"""Reconstruct all new exclusion-query outcomes with paired uncertainty."""
from pathlib import Path
from collections import defaultdict
import argparse,json,itertools,statistics
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('out',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((a.run/'config.resolved.json').read_text());q=json.loads((a.run/'QUERY_FREEZE.json').read_text())['rows'];runtime=json.loads((a.run/'query_results.json').read_text())['rows']
    objs=cfg['objectives'];seeds=cfg['seeds'];families=cfg['query_families'];strategies=cfg.get('query_strategies',['full','difference','own_matched','soft_matched']);queries=[f'{k}>{l}' for k,l in itertools.permutations(range(3),2)];tasks=cfg['source_tasks'];lo,hi=cfg['consumer_ranges']['evaluation'];n=hi-lo
    shape=(len(objs),len(seeds),len(families),4,6,3,n);error=np.full(shape,np.nan);margin=np.full(shape,np.nan);norm=np.full(shape,np.nan);count=0
    for line in (a.run/'metrics.raw.jsonl').open():
        r=json.loads(line);assert r['kind']=='relational_query';count+=1
        key=(objs.index(r['objective']),seeds.index(r['seed']),families.index(r['family']),strategies.index(r['strategy']),queries.index(r['operation']),tasks.index(r['task']),r['row_id']-lo)
        assert np.isnan(error[key]);error[key]=int(r['introduced_error']);margin[key]=r['clean_margin']-r['margin'];norm[key]=r['edit_norm']
    assert np.isfinite(error).all() and np.isfinite(margin).all();weights=np.asarray([[int(t==k)-int(t==l) for t in range(3)] for k,l in itertools.permutations(range(3),2)])
    u=(error.mean(-1)*weights).sum(-1);mu=(margin.mean(-1)*weights).sum(-1);rows=[];maxerr=0
    for item in q:
        oi=objs.index(item['objective']);si=seeds.index(item['source']);fi=families.index(item['family']);st=strategies.index(item['strategy']);qi=queries.index(item['operation']);values=error[oi,si,fi,st,qi].mean(-1);m=margin[oi,si,fi,st,qi].mean(-1)
        prev=next(x for x in runtime if all(x[k]==item[k] for k in ['objective','source','family','strategy','operation']))
        maxerr=max(maxerr,abs(u[oi,si,fi,st,qi]-prev['selectivity']),abs(mu[oi,si,fi,st,qi]-prev['margin_selectivity']))
        rows.append(dict(**item,selectivity=float(u[oi,si,fi,st,qi]),margin_selectivity=float(mu[oi,si,fi,st,qi]),requested_error_rate=float(values[item['delete_task']]),preserved_error_rate=float(values[item['preserve_task']]),task_error_rates=values.tolist(),task_margin_decrements=m.tolist(),task_edit_norms=norm[oi,si,fi,st,qi].mean(-1).tolist()))
    assert maxerr<1e-12
    aggregates=[]
    for obj,family,strategy in itertools.product(objs,families,strategies):
        rr=[x for x in rows if x['objective']==obj and x['family']==family and x['strategy']==strategy]
        aggregates.append(dict(objective=obj,family=family,strategy=strategy,**{key:statistics.mean(x[key] for x in rr) for key in ['selectivity','margin_selectivity','requested_error_rate','preserved_error_rate','actual_members','shared_members']},by_query={query:{k:statistics.mean(x[k] for x in rr if x['operation']==query) for k in ['selectivity','margin_selectivity','requested_error_rate','preserved_error_rate']} for query in queries}))
    rng=np.random.default_rng(310913);contrasts=[]
    for oi,obj in enumerate(objs):
        for fi,family in enumerate(families):
            for method,comparator in cfg.get('query_contrasts',[('soft_matched','own_matched'),('difference','own_matched'),('soft_matched','full'),('difference','full')]):
                mi=strategies.index(method);ci=strategies.index(comparator);diff=error[oi,:,fi,mi]-error[oi,:,fi,ci];mdiff=margin[oi,:,fi,mi]-margin[oi,:,fi,ci];boots=[];mboots=[]
                for _ in range(4000):
                    ss=np.arange(len(seeds)) if cfg.get('leave_target_out') else rng.integers(0,len(seeds),len(seeds));v=diff[ss];mv=mdiff[ss];draws=[rng.integers(0,n,n) for _ in tasks]
                    vv=np.stack([v[:,:,j,:][:,:,draws[j]].mean(-1) for j in range(3)],-1);mm=np.stack([mv[:,:,j,:][:,:,draws[j]].mean(-1) for j in range(3)],-1)
                    boots.append(float((vv*weights).sum(-1).mean()));mboots.append(float((mm*weights).sum(-1).mean()))
                contrasts.append(dict(objective=obj,family=family,method=method,comparator=comparator,gain_points=float((u[oi,:,fi,mi]-u[oi,:,fi,ci]).mean()*100),paired95=(np.quantile(boots,[.025,.975])*100).tolist(),margin_gain=float((mu[oi,:,fi,mi]-mu[oi,:,fi,ci]).mean()),margin_paired95=np.quantile(mboots,[.025,.975]).tolist(),target_gains_points=((u[oi,:,fi,mi]-u[oi,:,fi,ci]).mean(-1)*100).tolist()))
    summary=dict(raw_rows=count,rows=rows,aggregates=aggregates,contrasts=contrasts,focused_check=dict(complete_tensor=True,raw_recompute_max_error=maxerr),scope=cfg['scope'],inference='Paired generated pairs within each fixed grammar; '+('dependent SAE network held fixed' if cfg.get('leave_target_out') else 'independent target seeds resampled, fixed source bank')+'; frozen prior calibration and all six ordered requests retained. '+cfg['primary_endpoint'])
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');np.savez_compressed(a.out/'paired_outcomes.npz',error=error,margin=margin,norm=norm,objectives=np.asarray(objs),families=np.asarray(families),strategies=np.asarray(strategies),queries=np.asarray(queries),tasks=np.asarray(tasks),seeds=np.asarray(seeds))
    for row in contrasts:
        if row['family'] in ['shared_path_full','response_query']:print(json.dumps(row))


if __name__=='__main__':main()
