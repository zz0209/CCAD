"""Summarize full native-group operations and same-menu score decisions."""
from pathlib import Path
import argparse,json,statistics,hashlib,csv
from collections import defaultdict,Counter


def analyze(run,out):
    rows=[json.loads(s) for s in (run/'functional.raw.jsonl').read_text().splitlines()]
    grouped=defaultdict(list)
    for r in rows:grouped[r['query'],r['method'],r['stratum']].append(r)
    queries=[]
    for (q,m,stratum),rr in grouped.items():
        future=[r for r in rr if 'source_future_kl' in r]
        denominator=sum(r['source_effect_kl'] for r in rr)
        v=dict(query=q,objective=rr[0]['objective'],source_seed=rr[0]['source_seed'],target_seed=rr[0]['target_seed'],
            anchor=rr[0]['anchor'],method=m,stratum=stratum,rows=len(rr),
            source_kl=statistics.mean(r['source_kl'] for r in rr),source_effect=statistics.mean(r['source_effect_kl'] for r in rr),
            relative_kl=sum(r['source_kl'] for r in rr)/max(denominator,1e-15),
            nll_error=statistics.mean(abs(r['nll']-r['source_group_nll']) for r in rr),
            future_rows=len(future),future_kl=statistics.mean(r['source_future_kl'] for r in future) if future else None,
            future_effect=statistics.mean(r['source_future_effect_kl'] for r in future) if future else None,
            future_relative_kl=sum(r['source_future_kl'] for r in future)/max(sum(r['source_future_effect_kl'] for r in future),1e-15) if future else None,
            future_nll_error=statistics.mean(r['future_nll_error'] for r in future) if future else None)
        queries.append(v)
    aggregate={}
    for obj in sorted({r['objective'] for r in queries}):
        aggregate[obj]={}
        for stratum in ['source_active','uniform']:
            aggregate[obj][stratum]={}
            for method in sorted({r['method'] for r in queries}):
                qq=[r for r in queries if r['objective']==obj and r['method']==method and r['stratum']==stratum]
                v={key:statistics.mean(r[key] for r in qq if r[key] is not None) if any(r[key] is not None for r in qq) else None
                   for key in ['relative_kl','source_kl','source_effect','nll_error','future_relative_kl','future_kl','future_effect','future_nll_error']}
                v.update(queries=len(qq),rows=sum(r['rows'] for r in qq),future_rows=sum(r['future_rows'] for r in qq),
                    pooled_relative_kl=sum(r['source_kl']*r['rows'] for r in qq)/max(sum(r['source_effect']*r['rows'] for r in qq),1e-15))
                if v['future_kl'] is not None:v['future_pooled_relative_kl']=sum(r['future_kl']*r['future_rows'] for r in qq)/max(sum(r['future_effect']*r['future_rows'] for r in qq),1e-15)
                aggregate[obj][stratum][method]=v
    fit_file=run/('finite_fits.json' if (run/'finite_fits.json').exists() else 'response_fits.json')
    fits=json.loads(fit_file.read_text())['queries'];decisions=[]
    lookup={(r['query'],r['method'],r['stratum']):r for r in queries}
    for q in fits:
        candidates=sorted(q['scores'])
        for metric in ['euclidean','fisher']:
            choice=min(candidates,key=lambda m:(q['scores'][m][metric]['error'],m))
            v=lookup[q['query'],choice,'source_active']
            oracle=min(lookup[q['query'],m,'source_active']['relative_kl'] for m in candidates)
            decisions.append(dict(query=q['query'],objective=q['objective'],score=metric,choice=choice,candidates=candidates,
                relative_kl=v['relative_kl'],regret=v['relative_kl']-oracle,nll_error=v['nll_error'],future_relative_kl=v['future_relative_kl']))
    selectors={}
    for obj in aggregate:
        selectors[obj]={}
        for metric in ['euclidean','fisher']:
            dd=[d for d in decisions if d['objective']==obj and d['score']==metric]
            selectors[obj][metric]=dict(queries=len(dd),choices=dict(Counter(d['choice'] for d in dd)),
                mean_relative_kl=statistics.mean(d['relative_kl'] for d in dd),mean_regret=statistics.mean(d['regret'] for d in dd))
    comparisons={}
    for obj in aggregate:
        comparisons[obj]={}
        if 'finite_gates' not in aggregate[obj]['source_active']:continue
        for comparator in ['balanced_euclidean','balanced_fisher','finite_amplitude','target_group_rank1']:
            pairs=[(v,lookup[v['query'],comparator,v['stratum']]) for v in queries if v['objective']==obj and v['method']=='finite_gates' and v['stratum']=='source_active']
            comparisons[obj][comparator]={metric:dict(wins=sum(a[metric]<b[metric] for a,b in pairs if a[metric] is not None),
                pairs=sum(a[metric] is not None for a,b in pairs),mean_difference=statistics.mean(a[metric]-b[metric] for a,b in pairs if a[metric] is not None))
                for metric in ['relative_kl','nll_error','future_relative_kl','future_nll_error']}
    result=dict(run=str(run),scope=json.loads((run/'config.resolved.json').read_text())['scope'],
        inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [run/'functional.raw.jsonl',fit_file]],
        aggregate=aggregate,comparisons=comparisons,selectors=selectors,decisions=decisions,queries=queries,
        inference='Descriptive paired queries conditional on one shared seed edge and reused documents; no independent-seed interval or confirmation claim.')
    out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'query_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(queries[0]));w.writeheader();w.writerows(queries)
    print(json.dumps(dict(aggregate=aggregate,comparisons=comparisons,selectors=selectors),indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();analyze(a.run,a.out)
