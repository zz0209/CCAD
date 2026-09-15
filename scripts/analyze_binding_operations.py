"""Keep replacement success and source-deletion fidelity as separate endpoints."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np


def analyze(run,output):
    status=json.loads((run/'status.json').read_text());assert status['status']=='PASS'
    raw=run/'metrics.raw.jsonl';records=[json.loads(s) for s in raw.read_text().splitlines()]
    records=[r for r in records if r['kind']=='intervention']
    key=lambda r:(r['seed'],r['row_id'],r['operation'])
    source={op:{key(r):r for r in records if r['method']=='source_'+op} for op in ['replace','delete']}
    contexts=sorted({r['component'] for r in records});n=len(contexts)
    weights=np.random.default_rng(9481501).multinomial(n,np.full(n,1/n),10000)/n
    def summarize(x):
        assert x.shape==(n,) and np.isfinite(x).all()
        return dict(mean=float(x.mean()),interval=np.quantile(weights@x,[.025,.975]).tolist())
    cells=[];by_context={};payload={}
    for method in sorted({r['method'] for r in records}):
        op=method.split('_')[1] if method.startswith('source_') else method.split('_')[0]
        family='source' if method.startswith('source_') else method.split('_',1)[1]
        rr=[r for r in records if r['method']==method]
        assert len({key(r) for r in rr})==len(rr)
        for site in ['all','first','second','both']:
            values=[]
            for cid in contexts:
                rows=[r for r in rr if r['component']==cid and (site=='all' or r['operation']==site)]
                pair={};eqpair={};asked=[];protected=[];same=[];errors=[]
                for r in rows:
                    s=source[op][key(r)];assert r['expected_id']==s['expected_id']
                    k=(r['seed'],r['task'],r['operation'])
                    pair.setdefault(k,[]).append(r['correct']);eqpair.setdefault(k,[]).append(r['answer_id']==s['answer_id'])
                    if r['requested']:
                        asked.append(r['correct']);same.append(r['answer_id']==s['answer_id'])
                        errors.append(abs(r['expected_log_probability']-s['expected_log_probability']))
                    else:protected.append(r['correct'])
                assert all(len(v)==2 for v in pair.values()) and asked
                values.append([np.mean([all(v) for v in pair.values()]),np.mean(same),np.mean(asked),
                               np.mean(protected) if protected else 1.,np.mean(errors),np.mean([all(v) for v in eqpair.values()])])
            values=np.array(values);by_context[method,site]=values
            payload[method+'|'+site]=values
            cell=dict(method=method,family=family,kind=op,sites=site,contexts=n,
                      source_seeds=sorted({r['seed'] for r in rr}),
                      complete_expected_answer_pair=summarize(values[:,0]),requested_source_answer=summarize(values[:,1]),
                      requested_expected_answer=summarize(values[:,2]),requested_source_logp_mae=summarize(values[:,4]),
                      source_complete_pair=summarize(values[:,5]))
            if site!='both':cell['protected_expected_answer']=summarize(values[:,3])
            cells.append(cell)
    comparisons=[('absolute','difference'),('joint','difference'),('anchor_code','absolute'),
                 ('anchor_code','direct'),('anchor_code','gain'),('anchor_raw','absolute'),
                 ('anchor_exact','absolute'),('anchor_code','readout_code'),('anchor_raw','readout_raw')]
    contrasts=[]
    for op in ['replace','delete']:
        for method,ref in comparisons:
            for site in ['all','both']:
                mk=(op+'_'+method,site);rk=(op+'_'+ref,site)
                if mk not in by_context or rk not in by_context:continue
                d=by_context[mk]-by_context[rk]
                contrasts.append(dict(kind=op,method=method,reference=ref,sites=site,
                    complete_expected_pair_difference_points=summarize(100*d[:,0]),
                    requested_source_answer_difference_points=summarize(100*d[:,1]),
                    source_logp_mae_difference=summarize(d[:,4])))
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),run=run.as_posix(),status=status,
                raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),cells=cells,contrasts=contrasts,
                statistics='10,000 paired context resamples, seed9481501; forms, entity queries, operation sites and fixed dependent SAE directions kept together.',
                scope=json.loads((run/'config.resolved.json').read_text())['scope'],
                endpoint_definition='Replacement expected answer is donor city at requested sites and original city elsewhere; deletion expected answer is original city, so its complete_expected_answer_pair means retention. Source fidelity compares actual full-vocabulary answers.')
    output.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(output.with_suffix('.npz'),contexts=np.array(contexts),**payload)
    print(json.dumps([dict(method=c['method'],complete_expected=100*c['complete_expected_answer_pair']['mean'],requested_source=100*c['requested_source_answer']['mean']) for c in cells if c['sites']=='all']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.run,a.output)
