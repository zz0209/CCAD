"""Paired uncertainty for the already-frozen joint member requests."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    cfg=json.loads((a.run/'config.resolved.json').read_text());lo,hi=cfg['consumer_ranges']['evaluation'];n=hi-lo
    objs=cfg['objectives'];families=cfg['structure_families'];seeds=cfg['seeds'];labels=['3','5','6','7'];tasks=cfg['source_tasks']
    data=np.full((len(objs),len(families),len(seeds),4,3,n),np.nan)
    for line in (a.run/'metrics.raw.jsonl').open():
        r=json.loads(line)
        if r['kind']!='functional_union':continue
        key=(objs.index(r['objective']),families.index(r['method']),seeds.index(r['seed']),labels.index(r['operation']),tasks.index(r['task']),r['row_id']-lo)
        assert np.isnan(data[key]);data[key]=int(r['introduced_error'])
    assert np.isfinite(data).all()
    weights=np.asarray([[.5,.5,-1.],[.5,-1.,.5],[-1.,.5,.5],[1/3,1/3,1/3]])
    u=(data.mean(-1)*weights).sum(-1);rng=np.random.default_rng(290914);result=[]
    for oi,obj in enumerate(objs):
        diff=data[oi,0]-data[oi,1]
        for request_ids in [[0],[1],[2],[3],[0,1,2]]:
            # Explicit take avoids advanced-index axis reordering for aggregate requests.
            seed_gains=np.take(u[oi,0]-u[oi,1],request_ids,axis=1).mean(1)
            boot=[]
            for _ in range(4000):
                ss=np.arange(len(seeds)) if cfg.get('leave_target_out') else rng.integers(0,len(seeds),len(seeds));v=diff[ss]
                vv=np.stack([v[:,:,j,:][:,:,rng.integers(0,n,n)].mean(-1) for j in range(3)],-1)
                request_gains=(vv*weights).sum(-1)
                boot.append(float(np.take(request_gains,request_ids,axis=1).mean()))
            result.append(dict(objective=obj,requests=[labels[i] for i in request_ids],gain_points=float(seed_gains.mean()*100),target_seed_gains_points=(seed_gains*100).tolist(),paired95=(np.quantile(boot,[.025,.975])*100).tolist(),shared_points=float(np.take(u[oi,0],request_ids,axis=1).mean()*100),cached_points=float(np.take(u[oi,1],request_ids,axis=1).mean()*100)))
    output=dict(rows=result,scope='Secondary analysis of every pre-evaluation frozen union request. Same paired target-seed/within-grammar bootstrap as singleton analysis; source bank and calibration decisions fixed. Pair requests average requested errors minus the unrequested task error. The triple reports mean disruption because all three tasks are requested. All individual queries and the equal mean of the three pair queries are retained; no multiplicity correction.')
    if len(families)>2:
        extra=[]
        for oi,obj in enumerate(objs):
            for ci,comparator in enumerate(families[2:],start=2):
                diff=data[oi,0]-data[oi,ci]
                for request_ids in [[0],[1],[2],[3],[0,1,2]]:
                    seed_gains=np.take(u[oi,0]-u[oi,ci],request_ids,axis=1).mean(1);boot=[]
                    for _ in range(4000):
                        ss=np.arange(len(seeds)) if cfg.get('leave_target_out') else rng.integers(0,len(seeds),len(seeds));v=diff[ss]
                        vv=np.stack([v[:,:,j,:][:,:,rng.integers(0,n,n)].mean(-1) for j in range(3)],-1)
                        boot.append(float(np.take((vv*weights).sum(-1),request_ids,axis=1).mean()))
                    extra.append(dict(objective=obj,comparator=comparator,requests=[labels[i] for i in request_ids],gain_points=float(seed_gains.mean()*100),target_seed_gains_points=(seed_gains*100).tolist(),paired95=(np.quantile(boot,[.025,.975])*100).tolist(),shared_points=float(np.take(u[oi,0],request_ids,axis=1).mean()*100),comparator_points=float(np.take(u[oi,ci],request_ids,axis=1).mean()*100)))
        output['additional_contrasts']=extra
    if cfg.get('leave_target_out'):output['scope']='Complete pre-frozen joint requests, with paired intervals resampling generated pairs within each grammar and holding the dependent five-SAE network and calibration choices fixed. All pair requests and their equal mean are retained; the triple is mean disruption without a collateral task. Target-specific gains are descriptive dependent effects. No multiplicity correction.'
    a.out.write_text(json.dumps(output,indent=2)+'\n')
    for r in result:print(json.dumps(r))


if __name__=='__main__':main()
