from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np
from analyze_profile_confirmation import human_arrays, scalar_arrays, query_order, families as request_families


def summarize(nums, den, dw, names):
    rng=np.random.default_rng(2026092031)
    families=request_families(names,True)
    records={}; samples={}
    for family,ix in families.items():
        if not ix: continue
        queries=rng.choice(ix,size=(2000,len(ix)))
        for method,num in nums.items():
            per=np.sqrt(num[0].mean(-1)/den.mean(-1))
            boot=[]
            for h,weights in enumerate(dw):
                denominator=weights@den[h].T
                ratio=np.full_like(denominator,np.nan)
                np.divide(weights@num[0,h].T,denominator,out=ratio,where=denominator>0)
                boot.append(np.take_along_axis(np.sqrt(ratio),queries,axis=1).mean(1))
            sample=np.mean(boot,axis=0)
            valid=np.isfinite(sample)
            samples[method,family]=sample
            records.setdefault(method,{})[family]=dict(nrmse=float(per[:,ix].mean()),
                interval=np.quantile(sample[valid],[.025,.975]).tolist(),valid_draws=int(valid.sum()))
    chosen='initial_refined_conditional_source_metric'
    contrasts=[]
    for reference in nums:
        if reference==chosen: continue
        for family in records[chosen]:
            diff=samples[reference,family]-samples[chosen,family]
            valid=np.isfinite(diff)
            contrasts.append(dict(reference=reference,family=family,
                reduction=records[reference][family]['nrmse']-records[chosen][family]['nrmse'],
                interval=np.quantile(diff[valid],[.025,.975]).tolist(),valid_draws=int(valid.sum())))
    return dict(summary=records,contrasts=contrasts,families=families)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--direct-reference',type=Path)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    assert json.loads((a.run/'status.json').read_text())['status']=='PASS'
    membership=json.loads((a.run/'membership.json').read_text())
    names=query_order(a.run,'human')
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),run=str(a.run),queries=names,
                scope='Development on128 previously evaluated biographies, one target seed, four frozen later heads; paired document/request intervals conditional on these assets')
    num,den,dw,_,extra=human_arrays([a.run],membership)
    result['later_heads']=summarize(num,den,dw,names)
    result['later_heads']['provenance']=extra
    num,den,dw,_,_=scalar_arrays([a.run],membership,'human')
    result['original_head']=summarize(num,den,dw,names)
    with np.load(a.run/'responses.npz') as arr:
        reference=a.direct_reference or a.run
        ref=np.load(reference/'responses.npz')
        assert np.array_equal(ref['human__source'],arr['human__source'])
        assert np.array_equal(ref['human__none'],arr['human__none'])
        direct=ref['human__initial_refined_source_metric']
        cached=arr['human__initial_refined_cached_source_metric']
        result['cache_control']=dict(maximum_response_difference=float(np.abs(direct-cached).max()),
            rms_difference=float(np.sqrt(np.mean((direct-cached)**2))),
            rms_source_effect=float(np.sqrt(np.mean((arr['human__source']-arr['human__none'])**2))),
            direct_reference=str(reference),reference_sha256=hashlib.sha256((reference/'responses.npz').read_bytes()).hexdigest())
    result['prediction_sha256']=hashlib.sha256((a.run/'responses.npz').read_bytes()).hexdigest()
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({s:{m:v['participation']['nrmse'] for m,v in result[s]['summary'].items()} for s in ['later_heads','original_head']},indent=2))
    print(json.dumps(result['cache_control']))


if __name__=='__main__':
    main()
