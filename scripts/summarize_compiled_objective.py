"""Compare geometric and output fitting of the same retained native supports.

This reads completed outputs only. It neither fits a new operator nor selects
a task. Hidden error is measured against the common requested target readout,
not silently substituted for error against the true source intervention.
"""
from pathlib import Path
import argparse,json,csv,hashlib,time
from collections import defaultdict
from datetime import datetime,timezone
import numpy as np

def scalar_from_write(operator,write):
    supports=operator['positive_negative_members'];weights=operator['positive_negative_code_weights'].astype(float)
    positive=np.all(write['members']==supports[0],axis=1);negative=np.all(write['members']==supports[1],axis=1)
    assert np.all(positive|negative),'An edit does not use its recorded fixed support'
    sign=np.where(positive,0,1);w=weights[sign];den=(w*w).sum(1);assert np.all(den>0)
    magnitude=(w*write['code_increment']).sum(1)/den
    assert np.allclose(write['code_increment'],magnitude[:,None]*w,rtol=1e-5,atol=1e-6)
    return sign,magnitude

def summarize(run):
    assert json.loads((run/'status.json').read_text())['status']=='PASS',run
    choices=json.loads((run/'selection_choices.json').read_text())['choices'];metrics=defaultdict(dict);records=[];inputs=[]
    for line in (run/'metrics.raw.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['method'] in ['compiled_shared_axis','compiled_functional_axis']:metrics[(r['query'],r['method'])][r['row_id']]=r
    for c in choices:
        if 'compiled_functional_axis' not in c['held_summary']:continue
        query=c['query'];ops={};writes={}
        for method,suffix in [('compiled_shared_axis','_compiled_axis_operator.npz'),('compiled_functional_axis','_compiled_functional_axis_operator.npz')]:
            for dest,name in [(ops,query+suffix),(writes,query+'_'+method+'_write.npz')]:
                path=run/name
                with np.load(path,allow_pickle=False) as z:dest[method]={k:v.copy() for k,v in z.items()}
                inputs.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        geometric=ops['compiled_shared_axis'];functional=ops['compiled_functional_axis']
        for key in ['source_axis','target_reader','positive_negative_members']:assert np.array_equal(geometric[key],functional[key]),(query,key)
        coefficients=[]
        for method in ops:
            op=ops[method];write=writes[method];sign,magnitude=scalar_from_write(op,write);coefficients.append(np.where(sign==0,magnitude,-magnitude))
            held=metrics[query,method];mask=np.array([int(i) in held for i in write['row_ids']]);assert mask.sum()==len(held)
            unit=op['source_axis'][:,0].astype(float);ray_error=np.linalg.norm(op['realized_unit_directions'].astype(float)-np.stack([unit,-unit]),axis=1)
            norms=magnitude[mask]*ray_error[sign[mask]];rows=list(held.values())
            records.append(dict(
                run=run.name,query=query,task=c['task'],objective=c['objective'],
                source_seed=c['source_seed'],target_seed=c['target_seed'],method=method,
                held_rows=len(rows),positive_sign_rows=int(np.sum(sign[mask]==0)),
                zero_scalar_rows=int(np.sum(magnitude[mask]==0)),
                unit_ray_error_positive=float(ray_error[0]),unit_ray_error_negative=float(ray_error[1]),
                requested_hidden_rmse=float(np.sqrt(np.mean(norms**2))),
                requested_hidden_mean_norm_error=float(norms.mean()),
                mean_source_kl=float(np.mean([r['source_kl'] for r in rows])),
                iia=float(np.mean([r['iia'] for r in rows])),
                donor_ce=float(np.mean([r['donor_ce'] for r in rows]))))
        assert np.array_equal(writes['compiled_shared_axis']['row_ids'],writes['compiled_functional_axis']['row_ids'])
        assert np.allclose(coefficients[0],coefficients[1],rtol=1e-5,atol=1e-6),(query,'Reader coordinate changed')
    return records,inputs

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();timer=time.perf_counter();records=[];inputs=[]
    for run in a.runs:
        r,i=summarize(run);records+=r;inputs+=i
        for name in ['metrics.raw.jsonl','selection_choices.json','config.resolved.json']:
            path=run/name;inputs.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    grouped=defaultdict(dict)
    for r in records:grouped[r['run'],r['query']][r['method']]=r
    changes=[]
    for (run,query),methods in grouped.items():
        g=methods['compiled_shared_axis'];f=methods['compiled_functional_axis']
        changes.append(dict(run=run,query=query,objective=g['objective'],hidden_rmse_change=f['requested_hidden_rmse']-g['requested_hidden_rmse'],source_kl_change=f['mean_source_kl']-g['mean_source_kl'],iia_change=f['iia']-g['iia'],geometric=g,functional=f))
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'query_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),queries=len(changes),comparisons=changes,larger_hidden_error=int(sum(r['hidden_rmse_change']>0 for r in changes)),lower_source_kl=int(sum(r['source_kl_change']<0 for r in changes)),higher_iia=int(sum(r['iia_change']>0 for r in changes)),input_files=inputs,wall_seconds=time.perf_counter()-timer,scope='Paired retained-output description at identical source axis, target reader, fixed target supports and write allowance. Hidden RMSE is against the common requested readout alpha*u, not the true source edit. Fitting information differs: the functional variant adds source training outputs. This comparison establishes an objective tradeoff, not a unique cause, an independent development-data confirmation, or superiority over unrestricted distillation.')
    (a.output/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['comparisons','input_files']}))
    for r in changes:print(json.dumps({k:v for k,v in r.items() if k not in ['geometric','functional']}))
if __name__=='__main__':main()
