"""Measure the actual dimensionality of source-native contribution fields."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import json,hashlib,time
import numpy as np,torch


def main():
    start=time.perf_counter();root=Path(__file__).resolve().parents[1]
    run=root/'runs/REFORM_R22_native_coarsening_function_v4_20260913'
    cfg=json.loads((run/'config.resolved.json').read_text());ref=root/cfg['reference_run'];rc=json.loads((ref/'config.resolved.json').read_text())
    snaps=json.loads((root/rc['training_run']/'checkpoints.json').read_text())['checkpoints'];queries=json.loads((run/'query_results.json').read_text())['queries']
    decoders={};sparse={};inputs=[];rows=[]
    def checked(p):
        inputs.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()));return p
    for q in queries:
        key=q['objective'],q['source_seed']
        if key not in decoders:
            snap=next(s for s in snaps if (s['objective'],s['seed'])==key and s['step']==rc['checkpoint_step']);p=checked(Path(snap['path']));assert inputs[-1]['sha256']==snap['sha256']
            state=torch.load(p,map_location='cpu',weights_only=True);decoders[key]=(state['decoder.weight'].T if key[0]=='topk' else state['W_dec']).numpy()
            with np.load(checked(ref/f'natural_discovery_{key[0]}_seed{key[1]}.npz')) as a:sparse[key]={k:a[k] for k in a.files}
        with np.load(checked(run/(q['query']+'_groups.npz'))) as a:sp=a['source_members'];gs=a['source_gate']
        ar=sparse[key];lookup=np.full(int(ar['shape'][1]),-1);lookup[sp]=np.arange(len(sp));cols=lookup[ar['columns']];valid=cols>=0
        z=np.zeros((int(ar['shape'][0]),len(sp)));z[ar['rows'][valid],cols[valid]]=ar['values'][valid];d=decoders[key][sp].astype('float64');anchor=d[int(np.where(sp==q['anchor'])[0][0])].copy();anchor/=np.linalg.norm(anchor)
        A=(z*gs).T@(z*gs)/len(z);B=d@d.T;value,Q=np.linalg.eigh(A);sq=(Q*np.sqrt(np.maximum(value,0)))@Q.T
        ev=np.maximum(np.linalg.eigvalsh(sq@B@sq),0)[::-1];energy=ev.sum();v=d@anchor
        record=dict(query=q['query'],objective=key[0],discovery_rows=len(z),source_members=len(sp),source_energy=float(energy),
            discovery_rank1_fraction=float(ev[0]/energy),discovery_rank2_fraction=float(ev[:2].sum()/energy),discovery_anchor_span_fraction=float(v@A@v/energy))
        with np.load(checked(run/(q['query']+'_functional_edits.npz'))) as a:
            x=a['source_group'][a['strata']=='source_active'].astype('float64');ss=np.linalg.svd(x,compute_uv=False);record['calibration_active_rank1_fraction']=float(ss[0]**2/(x*x).sum())
        rows.append(record)
    out=root/'artifacts/scientific_reform_20260913/r23_source_dimension';out.mkdir(exist_ok=True)
    result=dict(rows=rows,inputs=inputs,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),wall_seconds=time.perf_counter()-start,
        scope='Post-outcome geometry diagnosis of the retained R22 source operators, not new model evidence or an independent experiment.',
        summary={obj:{metric:float(np.mean([r[metric] for r in rows if r['objective']==obj])) for metric in ['discovery_rank1_fraction','discovery_rank2_fraction','discovery_anchor_span_fraction','calibration_active_rank1_fraction']} for obj in ['topk','matryoshka']})
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
