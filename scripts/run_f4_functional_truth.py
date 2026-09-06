"""Known information loss versus executable split codes, with held-out nonlinear effects."""
from __future__ import annotations
import argparse
import itertools
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ccad.hook_transport import fit_hook_space_transport
from ccad.artifacts import sha256,validate_run_directory
from run_r011s1_raw_hook_asset import write_json as write,aggregate,entry
from run_f4_source_reference_causal import best_single_atom
from f4_probability_endpoints import log_prob

def sample(rng,n,context=None):
    x=(rng.random((n,5))<.25)*rng.uniform(.25,1,(n,5))
    x[:,3]=rng.integers(0,2,n) if context is None else context
    return x

def encode(x,family):
    u,v,w,c,n=x.T
    z=np.column_stack([np.maximum(u-c,0),np.maximum(u+c-1,0),v,w,c,n])
    if family=='source_function_deleted':z[:,:2]=0
    elif family=='context_alias':z[:,1]=np.maximum(n+c-1,0)
    assert np.all(z>=0)
    return z

def measure(cfg):
    rows=[];arrays={};details=[]
    ds=np.eye(5);dt=np.column_stack([ds[:,0],ds])
    sim=np.abs(ds.T@dt)
    assignment=max(itertools.permutations(range(6),5),key=lambda p:sum(sim[i,j] for i,j in enumerate(p)))
    mcc=sum(sim[i,j] for i,j in enumerate(assignment))/5
    span=float(np.linalg.norm(ds.T@np.linalg.qr(dt.T.T)[0],ord='fro')**2/5)
    assert mcc==1 and abs(span-1)<1e-12 and assignment[0]==0
    for seed in cfg['seeds']:
        rng=np.random.default_rng(seed)
        mean=sample(rng,cfg['mean_samples']);discovery=sample(rng,cfg['discovery_samples'])
        recipients=np.concatenate([sample(rng,cfg['pairs_per_context'],c) for c in [0,1]])
        donors=np.concatenate([sample(rng,cfg['pairs_per_context'],c) for c in [0,1]])
        w1=rng.normal(size=(5,9))*.6;w2=rng.normal(size=(9,7))*.6
        def downstream(x):return log_prob(np.tanh(x@w1)@w2)
        source_coord=cfg['intervention_scale']*(donors[:,0]-recipients[:,0])
        source_delta=np.zeros_like(recipients);source_delta[:,0]=source_coord
        base=downstream(recipients);ref=downstream(recipients+source_delta);prob=np.exp(ref)
        baseline_kl=np.sum(prob*(ref-base),axis=1)
        prefix=f'seed{seed}_';arrays.update({prefix+'mean':mean,prefix+'discovery':discovery,
            prefix+'recipients':recipients,prefix+'donors':donors,prefix+'w1':w1,prefix+'w2':w2,
            prefix+'source_coordinate':source_coord,prefix+'source_log_prob':ref,prefix+'base_log_prob':base})
        for family in cfg['families']:
            zm=encode(mean,family);zd=encode(discovery,family);zr=encode(recipients,family);zt=encode(donors,family)
            hm=zm@dt.T;hd=zd@dt.T
            source=np.zeros_like(discovery);source[:,0]=discovery[:,0]-mean[:,0].mean()
            fit=fit_hook_space_transport(hd-hm.mean(0),source,np.ones(len(hd)),rank=1,ridge_fraction=cfg['ridge_fraction'])
            single=best_single_atom(zd-zm.mean(0),source[:,0],np.ones(len(zd))/len(zd),cfg['ridge_fraction'],True)
            differences=cfg['intervention_scale']*(zt-zr)
            candidates={
                'fcc_hook_readout':fit.predict(differences@dt.T)[:,0],
                'best_dynamic_single_atom':differences[:,single['atom']]*single['coefficient'],
                'global_matching_operation':differences[:,assignment[0]],
                'raw_source_hook_oracle':source_coord.copy()}
            details.append(dict(seed=seed,family=family,fit_status=fit.status,mcc=mcc,span=span,assignment=list(assignment),
                best_atom=single,nonnegative=True,max_active_codes=int(np.max(np.sum(zd>0,axis=1))),
                reconstruction_squared_error=float(np.sum((hd-discovery)**2)/np.sum(discovery**2)),
                target_factors=fit.target_factors.tolist(),source_factors=fit.source_factors.tolist()))
            assert family!='context_split_preserved' or np.allclose(hd,discovery,rtol=0,atol=1e-15)
            for method,coord in candidates.items():
                delta=np.zeros_like(recipients);delta[:,0]=coord
                pred=downstream(recipients+delta);kl=np.sum(prob*(ref-pred),axis=1)
                assert np.min(kl)>-1e-12
                arrays[prefix+family+'_'+method+'_coordinate']=coord
                arrays[prefix+family+'_'+method+'_kl']=kl
                for scope in ['all','context0','context1']:
                    mask=np.ones(len(coord),dtype=bool) if scope=='all' else recipients[:,3]==int(scope[-1])
                    den=float(np.sum(source_coord[mask]**2)); kden=float(np.sum(baseline_kl[mask]));assert den>0 and kden>0
                    rows.append(dict(seed=seed,family=family,method=method,scope=scope,pairs=int(mask.sum()),
                        dictionary_mcc=mcc,decoder_span_score=span,
                        normalized_coordinate_error=float(np.sum((coord[mask]-source_coord[mask])**2)/den),
                        normalized_kl_error=float(np.sum(kl[mask])/kden),source_coordinate_energy=den,source_kl_sum=kden))
    return rows,arrays,details

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args()
    cfg=json.loads(args.config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False)
    start=time.perf_counter();write(run/'config.resolved.json',cfg);code=[]
    for rel in ['scripts/run_f4_functional_truth.py','src/ccad/hook_transport.py','src/ccad/artifacts.py',
                'scripts/run_r011s1_raw_hook_asset.py','scripts/run_f4_source_reference_causal.py','scripts/f4_probability_endpoints.py']:
        p=ROOT/rel;dst=run/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
        code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
    write(run/'inputs.json',dict(inputs=[entry(args.config,'prospective synthetic config','config')]))
    write(run/'manifest.json',dict(schema_version='fcc.functional.truth.v1',run_id=run.name,run_parent='F4',purpose=cfg['purpose'],milestone='M4',
        evidence_level='handcrafted_nonnegative_encoder_synthetic_truth',started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),
        config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(code),source_snapshot_required=True,audit_opened=False,
        candidate_family_frozen=True,mean_constants_source_split='independent generated mean',threshold_source_split='none',
        statistics_unit='generated seed; not real SAE seed',device='CPU',seeds=cfg['seeds'],resource_lease='none',resource_lease_reason=cfg['budget']))
    write(run/'status.json',dict(status='RUNNING'));rows=[];error=None
    try:
        rows,arrays,details=measure(cfg);np.savez_compressed(run/'synthetic_arrays.npz',**arrays);write(run/'fit_details.json',details)
    except Exception:error=traceback.format_exc()
    (run/'metrics.raw.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    summary=dict(model_forwards=0,synthetic_consumer='fixed generated tanh/softmax',rows=len(rows),wall_seconds=time.perf_counter()-start,
                 metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),limitations=cfg['limits'],aggregates=[])
    for family in cfg['families']:
        for method in cfg['methods']:
            for scope in ['all','context0','context1']:
                group=[r for r in rows if r['family']==family and r['method']==method and r['scope']==scope]
                if group:summary['aggregates'].append(dict(family=family,method=method,scope=scope,seeds=len(group),
                    **{metric:dict(mean=float(np.mean([r[metric] for r in group])),sd=float(np.std([r[metric] for r in group],ddof=1)),
                                    minimum=min(r[metric] for r in group),maximum=max(r[metric] for r in group))
                       for metric in ['normalized_coordinate_error','normalized_kl_error']}))
    write(run/'metrics.summary.json',summary)
    write(run/'environment.json',dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
          cuda='not_applicable',gpu='not_applicable',pytorch='not_imported',transformers='not_imported',sae='handcrafted, not trained'))
    (run/'stderr.log').write_text(error or '');(run/'stdout.log').write_text(json.dumps(summary)+'\n')
    write(run/'status.json',dict(status='FAIL' if error else 'PASS',error=error,updated_utc=datetime.now(timezone.utc).isoformat()))
    valid=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=valid.ok,errors=list(valid.errors)))
    print(json.dumps(dict(run=run.name,error=error,contract_ok=valid.ok,rows=len(rows),wall_seconds=summary['wall_seconds'])))
    return int(bool(error) or not valid.ok)

if __name__=='__main__':raise SystemExit(main())
