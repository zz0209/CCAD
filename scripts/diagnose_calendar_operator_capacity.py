"""Cached-data operator capacity versus available SAE code information."""
import argparse,json,os,sys,time,platform,traceback
from pathlib import Path
from datetime import datetime,timezone
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from run_r011s1_raw_hook_asset import ROOT,write_json as write,entry,aggregate
from ccad.artifacts import sha256,validate_run_directory
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    cfg=json.loads(ap.parse_args().config.read_text())
    run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False)
    start=time.perf_counter();write(run/'config.resolved.json',cfg)
    files=[]
    for rel in ['scripts/diagnose_calendar_operator_capacity.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py']:
        p=ROOT/rel;dst=run/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
        files.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json',dict(files=files,aggregate_sha256=aggregate(files),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='conditional.operator.capacity.v1',run_id=cfg['run_id'],
        run_parent=cfg['parent_run'],purpose=cfg['purpose'],milestone='conditional-variable-method-development',
        evidence_level='adaptive_cached_development',started_utc=datetime.now(timezone.utc).isoformat(),
        project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(files),
        source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,
        mean_constants_source_split='fixed donor differences; original means cancel',
        threshold_source_split='fit rows only; test exposed development,not confirmation',
        statistics_unit='shared templates,values andfiveSAEs',device='cuda:0',seeds=[1,2,3,4,5],
        resource_lease='gpu-0 manager plus1auxCPUthread',resource_lease_reason=cfg['budget']))
    for f in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/f).touch()
    write(run/'status.json',dict(status='RUNNING'))
    inputs=[];metrics=[];checks={};env={};error=None
    def read(name):
        p=ROOT/'runs'/cfg['parent_run']/name
        inputs.append(entry(p,'CCAD completed conditional composition data','parent input'))
        return p
    try:
        import torch
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
        t=lambda x:torch.tensor(np.asarray(x),dtype=torch.float64,device='cuda:0')
        host=lambda x:x.detach().cpu().numpy()
        assert json.loads(read('status.json').read_text())['status']=='PASS'
        pp=json.loads(read('prompts_and_pairs.json').read_text());rows=pp['rows'];pairs=pp['pairs']
        cs=json.loads(read('composition_sources.json').read_text())
        raw=np.load(read('raw_hooks.npz'))['layer5'];slots=np.array([r['slot'] for r in rows]);h=raw[np.arange(len(rows)),slots]
        ri=np.array([p['recipient'] for p in pairs]);di=np.array([p['donor'] for p in pairs]);dh=h[di]-h[ri]
        positive=np.array([rows[i]['role']=='calendar' for i in ri]);fit=np.array(cs['fit_pairs']);test=np.array(cs['test_pairs'])
        teacher=np.where(positive[:,None],dh,0.);tt=t(teacher[fit]);allpred={};replays=[]
        for seed in range(1,6):
            ar=np.load(read(f'codes_seed{seed}.npz'));z=ar['codes'][np.arange(len(rows)),slots].astype(float);dec=ar['decoder'].astype(float)
            dz=z[di]-z[ri];x=t(dz[fit]);dd=t(dec);n=len(fit);ridge=cfg['ridge_fraction']
            diag=torch.mean(x*x,dim=0)*torch.sum(dd*dd,dim=1);active=torch.where(diag>1e-12)[0]
            rhs=torch.sum((x.T@tt/n)*dd,dim=1);score=rhs*rhs/torch.clamp(diag,min=1e-30)
            rank=active[torch.argsort(score[active],descending=True,stable=True)]
            pred={};saved={};meta={}
            for name,ids in [(f'native{k}',rank[:k]) for k in cfg['supports']]+[('native_full_ridge',active)]:
                a=x[:,ids];d=dd[ids];scale=torch.sqrt(diag[ids]);g=(a.T@a/n)*(d@d.T)
                normalized=g/scale[:,None]/scale[None,:]+ridge*torch.eye(len(ids),dtype=torch.float64,device='cuda:0')
                right=rhs[ids]/scale;v=torch.linalg.solve(normalized,right);beta=v/scale
                residual=float(torch.linalg.norm(normalized@v-right)/torch.clamp(torch.linalg.norm(right),min=1e-30));assert residual<1e-8
                ii,bb=host(ids),host(beta);pred[name]=(dz[:,ii]*bb)@dec[ii]
                saved[name+'_ids']=ii;saved[name+'_coef']=bb
                meta[name]=dict(support=len(ids),negative_coefficients=int((beta<0).sum()),max_abs_coefficient=float(torch.abs(beta).max()),normal_residual=residual)
                if name=='native64':
                    old=next(r for r in cs['sources'] if r['source_seed']==seed)
                    replays.append(dict(seed=seed,support_equal=ii.tolist()==old['source_ids'],coefficient_max_error=float(np.max(np.abs(bb-old['source_coefficients'])))))
            for name,inputs_all in [('code_readout',dz),('raw_linear',dh)]:
                xx=t(inputs_all[fit]);scale=torch.sqrt(torch.mean(xx*xx,dim=0));use=torch.where(scale>1e-12)[0]
                xn=xx[:,use]/scale[use];u,s,v=torch.linalg.svd(xn,full_matrices=False)
                beta=(v.T*(s/(s*s+n*ridge)))@(u.T@tt)/scale[use,None]
                ii,bb=host(use),host(beta);pred[name]=inputs_all[:,ii]@bb
                saved[name+'_ids']=ii;saved[name+'_beta']=bb
                meta[name]=dict(input_columns=len(ii),rank_at_1e_8=int((s>s[0]*1e-8).sum()),output_class='unrestricted learned hook directions')
            for phase,ix in [('fit',fit),('test_exposed',test)]:
                for role,mask in [('calendar',positive),('noncalendar',~positive)]:
                    jj=ix[mask[ix]];energy=float(np.sum(dh[jj]**2))
                    for name,y in pred.items():
                        record=dict(seed=seed,phase=phase,role=role,method=name,pairs=len(jj),
                            relative_conditional_error=float(np.sum((y[jj]-teacher[jj])**2)/energy),
                            raw_variation_energy=energy,prediction_energy=float(np.sum(y[jj]**2)),**meta[name])
                        metrics.append(record)
                        with (run/'metrics.raw.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            np.savez_compressed(run/f'operators_seed{seed}.npz',**saved,**{name+'_prediction':y for name,y in pred.items()})
            elapsed=time.perf_counter()-start;progress=dict(seed=seed,seconds=elapsed,rows=len(metrics));write(run/'progress.json',progress);print(json.dumps(progress),flush=True)
            if elapsed>cfg['budget_seconds']:raise TimeoutError('Capacity diagnostic budget exceeded')
        write(run/'parent_replay.json',dict(rows=replays))
        checks=dict(parent64_replay=all(r['support_equal'] and r['coefficient_max_error']<1e-9 for r in replays),
                    all_rows=len(metrics)==5*2*2*5,finite=all(np.isfinite(r['relative_conditional_error']) for r in metrics),zero_new_LM=True)
        env=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
            gpu=torch.cuda.get_device_name(),cpu_threads=1,peak_vram_bytes=torch.cuda.max_memory_allocated(),dtype='float64 cached fits')
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';(run/'stderr.log').write_text(traceback.format_exc())
    status='PASS' if error is None and checks and all(checks.values()) else 'FAIL'
    summary=dict(status=status,error=error,checks=checks,rows=len(metrics),wall_seconds=time.perf_counter()-start,lm_sequences=0,
        scope=cfg['scope'],metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path='scripts/diagnose_calendar_operator_capacity.py',
        generator_script_sha256=sha256(run/'source_snapshot/scripts/diagnose_calendar_operator_capacity.py'))
    write(run/'inputs.json',dict(inputs=inputs));write(run/'environment.json',env);write(run/'metrics.summary.json',summary);write(run/'stdout.log',summary)
    write(run/'status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat()))
    v=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=v.ok,errors=list(v.errors)))
    print(json.dumps(dict(summary=summary,contract_ok=v.ok,errors=list(v.errors))),flush=True)
    return 0 if status=='PASS' and v.ok else 1


if __name__=='__main__':raise SystemExit(main())
