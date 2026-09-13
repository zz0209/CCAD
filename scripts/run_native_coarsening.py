"""Develop anchored cross-seed groups on retained natural paired states."""
from __future__ import annotations
import argparse,json,os,platform,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8')
from run_r011s1_raw_hook_asset import ROOT,entry,aggregate,write_json as write
from ccad.artifacts import sha256,validate_run_directory


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,required=True)
    cfg=json.loads(ap.parse_args().config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False)
    started=datetime.now(timezone.utc).isoformat();timer=time.perf_counter();cpu=time.process_time()
    write(run/'config.resolved.json',cfg);codes=[]
    for rel in ['scripts/run_native_coarsening.py','src/ccad/native_coarsening.py',
                'scripts/evaluate_native_coarsening.py','scripts/refit_response_groups.py','scripts/finite_group_refit.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py']:
        p=ROOT/rel;q=run/'source_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
        codes.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json',dict(files=codes,aggregate_sha256=aggregate(codes),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='native.coarsening.v1',run_id=cfg['run_id'],
        run_parent=cfg['round_id'],purpose=cfg['purpose'],milestone='M4',evidence_level='exposed_natural_development',
        started_utc=started,project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),
        code_snapshot_hash=aggregate(codes),source_snapshot_required=True,audit_opened=False,
        candidate_family_frozen=True,mean_constants_source_split='retained independent mean split',
        threshold_source_split='source-only discovery activity; fixed regularization for development',
        statistics_unit='natural documents and shared source/target SAE identities',device='cuda:0',
        seeds=cfg['seeds'],resource_lease='gpu-0 resource_manager.run',
        resource_lease_reason='Small selected contribution Grams on CUDA; two-thread bounded CPU convex fits'))
    for fn in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/fn).touch()
    write(run/'status.json',dict(status='RUNNING',updated_utc=started))
    inputs=[];results=[];env={};error=None
    def checked(path):
        p=Path(path);p=p if p.is_absolute() else ROOT/p
        inputs.append(entry(p,'retained natural paired state / controlled checkpoint','actual input','internal; original source licenses unchanged'))
        return p
    def log(event,**kw):
        item=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),event=event,
                  elapsed_seconds=time.perf_counter()-timer,**kw)
        write(run/'progress.json',item)
        with (run/'stdout.log').open('a') as f:f.write(json.dumps(item)+'\n')
        print(json.dumps(item),flush=True)
    try:
        import numpy as np,torch
        sys.path.append(cfg['scipy_overlay_dir'])
        import scipy
        from ccad.native_coarsening import contribution_gram,fit_groups
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
        reference=ROOT/cfg['reference_run'];rc=json.loads(checked(reference/'config.resolved.json').read_text())
        assert json.loads(checked(reference/'status.json').read_text())['status']=='PASS'
        tr=ROOT/rc['training_run'];snaps=json.loads(checked(tr/'checkpoints.json').read_text())['checkpoints']
        env=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,
                 numpy=np.__version__,scipy=scipy.__version__,gpu=torch.cuda.get_device_name(),threads=2)
        # A known split/merge problem verifies the actual constrained objective.
        a=np.diag([.5,.5]);b=np.array([[.5],[.5]]);d=np.array([[1.]])
        ss,tt,test=fit_groups(a,b,d,0,penalty=.001)
        fixed=np.array([1.,0.]);_,one,one_test=fit_groups(a,b,d,0,penalty=.001,fixed_source=fixed)
        assert ss[1]>.99 and tt[0]>.99 and test['absolute_residual']<1e-5
        assert one_test['absolute_residual']>.24
        write(run/'split_merge_check.json',dict(joint=test,fixed_atom=one_test,source=ss.tolist(),target=tt.tolist()))
        def states(split,objective,seed):
            with np.load(checked(reference/f'natural_{split}_{objective}_seed{seed}.npz')) as ar:
                z=torch.zeros(tuple(ar['shape']),device='cuda:0')
                z[torch.as_tensor(ar['rows'].astype('int64'),device='cuda:0'),torch.as_tensor(ar['columns'].astype('int64'),device='cuda:0')]=torch.as_tensor(ar['values'],device='cuda:0')
            return z
        def gram(zs,ds,zt,dt):return contribution_gram(zs,ds,zt,dt)
        for objective in cfg['objectives']:
            Z={};D={};means={}
            for seed in cfg['seeds']:
                snap=next(s for s in snaps if s['seed']==seed and s['objective']==objective and s['step']==rc['checkpoint_step'])
                p=checked(snap['path']);assert sha256(p)==snap['sha256']
                state=torch.load(p,map_location='cuda:0',weights_only=True)
                D[seed]=(state['decoder.weight'].T if objective=='topk' else state['W_dec']).detach().contiguous()
                Z[seed]={split:states(split,objective,seed) for split in ['mean','discovery','calibration']}
                means[seed]=Z[seed]['mean'].mean(0)
            for source,target in cfg['seed_pairs']:
                zs=Z[source]['discovery'];zt=Z[target]['discovery'];ds=D[source];dt=D[target]
                source_energy=zs.square().mean(0)*ds.square().sum(1)
                target_energy=zt.square().mean(0)*dt.square().sum(1)
                active=(zs>0).sum(0)
                eligible=torch.where((active>=cfg['minimum_source_activations'])&(active<=cfg['maximum_source_fraction']*len(zs)))[0]
                ordered=eligible[torch.argsort(source_energy[eligible],descending=True,stable=True)]
                positions=torch.linspace(0,len(ordered)-1,cfg['anchors_per_pair']+2,device='cuda:0').long()[1:-1]
                anchors=ordered[positions].cpu().tolist()
                log('SOURCE_ANCHORS',objective=objective,source=source,target=target,eligible=len(eligible),anchors=anchors)
                for anchor in anchors:
                    # Discovery-only native contribution affinity chooses a local pool.
                    ka=gram(zs[:,anchor:anchor+1],ds[anchor:anchor+1],zs,ds)[0]
                    sa=ka.abs()/source_energy.sqrt().clamp_min(1e-8)
                    sp=torch.argsort(sa,descending=True,stable=True)[:cfg['source_pool']]
                    if not bool((sp==anchor).any()):sp[-1]=anchor
                    cross=gram(zs[:,sp],ds[sp],zt,dt)
                    affinity=cross.abs()/source_energy[sp].sqrt().clamp_min(1e-8)[:,None]/target_energy.sqrt().clamp_min(1e-8)[None,:]
                    tp=torch.argsort(affinity.max(0).values,descending=True,stable=True)[:cfg['target_pool']]
                    Kss=gram(zs[:,sp],ds[sp],zs[:,sp],ds[sp]).double().cpu().numpy()
                    Kst=cross[:,tp].double().cpu().numpy()
                    Ktt=gram(zt[:,tp],dt[tp],zt[:,tp],dt[tp]).double().cpu().numpy()
                    aidx=int(torch.where(sp==anchor)[0][0]);s,t,fit=fit_groups(Kss,Kst,Ktt,aidx,penalty=cfg['penalty'],maxiter=cfg['maxiter'])
                    atom=np.zeros(len(sp));atom[aidx]=1.
                    _,atomt,atomfit=fit_groups(Kss,Kst,Ktt,aidx,penalty=cfg['penalty'],maxiter=cfg['maxiter'],fixed_source=atom)
                    masks={'joint_group':(s,t),'anchored_atom':(atom,atomt)}
                    # For the exact same learned source group, an optimally scaled
                    # single target atom is a stronger comparison than cosine alone.
                    cross_group=s@Kst;gain=np.clip(cross_group/np.diag(Ktt).clip(1e-12),0,1)
                    errors=s@Kss@s-2*gain*cross_group+gain*gain*np.diag(Ktt);best=int(errors.argmin())
                    onehot=np.zeros(len(tp));onehot[best]=gain[best];masks['same_group_best_atom']=(s,onehot)
                    metrics={}
                    for name,(sm,tm) in masks.items():
                        smt=torch.as_tensor(sm,dtype=zs.dtype,device='cuda:0');tmt=torch.as_tensor(tm,dtype=zs.dtype,device='cuda:0')
                        metrics[name]={}
                        for split in ['discovery','calibration']:
                            xs=Z[source][split][:,sp];xt=Z[target][split][:,tp]
                            ys=(xs*smt)@ds[sp];yt=(xt*tmt)@dt[tp]
                            mean_s=(means[source][sp]*smt)@ds[sp];mean_t=(means[target][tp]*tmt)@dt[tp]
                            error_actual=(ys-yt).square().sum(1).mean();energy_actual=ys.square().sum(1).mean()
                            phis=ys-mean_s;phit=yt-mean_t
                            metrics[name][split]=dict(actual_relative_error=float(error_actual/energy_actual.clamp_min(1e-12)),
                                actual_error=float(error_actual),source_energy=float(energy_actual),
                                phi_relative_error=float((phis-phit).square().sum(1).mean()/phis.square().sum(1).mean().clamp_min(1e-12)),
                                mean_difference_squared=float((mean_s-mean_t).square().sum()),
                                target_energy=float(yt.square().sum(1).mean()))
                    key=f'{objective}_s{source}_t{target}_a{anchor}'
                    np.savez_compressed(run/(key+'_groups.npz'),source_members=sp.cpu().numpy(),target_members=tp.cpu().numpy(),
                        source_gate=s,target_gate=t,atom_target_gate=atomt,best_atom_target_gate=onehot,anchor=np.array(anchor),
                        source_mean=means[source][sp].cpu().numpy(),target_mean=means[target][tp].cpu().numpy())
                    row=dict(query=key,objective=objective,source_seed=source,target_seed=target,anchor=anchor,
                        fit=fit,atom_fit=atomfit,metrics=metrics,source_active_count=int(active[anchor]))
                    results.append(row)
                    with (run/'metrics.raw.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    write(run/'query_results.json',dict(queries=results,scope=cfg['scope']))
                    log('QUERY_COMPLETE',query=key,source_members=fit['source_members'],target_members=fit['target_members'],
                        calibrated_errors={name:met['calibration']['actual_relative_error'] for name,met in metrics.items()})
                    if time.perf_counter()-timer>cfg['budget_seconds']:raise TimeoutError('Bounded coarsening development budget exceeded')
        if cfg.get('response'):
            from refit_response_groups import refit
            env['response']=refit(cfg,run,reference,rc,results,checked,write,log)
        if cfg.get('finite_refit'):
            from finite_group_refit import finite_refit
            env['finite_refit']=finite_refit(cfg,run,reference,rc,results,checked,write,log)
        if cfg.get('functional'):
            from evaluate_native_coarsening import evaluate
            functional=evaluate(cfg,run,reference,rc,results,checked,write,log)
            env.update(functional= functional)
    except BaseException as exc:
        error=repr(exc);(run/'stderr.log').write_text(traceback.format_exc());log('FAIL',error=error)
    status='PASS' if error is None else 'FAIL'
    summary=dict(status=status,error=error,completed_queries=len(results),wall_seconds=time.perf_counter()-timer,
        process_cpu_seconds=time.process_time()-cpu,metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),
        scope=cfg['scope'],generator_script_path=codes[0]['path'],generator_script_sha256=codes[0]['sha256'])
    write(run/'inputs.json',dict(inputs=inputs));write(run/'environment.json',env);write(run/'metrics.summary.json',summary)
    write(run/'status.json',dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),error=error))
    check=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=check.ok,errors=check.errors))
    log('COMPLETE',summary=summary,contract_ok=check.ok,errors=check.errors)
    return 0 if status=='PASS' and check.ok else 1


if __name__=='__main__':raise SystemExit(main())
