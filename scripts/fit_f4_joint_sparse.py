"""Li15-inspired standardized joint-support fitting, without task endpoints.

Uses sklearn MultiTaskLasso (BSD-3-Clause), not a bespoke optimizer. The group
penalty is an adaptation to two FCC components, not an exact Li15 reproduction.
"""
import argparse
import importlib.metadata as metadata
import json
import platform
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

from run_f4_source_reference_causal import ROOT, np, write, sha256
from run_r011s1_raw_hook_asset import entry, aggregate
from ccad.artifacts import validate_run_directory
from sklearn.linear_model import MultiTaskLasso, Lasso
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits


def standardized_inputs(z, y, weights):
    z=np.asarray(z,dtype=float); y=np.asarray(y,dtype=float); w=np.asarray(weights,dtype=float)
    if y.shape!=(len(z),2) or w.shape!=(len(z),) or np.any(w<0) or not w.sum()>0:
        raise ValueError('Invalid paired weighted two-output input')
    w=w/w.sum(); mx=w@z; my=w@y; xc=z-mx; yc=y-my
    sx=np.sqrt(w@(xc*xc)); sy=np.sqrt(w@(yc*yc)); active=(sx>0)&(np.ptp(z[w>0],axis=0)>0)
    # Exact constant columns can acquire a tiny residual from weighted-mean
    # rounding; never amplify this into a unit-variance input/output.
    if not np.all((sy>0)&(np.ptp(y[w>0],axis=0)>0)) or not np.any(active):raise ValueError('No conditional input/output variation')
    x=xc[:,active]/sx[active]; target=yc/sy
    scale=np.sqrt(len(z)*w)[:,None]
    return np.asfortranarray(x*scale),np.asfortranarray(target*scale),w,mx,my,sx,sy,active


def fit_joint(z, y, weights, cfg):
    started=time.perf_counter(); x,yy,w,mx,my,sx,sy,active=standardized_inputs(z,y,weights)
    n=len(x); alpha_max=float(np.max(np.linalg.norm(x.T@yy/n,axis=1)))
    factors=[cfg['pilot_alpha_fraction']] if cfg['pilot'] else np.geomspace(1.,cfg['minimum_alpha_fraction'],cfg['alpha_count'])
    model=MultiTaskLasso(fit_intercept=False,warm_start=True,selection='cyclic',tol=cfg['tol'],max_iter=cfg['max_iter'])
    path=[]; best=None; stop_reason='path_complete'
    for step,factor in enumerate(factors):
        model.set_params(alpha=max(alpha_max*float(factor),np.finfo(float).tiny))
        t=time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always',ConvergenceWarning);model.fit(x,yy)
        elapsed=time.perf_counter()-t; coef=model.coef_.T.copy(); support=np.flatnonzero(np.linalg.norm(coef,axis=1)>0)
        convergence_warnings=[str(r.message) for r in captured if issubclass(r.category,ConvergenceWarning)]
        # eps_ is the unnormalized kernel tolerance; dual_gap_ has been /n.
        converged=not convergence_warnings and float(model.dual_gap_)<=float(model.eps_)/n*1.01+1e-12
        row=dict(step=step,alpha_fraction=float(factor),alpha=model.alpha,support=len(support),iterations=int(model.n_iter_),
            dual_gap=float(model.dual_gap_),kernel_eps=float(model.eps_),converged=converged,seconds=elapsed,warnings=convergence_warnings)
        beta=np.zeros((z.shape[1],2));beta[active]=coef*sy/sx[active,None]
        if cfg['pilot']:
            best=(0.,beta,row.copy())
        elif converged and 0<len(support)<=cfg['support_budget']:
            # Relax shrinkage on the selected common support, using the same
            # normalized design and a fixed weak ridge. No task labels/metrics.
            xx=x[:,support]; gram=xx.T@xx/n
            lam=cfg['debias_ridge_fraction']*float(np.trace(gram))/len(support)
            refit=np.linalg.solve(gram+lam*np.eye(len(support)),xx.T@yy/n)
            loss=float(np.sum((yy-xx@refit)**2)/n)
            beta[:]=0.; ids=np.flatnonzero(active)[support]; beta[ids]=refit*sy/sx[ids,None]
            row.update(debiased_standardized_error=loss,ridge_lambda=lam,atom_ids=ids.tolist())
            if best is None or loss<best[0]:best=(loss,beta.copy(),row.copy())
        path.append(row)
        if time.perf_counter()-started>cfg['per_fit_budget_seconds']:raise TimeoutError('Per-fit numerical budget exceeded')
        if not cfg['pilot'] and len(support)>cfg['path_stop_support']:
            stop_reason='first_path_support_exceeds_prespecified_horizon';break
    if best is None:raise RuntimeError('No converged nonempty <=budget candidate; not a scientific negative')
    _,beta,selected=best; intercept=my-mx@beta
    residual=(np.asarray(y)-my)-(np.asarray(z)-mx)@beta
    return beta,intercept,dict(path=path,selected=selected,alpha_max=alpha_max,active_inputs=int(active.sum()),
        output_std=sy.tolist(),input_weighted_mean=mx.tolist(),output_weighted_mean=my.tolist(),
        standardized_training_error=float(np.sum(w[:,None]*(residual/sy)**2)),
        unstandardized_training_error=float(np.sum(w[:,None]*residual**2)),
        fit_seconds=time.perf_counter()-started,stop_reason=stop_reason)


def select_separate_supports(z, y, weights, cfg):
    """Scalar L1 paths on the same standardized design, <=8 per output.

    Each output uses its own alpha maximum. Path candidate scoring uses the
    same standardized ridge objective as fit_joint, but never behavioral data.
    The caller refits both the shared and union supports with one common kernel.
    """
    started=time.perf_counter()
    x,yy,w,mx,my,sx,sy,active=standardized_inputs(z,y,weights)
    n=len(x); active_ids=np.flatnonzero(active); chosen=[]; diagnostics=[]
    for j in range(2):
        target=yy[:,j]; alpha_max=float(np.max(np.abs(x.T@target/n)))
        model=Lasso(fit_intercept=False,warm_start=True,selection='cyclic',tol=cfg['tol'],max_iter=cfg['max_iter'])
        path=[];best=None
        for step,factor in enumerate(np.geomspace(1.,cfg['minimum_alpha_fraction'],cfg['alpha_count'])):
            model.set_params(alpha=max(alpha_max*float(factor),np.finfo(float).tiny))
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always',ConvergenceWarning);model.fit(x,target)
            support=np.flatnonzero(model.coef_!=0)
            ws=[str(r.message) for r in captured if issubclass(r.category,ConvergenceWarning)]
            gap_limit=cfg['tol']*float(np.mean(target**2))
            converged=not ws and float(model.dual_gap_)<=gap_limit*1.01+1e-12
            row=dict(step=step,alpha_fraction=float(factor),alpha=model.alpha,support=len(support),converged=converged,dual_gap=float(model.dual_gap_),gap_limit=gap_limit,iterations=int(model.n_iter_),warnings=ws)
            if converged and 0<len(support)<=cfg['per_output_budget']:
                xx=x[:,support];gram=xx.T@xx/n
                lam=cfg['debias_ridge_fraction']*float(np.trace(gram))/len(support)
                b=np.linalg.solve(gram+lam*np.eye(len(support)),xx.T@target/n)
                loss=float(np.mean((target-xx@b)**2))
                row.update(debiased_standardized_error=loss,atom_ids=active_ids[support].tolist())
                if best is None or loss<best[0]:best=(loss,active_ids[support].copy(),row.copy())
            path.append(row)
            if time.perf_counter()-started>cfg['per_fit_budget_seconds']:raise TimeoutError('Separate support fit budget exceeded')
            if len(support)>cfg['path_stop_support']:break
        if best is None:raise RuntimeError('No converged scalar support within budget')
        chosen.append(best[1]);diagnostics.append(dict(output=j,alpha_max=alpha_max,selected=best[2],path=path))
    union=np.unique(np.concatenate(chosen));assert len(union)<=2*cfg['per_output_budget']
    return chosen,dict(outputs=diagnostics,union=union.tolist(),per_output_supports=[x.tolist() for x in chosen],fit_seconds=time.perf_counter()-started,selection='Two scalar L1 paths, per-output standardized ridge training error; no endpoints')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False)
    started=time.perf_counter();now=datetime.now(timezone.utc).isoformat();write(run/'config.resolved.json',cfg)
    code=[]
    for rel in ('scripts/fit_f4_joint_sparse.py','scripts/run_f4_source_reference_causal.py','scripts/run_r011s1_raw_hook_asset.py',
                'src/ccad/artifacts.py','src/ccad/hook_transport.py','src/ccad/activation_contract.py'):
        p=ROOT/rel;dest=run/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path=f'source_snapshot/{rel}'))
    write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='fcc.joint.sparse.v1',run_id=cfg['run_id'],run_parent='F4',purpose=cfg['purpose'],
        milestone='M4',evidence_level='discovery_only_solver_pilot' if cfg['pilot'] else 'discovery_only_sparse_execution_fit',
        started_utc=now,project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(code),source_snapshot_required=True,
        audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='original independent mean; conditional intercept separately cancels in donor',
        threshold_source_split='fixed source discovery rows and weights; no task endpoints',statistics_unit='query/shared-seed dependencies',device='cpu',seeds=[1,2,3,4,5],
        resource_lease='cpu-heavy resource_manager.run',resource_lease_reason=cfg['budget']))
    write(run/'status.json',dict(status='RUNNING',updated_utc=now))
    for name in ('stdout.log','stderr.log','metrics.raw.jsonl'):(run/name).touch()
    inputs=[];seen={};records=[];arrays={};checks={};error=None;fit_seconds=0.
    def checked(path,expected=None):
        p=Path(path);p=p if p.is_absolute() else ROOT/p
        if p not in seen:seen[p]=sha256(p);inputs.append(entry(p,'CCAD existing discovery assets or official BSD/MIT solver','sparse_fit_input'))
        if expected and seen[p]!=expected:raise ValueError('Input identity changed: '+str(p))
        return p
    def load(path,expected=None):return json.loads(checked(path,expected).read_text())
    def mmap(m):return np.memmap(checked(m['path'],m['sha256']),dtype='<u2' if m['dtype']=='uint16' else '<f4',mode='r',shape=tuple(m['shape']))
    try:
        checked(args.config);load('.aris/compute/local-f4-sparse-env-spec.json');checked('.aris/compute/f4_sparse_install_report.json')
        import sklearn.linear_model._coordinate_descent as sklearn_source
        checked(sklearn_source.__file__)
        parent=ROOT/cfg['parent_run'];pc=load(parent/'config.resolved.json',cfg['parent_config_sha256'])
        partitions=[r for r in load(parent/'component_fits.json',cfg['parent_fits_sha256']) if r['kind']=='source_partition']
        old=np.load(checked(parent/'component_coefficients.npz',cfg['parent_coefficients_sha256']),allow_pickle=False)
        manifests={};base=None;ref=None
        for label in ('old_fit','new_fit'):
            spec=pc[label];fc=load(ROOT/spec['path']/'config.resolved.json',spec['config_sha256'])
            if base is None:
                base=load(fc['source_asset_config']);ref=load(fc['reference_config_template'].format(panel='original'))
                short=load(Path(base['bulk_asset_dir'])/'asset_manifest.json',base['asset_manifest_sha256'])
            lc=next(r for r in fc['long_configurations'] if r['name']=='long128');ac=load(lc['asset_config'])
            long=load(Path(ac['bulk_output_dir'])/'asset_manifest.json',lc['asset_manifest_sha256'])
            for item in ac['saes']:manifests['long',item['seed']]=long
        for seed in range(1,6):manifests['short',seed]=short
        f=np.load(checked(ref['factors_path'],ref['factors_sha256']),allow_pickle=False)
        fi={(int(s),int(a),int(t)):i for i,(s,a,t) in enumerate(zip(f['source_seed'],f['source_atom'],f['target_seed']))}
        codes={};means={};dec={}
        def dense(name,seed,rows,need_mean=False):
            key=name,seed;man=manifests[key]
            if key not in codes:
                split=next(r for r in man['splits'] if r['split']=='discovery')
                codes[key]={m['dtype']:mmap(m) for m in split['files'] if m['seed']==seed}
            if need_mean and key not in means:
                split=next(r for r in man['splits'] if r['split']=='mean');mm={m['dtype']:mmap(m) for m in split['files'] if m['seed']==seed}
                means[key]=np.bincount(mm['uint16'].ravel(),weights=mm['float32'].ravel(),minlength=base['num_latents'])/len(mm['uint16'])
                dm=next(r for r in man['decoders'] if r['seed']==seed);dec[key]=np.asarray(mmap(dict(dm,dtype='float32')),dtype=float)
            z=np.zeros((len(rows),base['num_latents']));np.add.at(z,(np.arange(len(rows))[:,None],codes[key]['uint16'][rows]),codes[key]['float32'][rows]);return z
        jobs=[]
        for r in sorted(partitions,key=lambda r:(r['source_seed'],r['source_atom'])):
            s,a=r['source_seed'],r['source_atom'];rows=np.asarray(r['discovery_rows']);w=np.asarray(r['discovery_weights']);targets=[t for t in range(1,6) if t!=s]
            b=f['source_basis'][fi[s,a,targets[0]],:,0].astype(float);z=dense('short',s,rows,True)
            groups=[np.asarray(g) for g in r['groups']]
            y=np.column_stack([(z[:,g]-means['short',s][g])@(dec['short',s][g]@b) for g in groups])
            for t in targets:
                assert np.array_equal(b,f['source_basis'][fi[s,a,t],:,0])
                for name in ('short','long'):jobs.append((s,a,t,name,rows,w,y))
        if cfg['pilot']:jobs=jobs[:1]
        with threadpool_limits(limits=cfg['cpu_threads']):
            for s,a,t,name,rows,w,y in jobs:
                z=dense(name,t,rows);beta,intercept,diag=fit_joint(z,y,w,cfg);fit_seconds+=diag['fit_seconds']
                key=f'{name}_joint_sparse_{s}_{a}_{t}';arrays[key]=beta;arrays[key+'_intercept']=intercept
                ow=w/w.sum();yc=y-ow@y;oldbeta=old[f'{name}_shared16_{s}_{a}_{t}'];oldres=yc-(z-ow@z)@oldbeta;sy=np.sqrt(ow@(yc*yc))
                row=dict(source_seed=s,source_atom=a,target_seed=t,configuration=name,array_key=key,discovery_rows=rows.tolist(),discovery_weights=w.tolist(),
                    shared_support=np.flatnonzero(np.linalg.norm(beta,axis=1)>0).tolist(),
                    previous_energy16_standardized_training_error=float(np.sum(ow[:,None]*(oldres/sy)**2)),**diag)
                records.append(row)
                with (run/'metrics.raw.jsonl').open('a',encoding='utf-8') as sink:sink.write(json.dumps(row,sort_keys=True)+'\n')
                np.savez_compressed(run/'sparse_coefficients.npz',**arrays)
                print(json.dumps(dict(stage='FIT_COMPLETE',source=[s,a],target=t,configuration=name,support=len(row['shared_support']),seconds=diag['fit_seconds'])),flush=True)
                if fit_seconds>cfg['total_fit_budget_seconds']:raise TimeoutError('Total scientific fitting budget exceeded')
        checks=dict(expected_fits=len(records)==(1 if cfg['pilot'] else 24),no_task_endpoints=True,no_audit_materialized=True,
            selected_converged=all(r['selected']['converged'] for r in records),support_budget=cfg['pilot'] or all(0<len(r['shared_support'])<=cfg['support_budget'] for r in records))
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';(run/'stderr.log').write_text(traceback.format_exc())
    versions={name:metadata.version(name) for name in ('numpy','scikit-learn','scipy','joblib','threadpoolctl','narwhals')}
    write(run/'environment.json',dict(python=sys.executable,python_version=platform.python_version(),platform=platform.platform(),packages=versions,
        cuda='not_applicable',gpu='not_applicable',torch='not_applicable',transformers='not_applicable',sae_framework='precomputed assets only'))
    write(run/'inputs.json',dict(inputs=inputs));status='PASS' if not error and checks and all(checks.values()) else 'FAIL'
    summary=dict(status=status,error=error,checks=checks,fits=len(records),fit_seconds=fit_seconds,wall_seconds=time.perf_counter()-started,
        model_forwards=0,metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path='scripts/fit_f4_joint_sparse.py',generator_script_sha256=sha256(Path(__file__)),scope_limit=cfg['scope'])
    write(run/'metrics.summary.json',summary);write(run/'stdout.log',summary);write(run/'status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat()))
    val=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=val.ok,errors=list(val.errors)))
    print(json.dumps(dict(summary=summary,contract_ok=val.ok,contract_errors=list(val.errors))),flush=True)
    return 0 if status=='PASS' and val.ok else 1


if __name__=='__main__':raise SystemExit(main())
