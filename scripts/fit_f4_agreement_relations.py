"""Small discovery-only fit and compilation of task-directed SAE operations."""
from __future__ import annotations
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ccad.artifacts import sha256,validate_run_directory
from ccad.causal_metric_probe import select_document_balanced_states
from ccad.hook_transport import fit_basis_constrained_transport
from run_r011s1_raw_hook_asset import entry,aggregate,write_json as write
from run_f4_agreement_source import swap_indices,box_ridge


def native_fit(dz,dec,c,b,ids,ridge_fraction):
    """Ridge then box clipping for D diag(g) delta-z; not a box-QP optimum."""
    x=dz[:,ids];d=dec[ids];n=len(x)
    gram=(x.T@x/n)*(d@d.T)
    rhs=(x.T@c/n)*(d@b)
    ridge=ridge_fraction*float(np.trace(gram))/len(ids)
    unconstrained=np.linalg.solve(gram+max(ridge,1e-12)*np.eye(len(ids)),rhs)
    g=np.clip(unconstrained,-1.,1.)
    return g,dict(ridge=ridge,clipped=int(np.sum(g!=unconstrained)),max_unclipped=float(np.max(np.abs(unconstrained))))


def native_vector_fit(x,d,teacher,ids,ridge_fraction):
    a=x[:,ids];decoder=d[ids];n=len(x)
    gram=(a.T@a/n)*(decoder@decoder.T);rhs=np.sum((a.T@teacher/n)*decoder,axis=1)
    ridge=ridge_fraction*float(np.trace(gram))/len(ids)
    g,info=box_ridge(gram+max(ridge,1e-12)*np.eye(len(ids)),rhs)
    return g,dict(ridge=ridge,**info)


def neighborhood_pairs(candidate_codes,task_codes,donors,count):
    """Source-only cosine retrieval, unique endpoints, round-robin task pairs."""
    pairs=[(i,int(j)) for i,j in enumerate(donors) if i<j]
    if count%(2*len(pairs)) or count>len(candidate_codes):raise ValueError('Invalid retrieval budget')
    a=candidate_codes/np.maximum(np.linalg.norm(candidate_codes,axis=1,keepdims=True),1e-30)
    b=task_codes/np.maximum(np.linalg.norm(task_codes,axis=1,keepdims=True),1e-30)
    similarity=a@b.T;used=np.zeros(len(a),dtype=bool);selected=[];prototypes=[];scores=[]
    for _ in range(count//(2*len(pairs))):
        for pair in pairs:
            for prototype in pair:
                score=np.where(used,-np.inf,similarity[:,prototype]);j=int(np.argmax(score))
                used[j]=True;selected.append(j);prototypes.append(prototype);scores.append(float(score[j]))
    return np.array(selected),np.array(prototypes),np.array(scores)


def read_codes(meta,seed,selected,num_latents):
    files=[f for f in meta['files'] if f['seed']==seed]
    im=next(f for f in files if f['dtype']=='uint16');am=next(f for f in files if f['dtype']=='float32')
    ix=np.asarray(np.memmap(im['path'],dtype='<u2',mode='r',shape=tuple(im['shape']))[selected])
    act=np.asarray(np.memmap(am['path'],dtype='<f4',mode='r',shape=tuple(am['shape']))[selected])
    z=np.zeros((len(selected),num_latents));np.put_along_axis(z,ix,act,axis=1)
    return z,ix,act


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(parents=True,exist_ok=False)
    write(run/'config.resolved.json',cfg);now=datetime.now(timezone.utc).isoformat();started=time.perf_counter()
    paths=[Path(__file__),ROOT/'scripts/run_f4_agreement_source.py',ROOT/'scripts/run_r011s1_raw_hook_asset.py',
           ROOT/'src/ccad/artifacts.py',ROOT/'src/ccad/causal_metric_probe.py',ROOT/'src/ccad/hook_transport.py',ROOT/'src/ccad/activation_contract.py']
    code=[]
    for p in paths:
        rel=p.relative_to(ROOT).as_posix();snap=run/'source_snapshot'/rel;snap.parent.mkdir(parents=True,exist_ok=True);snap.write_bytes(p.read_bytes())
        code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path=f'source_snapshot/{rel}'))
    write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='fcc.task.relation.fit.v1',run_id=cfg['run_id'],run_parent='F4',
        purpose='Discovery-only task-basis transport and native signed-edit pilot',milestone='M4',evidence_level='development_fit_not_behavioral_result',
        started_utc=now,started_local=datetime.now().astimezone().isoformat(),project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),
        code_snapshot_hash=aggregate(code),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,
        mean_constants_source_split='independent original mean cancels in same-representation differences',threshold_source_split='fixed before target forwards',
        statistics_unit='document-blocked sampled discovery pairs; target seeds share source',device='cpu',seeds=[1,2,3,4,5],
        resource_lease='cpu-heavy resource_manager.run',resource_lease_reason='bounded sampled-row ridge fits; no full discovery matrix or GPU',
        git_head_at_run=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    write(run/'status.json',dict(status='RUNNING',updated_utc=now))
    for name in ('stdout.log','stderr.log','metrics.raw.jsonl'):(run/name).touch()
    status='FAIL';summary={};inputs=[]
    try:
        if cfg['audit_opened'] or cfg['split']!='discovery':raise ValueError('Discovery only')
        p={role:Path(v['path']) if Path(v['path']).is_absolute() else ROOT/v['path'] for role,v in cfg['inputs'].items()}
        for role,path in p.items():
            if sha256(path)!=cfg['inputs'][role]['sha256']:raise ValueError(f'Input changed: {role}')
            inputs.append(entry(path,'CCAD fixed input',role))
        write(run/'inputs.json',dict(inputs=inputs))
        asset=json.loads(p['asset_manifest'].read_text());meta=next(x for x in asset['splits'] if x['split']=='discovery')
        task_paired=cfg.get('selection_mode')=='task_paired'
        records=[] if task_paired else json.loads(p['sequences'].read_text())['sequences']
        neighborhood=cfg.get('selection_mode')=='source_neighborhood'
        if task_paired:
            states=json.loads(p['paired_rows'].read_text())['rows']
            if len(states)!=cfg['state_count'] or any(r['split']!='discovery' for r in states):raise ValueError('Paired discovery scope mismatch')
            if not all(a['template']==b['template'] and a['attractor_number']==b['attractor_number'] and a['subject_number']==0 and b['subject_number']==1 for a,b in zip(states[::2],states[1::2])):raise ValueError('Subject pairing mismatch')
        else:states=select_document_balanced_states(records,split='discovery',count=cfg.get('candidate_count',cfg['state_count']),token_positions=(31,63,95,127),salt=cfg['state_salt'])
        if neighborhood:
            candidates=np.array([r['sequence_index']*128+r['token_position'] for r in states])
            z,ix,act=read_codes(meta,cfg['source_seed'],candidates,cfg['num_latents'])
            tasks=json.loads(p['task_tokens'].read_text())['rows']
            with np.load(p['task_cache'],allow_pickle=False) as cache:source_task_codes=cache[f"codes_{cfg['source_seed']}"]
            chosen,prototypes,scores=neighborhood_pairs(z,source_task_codes,swap_indices(tasks,'subject'),cfg['state_count'])
            np.savez_compressed(run/'source_retrieval.npz',candidate_indices=candidates,indices=ix,acts=act,chosen=chosen,prototypes=prototypes,cosines=scores,source_task_codes=source_task_codes)
            write(run/'retrieval.json',dict(candidate_rows=states,selected_cosine_min=float(scores.min()),selected_cosine_mean=float(scores.mean()),selected_cosine_max=float(scores.max()),source_seed=cfg['source_seed'],target_task_codes_read=False,source_retrieval_sha256=sha256(run/'source_retrieval.npz')))
            states=[dict(states[j],task_prototype=int(t),source_cosine=float(s)) for j,t,s in zip(chosen,prototypes,scores)]
        selected=np.arange(len(states)) if task_paired else np.array([r['sequence_index']*128+r['token_position'] for r in states])
        pairing='adjacent singular/plural subject; fixed attractor; source-defined task-conditioned synthetic discovery' if task_paired else ('adjacent source-nearest endpoints for each task subject pair; corpus neighbors are not verified linguistic counterfactuals' if neighborhood else 'adjacent selected rows; global corpus differences, not task-matched counterfactuals')
        write(run/'selected_states.json',dict(rows=states,pairing=pairing))
        b=np.load(p['direction'],allow_pickle=False)['basis'];codes={};decoders={};snapshots={}
        paired=np.load(p['paired_cache'],allow_pickle=False) if task_paired else None
        for seed in cfg['seeds']:
            if task_paired:
                z=paired[f'codes_{seed}'].astype(np.float64)
                if z.shape!=(len(states),cfg['num_latents']):raise ValueError('Paired code shape mismatch')
                snapshots[f'codes_{seed}']=z
            else:z,ix,act=read_codes(meta,seed,selected,cfg['num_latents'])
            codes[seed]=z;dm=next(f for f in asset['decoders'] if f['seed']==seed)
            if sha256(Path(dm['path']))!=dm['sha256']:raise ValueError('Decoder identity changed')
            decoders[seed]=np.asarray(np.memmap(dm['path'],dtype='<f4',mode='r',shape=tuple(dm['shape'])),dtype=np.float64)
            if not task_paired:snapshots[f'indices_{seed}']=ix;snapshots[f'acts_{seed}']=act
        if task_paired:h=paired['hidden'].astype(np.float64);paired.close()
        else:
            rm=next(x for x in json.loads(p['raw_manifest'].read_text())['splits'] if x['split']=='discovery')
            h=np.asarray(np.memmap(rm['path'],dtype='<f4',mode='r',shape=tuple(rm['shape']))[selected],dtype=np.float64)
        snapshots['raw_hook']=h;snapshots['selected_indices']=selected
        np.savez_compressed(run/'selected_discovery_data.npz',**snapshots)
        write(run/'read_scope.json',dict(materialized_splits=['discovery'],rows=len(selected),
            selection_mode=cfg.get('selection_mode','global'),task_conditioned_synthetic=task_paired,
            source_retrieval_rows=cfg.get('candidate_count',0),source_task_codes_used_for_selection=neighborhood,target_task_codes_used_for_fit=False,
            whole_bulk_file_hashes_recomputed=False,identity='Parent manifests bound; decoder full hashes verified; actually read selected rows retained in NPZ with run hash',
            selected_data_sha256=sha256(run/'selected_discovery_data.npz'),mean_materialized=False,calibration_materialized=False,audit_materialized=False))
        dz={s:z[::2]-z[1::2] for s,z in codes.items()};dh=h[::2]-h[1::2]
        native_teacher=cfg.get('teacher_mode')=='source_native'
        c=dz[cfg['source_seed']]@(decoders[cfg['source_seed']]@b);basis=b[:,None];teacher=c[:,None]*b
        frozen=None
        if native_teacher:
            frozen=np.load(p['source_native'],allow_pickle=False);sid=frozen['ids'];sg=frozen['g'];sd=decoders[cfg['source_seed']][sid]
            teacher=(dz[cfg['source_seed']][:,sid]*sg)@sd
            _,singular,vt=np.linalg.svd(sg[:,None]*sd,full_matrices=False)
            rank=int(np.sum(singular>cfg['basis_relative_tolerance']*singular[0]));basis=vt[:rank].T;c=teacher@basis
            write(run/'teacher_basis.json',dict(rank=rank,singular_values=singular.tolist(),span_relative_error=float(np.linalg.norm(teacher-c@basis.T)/np.linalg.norm(teacher)),rule='Full numerical source weighted-decoder span, not variance truncation',source_native_sha256=sha256(p['source_native'])))
        else:c=c[:,None]
        weights=np.full(len(c),1/len(c));factors={'basis':basis if native_teacher else b};obs=[]
        rawfit=fit_basis_constrained_transport(dh,c,basis,weights,ridge_fraction=cfg['ridge_fraction'],preserve_basis=native_teacher)
        factors['raw_w']=rawfit.target_factors if native_teacher else rawfit.target_factors[:,0]
        for target in cfg['target_seeds']:
            if time.perf_counter()-started>cfg['budget_seconds']:raise TimeoutError('Fit budget exceeded')
            d=decoders[target];x=dz[target];regressor=x if cfg.get('fcc_input')=='codes' else x@d
            fit=fit_basis_constrained_transport(regressor,c,basis,weights,ridge_fraction=cfg['ridge_fraction'],preserve_basis=native_teacher)
            if fit.status!='OK' and not (native_teacher and fit.status=='RANK_DEFICIENT'):raise ValueError(f'No target transport rank: {target}')
            factors[f'fcc_w_{target}']=fit.target_factors if native_teacher else fit.target_factors[:,0]
            diag=np.mean(x*x,axis=0)*np.sum(d*d,axis=1)
            rhs=np.sum((x.T@teacher/len(c))*d,axis=1) if native_teacher else (x.T@c[:,0]/len(c))*(d@b)
            score=rhs*rhs/np.maximum(diag,1e-30)
            ranking=np.lexsort((np.arange(len(score)),-score));active=ranking[diag[ranking]>1e-12]
            chosen=active[:cfg['support_size']];random=np.random.default_rng(cfg['random_seed']+target).choice(active,len(chosen),replace=False)
            for label,ids in [('native',chosen),('single',active[:1]),('random',random)]:
                g,detail=native_vector_fit(x,d,teacher,ids,cfg['ridge_fraction']) if native_teacher else native_fit(x,d,c[:,0],b,ids,cfg['ridge_fraction'])
                factors[f'{label}_ids_{target}']=ids;factors[f'{label}_g_{target}']=g
                predicted=(x[:,ids]*g)@d[ids]
                obs.append(dict(run_id=cfg['run_id'],target_seed=target,method=label,support=ids.tolist(),coefficients=g.tolist(),
                    training_normalized_error=float(np.sum((predicted-teacher)**2)/np.sum(teacher*teacher)),**detail))
            obs.append(dict(run_id=cfg['run_id'],target_seed=target,method='fcc',input_kind=cfg.get('fcc_input','decoded_sum'),input_dimension=regressor.shape[1],effective_rank=fit.effective_rank,requested_rank=fit.requested_rank,training_normalized_error=float(np.sum((regressor@fit.target_factors@basis.T-teacher)**2)/np.sum(teacher*teacher))))
            progress=json.dumps(dict(fitted_target=target,seconds=time.perf_counter()-started))
            with (run/'stdout.log').open('a') as log:log.write(progress+'\n')
            print(progress,flush=True)
        np.savez_compressed(run/'relation_factors.npz',**factors)
        # Only after fitting is complete, apply fixed factors to already-developed task codes.
        tasks=json.loads(p['task_tokens'].read_text())['rows'];cache=np.load(p['task_cache'],allow_pickle=False)
        bank={};methods=[]
        for target in cfg['target_seeds']:
            d=decoders[target]
            for label in ('fcc','native','single','random'):
                key=f'{label}_target{target}';methods.append(dict(key=key,target_seed=target,operation=label))
                for axis in ('subject','attractor'):
                    donor=swap_indices(tasks,axis);x=cache[f'codes_{target}']-cache[f'codes_{target}'][donor]
                    if label=='fcc':
                        regressor=x if cfg.get('fcc_input')=='codes' else x@d
                        delta=(regressor@factors[f'fcc_w_{target}'])@basis.T if native_teacher else (regressor@factors[f'fcc_w_{target}'])[:,None]*b
                    else:
                        ids=factors[f'{label}_ids_{target}'];g=factors[f'{label}_g_{target}'];delta=(x[:,ids]*g)@d[ids]
                    bank[f'{key}_{axis}']=delta
        methods.append(dict(key='raw_fitted',target_seed=None,operation='raw_basis_transport'))
        for axis in ('subject','attractor'):
            donor=swap_indices(tasks,axis);raw_value=(cache['hidden']-cache['hidden'][donor])@factors['raw_w']
            bank[f'raw_fitted_{axis}']=raw_value@basis.T if native_teacher else raw_value[:,None]*b
            if native_teacher:bank[f'source_native_reference_{axis}']=((cache[f"codes_{cfg['source_seed']}"][:,sid]-cache[f"codes_{cfg['source_seed']}"][donor][:,sid])*sg)@sd
        np.savez_compressed(run/'compiled_natural_deltas.npz',**bank)
        write(run/'compiled_methods.json',dict(methods=methods,sample_ids=[r['id'] for r in tasks],basis=b.tolist(),teacher_mode=cfg.get('teacher_mode','projection'),output_rank=basis.shape[1],source_native_sha256=sha256(p['source_native']) if native_teacher else None,
            target_endpoints_used=False,target_task_codes_used_for_fit=False,source_task_codes_used_for_selection=neighborhood,scope='development application; source task prototypes used only when configured; no target-task or target-response fit; no held-out causal claim'))
        with (run/'metrics.raw.jsonl').open('w') as out:
            for row in obs:out.write(json.dumps(row,sort_keys=True)+'\n')
        summary=dict(checks=dict(all_finite=all(np.isfinite(v).all() for v in bank.values()),discovery_only=True,no_target_task_fit=True),
                     fitted_targets=cfg['target_seeds'],methods=len(methods),fit_observations=len(obs),pair_count=len(c),compiled_delta_sha256=sha256(run/'compiled_natural_deltas.npz'))
        status='PASS' if all(summary['checks'].values()) else 'FAIL'
    except Exception:
        error=traceback.format_exc();(run/'stderr.log').write_text(error);summary['error']=error;print(error,flush=True)
    write(run/'environment.json',dict(python=sys.executable,python_version=platform.python_version(),platform=platform.platform(),numpy=np.__version__,cuda='not_applicable',torch='not_applicable',transformers='not_applicable',sae='frozen cache; no encoder execution'))
    summary.update(wall_seconds=time.perf_counter()-started,metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path=Path(__file__).relative_to(ROOT).as_posix(),generator_script_sha256=sha256(Path(__file__)))
    write(run/'metrics.summary.json',summary);write(run/'status.json',dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat()))
    check=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=check.ok,errors=check.errors))
    result=dict(status=status,contract_pass=check.ok,seconds=summary['wall_seconds'])
    with (run/'stdout.log').open('a') as log:log.write(json.dumps(result)+'\n')
    print(json.dumps(result),flush=True)
    return 0 if status=='PASS' and check.ok else 1


if __name__=='__main__':raise SystemExit(main())
