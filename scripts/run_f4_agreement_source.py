"""Bounded factorial SVA source screen; no target-map endpoints or fitting."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ccad.activation_contract import HookPointContract, extract_primary_hook_tensor, replace_primary_hook_tensor
from ccad.artifacts import sha256, validate_run_directory
from run_r011s1_raw_hook_asset import entry, aggregate, write_json as write


def make_prompts(design):
    rows=[]
    for split in ('development','reserved'):
        spec=design[split]
        for i,subject in enumerate(spec['subjects']):
            for j,prep in enumerate(spec['prepositions']):
                attractor=spec['attractors'][(i+j)%4]
                for sn in (0,1):
                    for an in (0,1):
                        text=design['construction'].format(subject=subject[sn],preposition=prep,attractor=attractor[an])
                        rows.append(dict(id=f'{split}:{i}:{j}:{sn}:{an}',split=split,template=f'{split}:{i}:{j}',
                                         subject_number=sn,attractor_number=an,text=text,
                                         text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                         label_source='programmatic_synthetic_not_human'))
    return rows


def swap_indices(rows,axis):
    index={(r['template'],r['subject_number'],r['attractor_number']):i for i,r in enumerate(rows)}
    return np.array([index[r['template'],1-r['subject_number'] if axis=='subject' else r['subject_number'],
                          1-r['attractor_number'] if axis=='attractor' else r['attractor_number']] for r in rows])


def margins(logprobs,rows):
    sign=np.array([1 if r['subject_number']==0 else -1 for r in rows])
    return np.column_stack(((logprobs[:,0]-logprobs[:,1])*sign,
                            (logprobs[:,2]-logprobs[:,3])*sign,
                            np.logaddexp(logprobs[:,0],logprobs[:,1])-np.logaddexp(logprobs[:,2],logprobs[:,3])))


def capped(delta,hidden,fraction):
    scale=np.minimum(1.,fraction*np.linalg.norm(hidden,axis=1)/np.maximum(np.linalg.norm(delta,axis=1),1e-30))
    return delta*scale[:,None],scale


def task_contrast_basis(codes,decoder,rows):
    """Source-only plural-minus-singular decoded contrast, balanced over attractors."""
    y=codes@decoder
    donors=swap_indices(rows,'subject')
    plural=np.array([r['subject_number']==1 for r in rows])
    contrast=(y[plural]-y[donors[plural]]).mean(axis=0)
    norm=float(np.linalg.norm(contrast))
    if not np.isfinite(norm) or norm<=1e-12:
        raise ValueError('No nonzero source task contrast in this representation')
    return contrast/norm,y,norm


def box_ridge(gram,rhs,bound=1.,max_sweeps=2000,tolerance=1e-8):
    """Cyclic exact coordinate minimization of a positive-definite box QP."""
    g=np.zeros(len(rhs))
    for sweep in range(max_sweeps):
        for j in range(len(g)):
            g[j]=np.clip(g[j]+(rhs[j]-gram[j]@g)/gram[j,j],-bound,bound)
        residual=float(np.max(np.abs(g-np.clip(g-(gram@g-rhs),-bound,bound))))
        if residual<=tolerance:break
    return g,dict(sweeps=sweep+1,projected_gradient_residual=residual,converged=residual<=tolerance)


def source_native_group(codes,decoder,rows,b,spec,gradients=None):
    """Fit only source task differences to the existing projected teacher."""
    from fit_f4_agreement_relations import native_fit
    donor=swap_indices(rows,'subject');keep=np.arange(len(rows))<donor
    if gradients is not None:
        # Per-input Jacobian design, not a squared average Jacobian.
        x=codes-codes[donor];design=x*(gradients@decoder.T)
        c=(x@decoder@b)*(gradients@b)
        diag=np.mean(design*design,axis=0);rhs=design.T@c/len(c)
        score=rhs*rhs/np.maximum(diag,1e-30);order=np.lexsort((np.arange(len(score)),-score))
        ids=order[diag[order]>1e-12][:spec['support_size']]
        if len(ids)!=spec['support_size']:raise ValueError('Insufficient source adjoint support')
        a=design[:,ids];gram=a.T@a/len(c);ridge=spec['ridge_fraction']*float(np.trace(gram))/len(ids)
        regularized=gram+max(ridge,1e-12)*np.eye(len(ids));target=a.T@c/len(c)
        u=np.linalg.solve(regularized,target);g=np.clip(u,-1.,1.);solver_info=dict(solver='ridge_then_clip')
        if spec.get('solver')=='box_coordinate_descent':
            g,solver_info=box_ridge(regularized,target,max_sweeps=spec['max_sweeps'],tolerance=spec['solver_tolerance'])
            solver_info['solver']='box_coordinate_descent'
        return ids,g,dict(ridge=ridge,unconstrained_exceeds_bounds=int(np.sum(np.abs(u)>1)),at_bounds=int(np.sum(np.abs(g)>=1)),max_unclipped=float(np.max(np.abs(u))),pair_count=len(c),**solver_info,
            training_normalized_error=float(np.sum((a@g-c)**2)/np.sum(c*c)),objective='per-input primary-margin adjoint prediction of source projection',
            source_only=True,target_task_codes_used=False,behavioral_endpoints_used_for_fit=True,
            scope='Source primary-gradient development only; past/tense gradients absent; actual forward and unseen-input validation distinct')
    x=(codes-codes[donor])[keep];c=x@decoder@b
    diag=np.mean(x*x,axis=0)*np.sum(decoder*decoder,axis=1)
    rhs=(x.T@c/len(c))*(decoder@b);score=rhs*rhs/np.maximum(diag,1e-30)
    order=np.lexsort((np.arange(len(score)),-score));ids=order[diag[order]>1e-12][:spec['support_size']]
    if len(ids)!=spec['support_size']:raise ValueError('Insufficient active source support')
    g,info=native_fit(x,decoder,c,b,ids,spec['ridge_fraction'])
    predicted=(x[:,ids]*g)@decoder[ids]
    return ids,g,dict(**info,pair_count=len(c),training_normalized_error=float(np.sum((predicted-c[:,None]*b)**2)/np.sum(c*c)),
        source_only=True,target_task_codes_used=False,behavioral_endpoints_used_for_fit=False,objective='unweighted hook vector error',
        scope='Source task development vector fit; not unseen-input validation or optimal box-QP')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    asset=json.loads((ROOT/cfg['asset_config']).read_text())
    for name in ('model_id','model_local_dir','model_revision','model_license','hook_module_path','hook_hidden_size',
                 'num_latents','k','saes','device','sparsify_source_dir','sparsify_overlay_dir','sparsify_commit'):
        cfg[name]=asset[name]
    frozen_task=cfg.get('source_reference_mode')=='frozen_task'
    task_mode=cfg.get('source_reference_mode') in ('task_contrast','frozen_task')
    if task_mode:
        if not frozen_task:cfg['saes']=[s for s in cfg['saes'] if s['seed']==cfg['source_seed']]
        if (not frozen_task and len(cfg['saes'])!=1) or cfg['queries']:raise ValueError('Task mode requires fixed source SAE and no query pool')
    if frozen_task and (cfg['split']!='reserved' or not cfg.get('source_native',{}).get('frozen_path') or cfg['source_native'].get('objective')):
        raise ValueError('Reserved confirmation requires frozen native group and no gradient/fit objective')
    run=ROOT/'runs'/cfg['run_id'];run.mkdir(parents=True,exist_ok=False)
    write(run/'config.resolved.json',cfg)
    code_paths=[Path(__file__),ROOT/'scripts/run_r011s1_raw_hook_asset.py',ROOT/'src/ccad/artifacts.py',ROOT/'src/ccad/activation_contract.py']
    if cfg.get('source_native'):
        code_paths += [ROOT/'scripts/fit_f4_agreement_relations.py',ROOT/'src/ccad/causal_metric_probe.py',ROOT/'src/ccad/hook_transport.py']
    if frozen_task:code_paths.append(ROOT/'scripts/assemble_f4_paired_panel.py')
    codes=[]
    for path in code_paths:
        rel=path.relative_to(ROOT).as_posix();snap=run/'source_snapshot'/rel;snap.parent.mkdir(parents=True,exist_ok=True);snap.write_bytes(path.read_bytes())
        codes.append(dict(path=rel,sha256=sha256(path),bytes=path.stat().st_size,snapshot_path=f'source_snapshot/{rel}'))
    write(run/'code_hashes.json',dict(files=codes,aggregate_sha256=aggregate(codes),snapshot_root='source_snapshot'))
    now=datetime.now(timezone.utc).isoformat()
    write(run/'manifest.json',dict(schema_version='fcc.agreement.source.v1',run_id=cfg['run_id'],run_parent='F4',
        purpose=cfg.get('purpose','Task-specific source intervention screening before target evaluation'),milestone='M4',
        evidence_level=cfg['evidence_level'],started_utc=now,started_local=datetime.now().astimezone().isoformat(),
        trigger='ccad heartbeat',project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),
        code_snapshot_hash=aggregate(codes),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,
        reserved_confirmation_opened=frozen_task,
        mean_constants_source_split='original mean cancels in donor differences',threshold_source_split=cfg.get('selection','development ranking only'),
        statistics_unit='lexicalized template; 4number conditions and shared SAE directions dependent',device=cfg['device'],
        seeds=sorted(set([r['seed'] for r in cfg['saes']]+cfg.get('target_seeds',[]))),resource_lease=cfg['resource_lease'],resource_lease_reason='bounded model forwards and existing SAE encoder/cache',
        model_id=cfg['model_id'],model_revision=cfg['model_revision'],tokenizer_revision=cfg['model_revision'],sae_framework_revision=cfg['sparsify_commit'],
        git_head_at_run=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status_porcelain=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).splitlines()))
    write(run/'status.json',dict(status='RUNNING',updated_utc=now))
    (run/'stdout.log').touch();(run/'stderr.log').touch();(run/'metrics.raw.jsonl').touch()
    forwards=0;start=time.perf_counter();rows=[];status='FAIL';summary={}
    try:
        if cfg['split']!=('reserved' if frozen_task else 'development') or cfg['audit_opened']:raise ValueError('Task split mismatch')
        paths={'surface':ROOT/cfg['surface_path'],'factors':ROOT/cfg['factors_path'],'asset_config':ROOT/cfg['asset_config'],
               'environment_spec':ROOT/'.aris/compute/local-r006b1-env-spec.json'}
        for role in ('surface','factors'):
            if sha256(paths[role])!=cfg[role+'_sha256']:raise ValueError(f'{role} identity mismatch')
        if task_mode and not frozen_task:
            for role in ('replay_activations','replay_tokens'):
                paths[role]=ROOT/cfg[role+'_path']
                if sha256(paths[role])!=cfg[role+'_sha256']:raise ValueError(f'{role} identity mismatch')
        if 'compiled_deltas_path' in cfg:
            for role in ('compiled_deltas','compiled_methods','source_predictions'):
                paths[role]=ROOT/cfg[role+'_path']
                if sha256(paths[role])!=cfg[role+'_sha256']:raise ValueError(f'{role} identity mismatch')
        if cfg.get('source_native',{}).get('frozen_path'):
            paths['source_native_frozen']=ROOT/cfg['source_native']['frozen_path']
            if sha256(paths['source_native_frozen'])!=cfg['source_native']['frozen_sha256']:raise ValueError('Frozen source group changed')
        if frozen_task:
            for role,spec in cfg['frozen_relations'].items():
                paths[role]=ROOT/spec['path']
                if sha256(paths[role])!=spec['sha256']:raise ValueError(f'Frozen relation changed: {role}')
            if cfg['design']!=json.loads(paths['original_task_design'].read_text())['design']:raise ValueError('Original reserved task design changed')
        if sha256(paths['environment_spec'])!='3129a184d787ae9be38ac6d8d97dbf5087e5c838c112473fe45f3862064bb60f':raise ValueError('Environment spec changed')
        for sae in cfg['saes']:
            paths[f'sae_{sae["seed"]}']=ROOT/sae['path']/'sae.safetensors'
            paths[f'sae_cfg_{sae["seed"]}']=ROOT/sae['path']/'cfg.json'
            if sha256(paths[f'sae_{sae["seed"]}'])!=sae['sha256']:raise ValueError('SAE identity mismatch')
        modeldir=Path(cfg['model_local_dir'])
        for name in ('config.json','tokenizer.json','tokenizer_config.json'):
            paths['model_'+name]=modeldir/name
        write(run/'inputs.json',dict(inputs=[entry(p,'CCAD saved input / model revision',role) for role,p in paths.items()]))
        all_prompts=make_prompts(cfg['design']);write(run/'task_prompts.json',dict(rows=all_prompts,reserved_forwarded=frozen_task))
        prompts=[r for r in all_prompts if r['split']==cfg['split']]
        random.Random(cfg['random_seed']).shuffle(prompts)
        if len(prompts)!=64 or len({r['text'] for r in all_prompts})!=128:raise ValueError('Incomplete factorial task')
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_DISABLED='true',SPARSIFY_DISABLE_TRITON='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        sys.path[:0]=[cfg['sparsify_source_dir'],cfg['sparsify_overlay_dir']]
        import torch
        import transformers
        from sparsify.sparse_coder import SparseCoder
        torch.set_num_threads(cfg['cpu_threads']);torch.use_deterministic_algorithms(True);torch.cuda.set_device(cfg['device'])
        tokenizer=transformers.AutoTokenizer.from_pretrained(modeldir,local_files_only=True)
        encoded=[tokenizer.encode(r['text'],add_special_tokens=False) for r in prompts]
        continuation_ids=[tokenizer.encode(x,add_special_tokens=False) for x in cfg['continuations']]
        if any(len(x)!=1 for x in continuation_ids):raise ValueError('Need single-token continuation, not partial-word score')
        # Check actual concatenation, not just isolated continuation tokenization.
        for r,ids in zip(prompts,encoded):
            for word,wid in zip(cfg['continuations'],continuation_ids):
                if tokenizer.encode(r['text']+word,add_special_tokens=False)!=ids+wid:raise ValueError('Continuation boundary changes prompt tokens')
        last=np.array([len(x)-1 for x in encoded]);length=int(max(last)+1)
        if length>64:raise ValueError('Prompt exceeds bounded length')
        tokens=np.full((64,length),tokenizer.eos_token_id,dtype=np.int64);attention=np.zeros_like(tokens)
        for i,ids in enumerate(encoded):tokens[i,:len(ids)]=ids;attention[i,:len(ids)]=1
        token_artifact='tokenized_reserved.json' if frozen_task else 'tokenized_development.json'
        write(run/token_artifact,dict(rows=[dict(r,token_ids=ids,final_position=int(p)) for r,ids,p in zip(prompts,encoded,last)],
              continuation_ids=continuation_ids,tokenizer_revision=cfg['model_revision'],right_padding=True,reserved_tokenized=frozen_task))
        model=transformers.AutoModelForCausalLM.from_pretrained(modeldir,local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to(cfg['device'])
        saes={r['seed']:SparseCoder.load_from_disk(ROOT/r['path'],device=cfg['device']).eval() for r in cfg['saes']}
        hook=model.get_submodule(cfg['hook_module_path']);contract=HookPointContract(cfg['hook_module_path'],5,'resid_post',cfg['hook_hidden_size'])
        torch.cuda.reset_peak_memory_stats();numeric_start=time.perf_counter();batchsize=cfg['batch_size']
        ids4=torch.tensor([x[0] for x in continuation_ids],device=cfg['device'])
        def forward(indices,delta=None,unpad=False):
            nonlocal forwards
            if time.perf_counter()-numeric_start>cfg['budget_seconds'] or forwards>=cfg['budget_forwards']+2:raise TimeoutError('Frozen compute budget exceeded')
            n=len(indices);width=int(last[indices[0]]+1) if unpad else length
            batch=torch.tensor(tokens[indices,:width],device=cfg['device']);mask=torch.tensor(attention[indices,:width],device=cfg['device'])
            pos=torch.tensor(last[indices],device=cfg['device']);rr=torch.arange(n,device=cfg['device']);cap={}
            def intervene(m,i,out):
                h=extract_primary_hook_tensor(out,contract);cap['hook']=h[rr,pos].detach().clone()
                if delta is not None:
                    h=h.clone();h[rr,pos]-=torch.tensor(delta,dtype=h.dtype,device=h.device)
                    return replace_primary_hook_tensor(out,h,contract)
            handle=hook.register_forward_hook(intervene)
            try:
                with torch.no_grad():logits=model(batch,attention_mask=mask,use_cache=False).logits[rr,pos].double()
                out=torch.log_softmax(logits,dim=-1)[:,ids4].cpu().numpy()
            finally:handle.remove()
            forwards+=1;return out,cap['hook'].cpu().numpy()
        batches=[np.arange(i,min(i+batchsize,len(prompts))) for i in range(0,len(prompts),batchsize)]
        base=np.empty((64,4));hidden=np.empty((64,cfg['hook_hidden_size']));noop=0.
        for ix in batches:base[ix],hidden[ix]=forward(ix)
        for ix in batches:
            lp,_=forward(ix,np.zeros_like(hidden[ix]));noop=max(noop,float(np.max(np.abs(lp-base[ix]))))
        unpad_error=0.;padding_checks=[]
        for i in (int(np.argmin(last)),int(np.argmax(last))):
            lp,h=forward(np.array([i]),unpad=True)
            padding_checks.append(dict(sample_id=prompts[i]['id'],logprob_max_abs=float(np.max(np.abs(lp-base[i]))),
                hook_max_abs=float(np.max(np.abs(h-hidden[i]))),hook_relative_l2=float(np.linalg.norm(h-hidden[i])/np.linalg.norm(hidden[i])),
                margin_max_abs=float(np.max(np.abs(margins(lp,[prompts[i]])-margins(base[i:i+1],[prompts[i]]))))))
        write(run/'padding_checks.json',dict(noop_max_abs=noop,checks=padding_checks,tolerances=cfg['padding_tolerances'],
            scope='Float32 batch/unpadded equivalence diagnostic; nonzero noise retained against small source effects'))
        unpad_error=max(r['logprob_max_abs'] for r in padding_checks)
        padding_ok=all(r['logprob_max_abs']<=cfg['padding_tolerances']['logprob_absolute'] and r['hook_relative_l2']<=cfg['padding_tolerances']['hook_relative_l2'] for r in padding_checks)
        if noop>1e-6 or not padding_ok:raise ValueError(f'Noop/padding mismatch {noop} {padding_checks}')
        baseline=margins(base,prompts)
        z={};dec={}
        for seed,sae in saes.items():
            with torch.no_grad():sparse=sae.encode(torch.tensor(hidden,dtype=torch.float32,device=cfg['device']))
            z[seed]=np.zeros((64,cfg['num_latents']));np.put_along_axis(z[seed],sparse.top_indices.cpu().numpy(),sparse.top_acts.cpu().numpy(),axis=1)
            dec[seed]=sae.W_dec.detach().cpu().numpy().astype(np.float64)
        cache_artifact='reserved_activations.npz' if frozen_task else 'development_activations.npz'
        np.savez_compressed(run/cache_artifact,hidden=hidden,logprobs=base,**{f'codes_{s}':v for s,v in z.items()})
        replay_ok=True
        if task_mode and not frozen_task:
            old=np.load(paths['replay_activations'],allow_pickle=False)
            old_tokens=json.loads(paths['replay_tokens'].read_text())
            current_tokens=json.loads((run/'tokenized_development.json').read_text())
            replay={name:float(np.max(np.abs(current-old[name]))) for name,current in
                    [('hidden',hidden),('logprobs',base),(f'codes_{cfg["source_seed"]}',z[cfg['source_seed']])]}
            replay_ok=old_tokens==current_tokens and all(v<=1e-8 for v in replay.values())
            write(run/'input_replay.json',dict(max_abs_errors=replay,tokenized_rows_equal=old_tokens==current_tokens,passed=replay_ok))
            if not replay_ok:raise ValueError(f'Saved development input replay changed: {replay}')
            task_b,task_y,contrast_norm=task_contrast_basis(z[cfg['source_seed']],dec[cfg['source_seed']],prompts)
            np.savez_compressed(run/'source_task_direction.npz',basis=task_b,source_contributions=task_y)
            write(run/'source_task_direction.json',dict(source_seed=cfg['source_seed'],contrast_norm=contrast_norm,
                rule='Mean plural-minus-singular full source decoded contribution over 32 matched template/attractor pairs; normalized once',
                input_split='development',target_endpoints_used=False,task_effects_used_to_fit_direction=False,
                sae_config=json.loads(paths[f'sae_cfg_{cfg["source_seed"]}'].read_text())))
        if frozen_task:
            task_b=np.load(paths['frozen_direction'],allow_pickle=False)['basis'];task_y=z[cfg['source_seed']]@dec[cfg['source_seed']]
            if task_b.shape!=(cfg['hook_hidden_size'],) or not np.isclose(np.linalg.norm(task_b),1):raise ValueError('Frozen basis invalid')
            np.savez_compressed(run/'source_task_direction.npz',basis=task_b,source_contributions=task_y)
            write(run/'source_task_direction.json',dict(source_seed=cfg['source_seed'],rule='Frozen development basis, no reserved contrasts fitted',
                frozen_sha256=sha256(paths['frozen_direction']),input_split='reserved_application_only',refitted=False,target_endpoints_used=False))
        surface=[json.loads(x) for x in paths['surface'].read_text().splitlines() if x]
        factors=np.load(paths['factors'],allow_pickle=False)
        findex={(int(s),int(a),int(t)):i for i,(s,a,t) in enumerate(zip(factors['source_seed'],factors['source_atom'],factors['target_seed']))}
        families={}
        for s,a in cfg['queries']:
            records=sorted([r for r in surface if r['source_seed']==s and r['source_atom']==a and r['rank']==1 and r['query_role']=='anchor'],key=lambda r:r['target_seed'])
            r=records[0];support=r['source_candidate_ids'];b=np.asarray(factors['source_basis'][findex[s,a,r['target_seed']],:,0],dtype=np.float64)
            # Source reference must not change with the arbitrary target-index carrier.
            for other in records[1:]:
                ob=np.asarray(factors['source_basis'][findex[s,a,other['target_seed']],:,0],dtype=np.float64)
                if other['source_candidate_ids']!=support or not np.allclose(np.outer(b,b),np.outer(ob,ob),atol=1e-8):raise ValueError('Target-dependent source reference')
            families[s,a]=(support,b)
        for i,p in enumerate(prompts):rows.append(dict(run_id=cfg['run_id'],method='baseline',**p,logprobs=base[i].tolist(),margins=baseline[i].tolist()))
        specs=([('source_task_projection',cfg['source_seed'],None),('source_full_sae_swap',cfg['source_seed'],None)] if task_mode
               else [('raw_hook_swap',None,None)]+[('source_query',s,a) for s,a in cfg['queries']])
        if cfg.get('source_native'):
            if not task_mode:raise ValueError('Source native teacher requires task mode')
            fit_start=time.perf_counter();s=cfg['source_seed']
            gradients=None
            if cfg['source_native'].get('objective')=='primary_adjoint':
                gradients=np.empty_like(hidden)
                for parameter in model.parameters():parameter.requires_grad_(False)
                for ix in batches:
                    cap={};pos=torch.tensor(last[ix],device=cfg['device']);rr=torch.arange(len(ix),device=cfg['device'])
                    def gradient_hook(m,i,out):
                        leaf=extract_primary_hook_tensor(out,contract).detach().clone().requires_grad_(True);cap['leaf']=leaf
                        return replace_primary_hook_tensor(out,leaf,contract)
                    handle=hook.register_forward_hook(gradient_hook)
                    try:
                        logits=model(torch.tensor(tokens[ix],device=cfg['device']),attention_mask=torch.tensor(attention[ix],device=cfg['device']),use_cache=False).logits[rr,pos]
                        sign=torch.tensor([1. if prompts[i]['subject_number']==0 else -1. for i in ix],device=cfg['device'])
                        goal=((logits[:,ids4[0]]-logits[:,ids4[1]])*sign).sum()
                        gradients[ix]=torch.autograd.grad(goal,cap['leaf'])[0][rr,pos].detach().cpu().numpy()
                        forwards+=1
                    finally:handle.remove()
                np.savez_compressed(run/'source_primary_gradients.npz',gradients=gradients)
                v=gradients[:1]/np.linalg.norm(gradients[:1]);eps=.001*np.linalg.norm(hidden[0])
                plus,_=forward(np.array([0]),-eps*v);minus,_=forward(np.array([0]),eps*v)
                observed=float((margins(plus,prompts[:1])[0,0]-margins(minus,prompts[:1])[0,0])/(2*eps));predicted=float(gradients[0]@v[0])
                witness=dict(observed_derivative=observed,predicted_derivative=predicted,epsilon=float(eps),passed=bool(np.isclose(observed,predicted,rtol=.03,atol=.001)),gradient_batches=len(batches),source_only=True)
                write(run/'gradient_witness.json',witness)
                if not witness['passed']:raise ValueError('Source gradient finite-difference mismatch')
            if 'source_native_frozen' in paths:
                frozen=np.load(paths['source_native_frozen'],allow_pickle=False);native_ids=frozen['ids'];native_g=frozen['g']
                if not np.allclose(frozen['basis'],task_b,rtol=0,atol=1e-12):raise ValueError('Frozen source task basis changed')
                native_info=dict(frozen=True,source_native_sha256=sha256(paths['source_native_frozen']),refitted=False)
            else:native_ids,native_g,native_info=source_native_group(z[s],dec[s],prompts,task_b,cfg['source_native'],gradients)
            native_info.update(fit_seconds=time.perf_counter()-fit_start,support=native_ids.tolist(),coefficients=native_g.tolist())
            if native_info['fit_seconds']>cfg['source_native']['fit_budget_seconds']:raise TimeoutError('Source native fit budget')
            np.savez_compressed(run/'source_native_factors.npz',ids=native_ids,g=native_g,basis=task_b)
            write(run/'source_native_fit.json',native_info)
            specs.append(('source_native_group',s,None))
        compiled={};panel=np.arange(len(prompts));bank=None
        if frozen_task:
            from assemble_f4_paired_panel import compile_frozen_task_panel
            application_cache=dict(hidden=hidden,**{f'codes_{s}':v for s,v in z.items()})
            with np.load(paths['decoded_relations'],allow_pickle=False) as decoded, np.load(paths['code_relations'],allow_pickle=False) as code, np.load(paths['source_native_frozen'],allow_pickle=False) as native:
                bank,methods=compile_frozen_task_panel(prompts,application_cache,dec,decoded,code,native,cfg['source_seed'])
            cm=dict(methods=methods,sample_ids=[p['id'] for p in prompts],basis=task_b.tolist(),source_native_sha256=sha256(paths['source_native_frozen']),target_endpoints_used=False,refitted=False)
            np.savez_compressed(run/'compiled_natural_deltas.npz',**bank);write(run/'compiled_methods.json',cm)
            write(run/'application_before_target_forward.json',dict(frozen_relations=cfg['frozen_relations'],
                compiled_delta_sha256=sha256(run/'compiled_natural_deltas.npz'),compiled_methods_sha256=sha256(run/'compiled_methods.json'),
                forward_count=forwards,target_interventions_started=False,refitted=False,scope='Reserved codes applied to frozen parameters; no target behavior/gradients used to select or fit'))
        if 'compiled_deltas_path' in cfg or frozen_task:
            if not task_mode:raise ValueError('Compiled operations require the frozen task source reference')
            if not frozen_task:cm=json.loads(paths['compiled_methods'].read_text());bank=np.load(paths['compiled_deltas'],allow_pickle=False)
            if cm['sample_ids']!=[p['id'] for p in prompts] or not np.allclose(cm['basis'],task_b,rtol=0,atol=1e-12):raise ValueError('Compiled input order/source basis changed')
            native_reference=cfg.get('compiled_reference_method')=='source_native_group'
            if native_reference:
                if cm.get('source_native_sha256')!=sha256(paths['source_native_frozen']):raise ValueError('Compiled native teacher identity mismatch')
                for axis in ('subject','attractor'):
                    donor=swap_indices(prompts,axis);reference=((z[cfg['source_seed']][:,native_ids]-z[cfg['source_seed']][donor][:,native_ids])*native_g)@dec[cfg['source_seed']][native_ids]
                    np.testing.assert_allclose(bank[f'source_native_reference_{axis}'],reference,rtol=1e-10,atol=1e-12)
            panel=np.array([i for i,p in enumerate(prompts) if p['template'] in cfg['panel_templates']])
            if len(panel)!=cfg['panel_size']:raise ValueError('Compiled panel size mismatch')
            compiled={r['key']:r for r in cm['methods']}
            if len(compiled)!=cfg['compiled_method_count']:raise ValueError('Compiled method count mismatch')
            if not frozen_task:
                source_rows=[json.loads(line) for line in paths['source_predictions'].read_text().splitlines() if line]
                source_predictions={(r['id'],r['axis']):r['margin_loss'] for r in source_rows if r['method']==cfg.get('compiled_reference_method','source_task_projection')}
                predictions=[dict(method=method,target_seed=compiled[method]['target_seed'],sample_id=prompts[i]['id'],axis=axis,
                                  predicted_margin_loss=source_predictions[prompts[i]['id'],axis])
                             for method in compiled for axis in ('subject','attractor') for i in panel]
                write(run/'predictions_before_target_forward.json',dict(rows=predictions,rule='Matched operation should reproduce previously observed source reference effects on same development input',
                    limitation='Prospective target-response prediction, not unseen-input prediction; source effects were already observed',target_forward_started=False))
            specs += [(method,cfg['source_seed'],None) for method in compiled]
        for method,s,a in specs:
            if frozen_task and method in compiled and not (run/'predictions_before_target_forward.json').exists():
                source_predictions={(r['id'],r['axis']):r['margin_loss'] for r in rows if r['method']=='source_native_group'}
                predictions=[dict(method=m,target_seed=compiled[m]['target_seed'],sample_id=prompts[i]['id'],axis=axis,predicted_margin_loss=source_predictions[prompts[i]['id'],axis]) for m in compiled for axis in ('subject','attractor') for i in panel]
                write(run/'predictions_before_target_forward.json',dict(rows=predictions,rule='Predict reserved target effects from measured frozen source-native effects, no refit or outcome selection',
                    limitation='New-input transport test with source effects measured first, not source-free numerical forecast',forward_count=forwards,target_forward_started=False))
            for axis in ('subject','attractor'):
                donor=swap_indices(prompts,axis)
                if method in compiled:natural=np.asarray(bank[f'{method}_{axis}'])
                elif method=='raw_hook_swap':natural=hidden-hidden[donor]
                elif method=='source_full_sae_swap':natural=task_y-task_y[donor]
                elif method=='source_task_projection':natural=((task_y-task_y[donor])@task_b)[:,None]*task_b[None,:]
                elif method=='source_native_group':natural=((z[s][:,native_ids]-z[s][donor][:,native_ids])*native_g)@dec[s][native_ids]
                else:
                    support,b=families[s,a];local=(z[s][:,support]-z[s][donor][:,support])@dec[s][support]
                    natural=(local@b)[:,None]*b[None,:]
                if method in compiled:
                    ref=np.asarray(bank[f'source_native_reference_{axis}']) if native_reference else ((task_y-task_y[donor])@task_b)[:,None]*task_b
                    _,scale=capped(ref,hidden,cfg['maximum_source_hook_fraction'])
                    delta=natural*scale[:,None];indices=panel
                    work_batches=[panel[i:i+batchsize] for i in range(0,len(panel),batchsize)]
                else:
                    delta,scale=capped(natural,hidden,cfg['maximum_source_hook_fraction']);indices=np.arange(len(prompts));work_batches=batches
                lp=base.copy()
                for ix in work_batches:lp[ix],_=forward(ix,delta[ix])
                changed=margins(lp,prompts)
                for i in indices:
                    p=prompts[i]
                    rows.append(dict(run_id=cfg['run_id'],method=method,source_seed=s,source_atom=a,axis=axis,**p,
                        **(dict(target_seed=compiled[method]['target_seed'],operation=compiled[method]['operation'],scale_rule='common_source_native_scale' if native_reference else 'common_source_projection_scale') if method in compiled else {}),
                        donor_id=prompts[donor[i]]['id'],dose_scale=float(scale[i]),hook_fraction=float(np.linalg.norm(delta[i])/np.linalg.norm(hidden[i])),
                        natural_norm=float(np.linalg.norm(natural[i])),logprobs=lp[i].tolist(),margins=changed[i].tolist(),
                        margin_loss=(baseline[i]-changed[i]).tolist()))
            with (run/'metrics.raw.jsonl').open('w',encoding='utf-8') as sink:
                for row in rows:sink.write(json.dumps(row,sort_keys=True)+'\n')
            progress=json.dumps(dict(completed=method,query=[s,a],forwards=forwards))
            with (run/'stdout.log').open('a',encoding='utf-8') as log:log.write(progress+'\n')
            print(progress,flush=True)
        # All reported numeric summaries are reductions of the saved raw observations.
        rows=[json.loads(x) for x in (run/'metrics.raw.jsonl').read_text().splitlines()]
        baseline_rows=[r for r in rows if r['method']=='baseline'];ranking=[]
        for s,a in cfg['queries']:
            vals={axis:np.array([r['margin_loss'] for r in rows if r['method']=='source_query' and r['source_seed']==s and r['source_atom']==a and r['axis']==axis]) for axis in ('subject','attractor')}
            ranking.append(dict(source_seed=s,source_atom=a,subject_mean_loss=float(vals['subject'][:,0].mean()),
                subject_median_loss=float(np.median(vals['subject'][:,0])),subject_positive=int((vals['subject'][:,0]>0).sum()),
                attractor_mean_abs_loss=float(np.abs(vals['attractor'][:,0]).mean()),
                subject_mean_past_loss=float(vals['subject'][:,1].mean()),subject_mean_abs_tense_shift=float(np.abs(vals['subject'][:,2]).mean())))
        ranking.sort(key=lambda r:(-r['subject_mean_loss'],r['source_atom'],r['source_seed']))
        write(run/'source_shortlist.json',dict(rule=cfg['selection'],ranking=ranking,selected=ranking[:2],target_endpoints_used=False,reserved_forwarded=frozen_task))
        by_condition={}
        for sn in (0,1):
            for an in (0,1):
                values=np.array([r['margins'] for r in baseline_rows if r['subject_number']==sn and r['attractor_number']==an])
                by_condition[f'{sn}/{an}']=dict(n=len(values),primary_correct=int((values[:,0]>0).sum()),primary_mean_margin=float(values[:,0].mean()),past_correct=int((values[:,1]>0).sum()))
        expected_rows=len(prompts)*(1+2*(len(specs)-len(compiled)))+2*len(panel)*len(compiled)
        checks=dict(noop=noop<=1e-6,padding_equivalence=padding_ok,all_rows=len(rows)==expected_rows,
                    all_finite=all(np.isfinite(r['logprobs']).all() for r in rows),source_cap=max(r.get('hook_fraction',0) for r in rows if r['method'] not in compiled)<=.10000001)
        if frozen_task:checks['reserved_parameters_not_refitted']=native_info['frozen'] and not native_info['refitted'] and not ranking
        else:checks['reserved_not_forwarded']=True
        if task_mode and not frozen_task:checks['saved_input_replay']=replay_ok
        method_summary=[]
        for method,s,a in specs:
            for axis in ('subject','attractor'):
                obs=[r for r in rows if r['method']==method and r['axis']==axis and r['source_seed']==s and r['source_atom']==a]
                values=np.array([r['margin_loss'] for r in obs]);template_means={t:float(np.mean([r['margin_loss'][0] for r in obs if r['template']==t])) for t in sorted({r['template'] for r in obs})}
                method_summary.append(dict(method=method,source_seed=s,source_atom=a,axis=axis,n=len(obs),
                    mean_margin_loss=values.mean(axis=0).tolist(),mean_abs_margin_loss=np.abs(values).mean(axis=0).tolist(),
                    median_primary_loss=float(np.median(values[:,0])),positive_primary=int((values[:,0]>0).sum()),
                    primary_range=[float(values[:,0].min()),float(values[:,0].max())],template_means=template_means,
                    positive_templates=sum(v>0 for v in template_means.values()),
                    mean_hook_fraction=float(np.mean([r['hook_fraction'] for r in obs])),
                    mean_natural_norm=float(np.mean([r['natural_norm'] for r in obs]))))
        summary['method_summary']=method_summary
        summary.update(checks=checks,baseline_by_condition=by_condition,source_ranking=ranking,model_forwards=forwards,numeric_seconds=time.perf_counter()-numeric_start,
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),max_noop=noop,max_unpadded_error=unpad_error,rows=len(rows),
            primary_correct=sum(r['margins'][0]>0 for r in baseline_rows),primary_total=len(baseline_rows),
            scope=cfg.get('scope','Synthetic source-only development; no cross-seed correspondence claim or reserved results'))
        status='PASS' if all(checks.values()) else 'FAIL'
        write(run/'environment.json',dict(platform=platform.platform(),python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,
             transformers=transformers.__version__,numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),sae=cfg['sparsify_commit']))
    except Exception:
        error=traceback.format_exc();(run/'stderr.log').write_text(error,encoding='utf-8');print(error,flush=True)
        summary['error']=error
        if not (run/'environment.json').exists():write(run/'environment.json',dict(python=sys.executable,platform=platform.platform(),state='failed_before_full_capture'))
    summary.update(model_forwards=forwards,wall_seconds=time.perf_counter()-start,metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),
                   generator_script_path=Path(__file__).relative_to(ROOT).as_posix(),generator_script_sha256=sha256(Path(__file__)))
    write(run/'metrics.summary.json',summary);write(run/'status.json',dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat()))
    validation=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=validation.ok,errors=validation.errors))
    result=dict(status=status,contract_pass=validation.ok,forwards=forwards,wall_seconds=summary['wall_seconds'])
    with (run/'stdout.log').open('a',encoding='utf-8') as log:log.write(json.dumps(result)+'\n')
    print(json.dumps(result),flush=True)
    return 0 if status=='PASS' and validation.ok else 1


if __name__=='__main__':raise SystemExit(main())
