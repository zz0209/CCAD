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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    asset=json.loads((ROOT/cfg['asset_config']).read_text())
    for name in ('model_id','model_local_dir','model_revision','model_license','hook_module_path','hook_hidden_size',
                 'num_latents','k','saes','device','sparsify_source_dir','sparsify_overlay_dir','sparsify_commit'):
        cfg[name]=asset[name]
    run=ROOT/'runs'/cfg['run_id'];run.mkdir(parents=True,exist_ok=False)
    write(run/'config.resolved.json',cfg)
    code_paths=[Path(__file__),ROOT/'scripts/run_r011s1_raw_hook_asset.py',ROOT/'src/ccad/artifacts.py',ROOT/'src/ccad/activation_contract.py']
    codes=[]
    for path in code_paths:
        rel=path.relative_to(ROOT).as_posix();snap=run/'source_snapshot'/rel;snap.parent.mkdir(parents=True,exist_ok=True);snap.write_bytes(path.read_bytes())
        codes.append(dict(path=rel,sha256=sha256(path),bytes=path.stat().st_size,snapshot_path=f'source_snapshot/{rel}'))
    write(run/'code_hashes.json',dict(files=codes,aggregate_sha256=aggregate(codes),snapshot_root='source_snapshot'))
    now=datetime.now(timezone.utc).isoformat()
    write(run/'manifest.json',dict(schema_version='fcc.agreement.source.v1',run_id=cfg['run_id'],run_parent='F4',
        purpose='Task-specific source intervention screening before target evaluation',milestone='M4',
        evidence_level=cfg['evidence_level'],started_utc=now,started_local=datetime.now().astimezone().isoformat(),
        trigger='ccad heartbeat',project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),
        code_snapshot_hash=aggregate(codes),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,
        mean_constants_source_split='original mean cancels in donor differences',threshold_source_split='development ranking only',
        statistics_unit='lexicalized template; 4number conditions and shared SAE directions dependent',device=cfg['device'],
        seeds=[r['seed'] for r in cfg['saes']],resource_lease=cfg['resource_lease'],resource_lease_reason='bounded model forwards and five existing SAE encoders',
        model_id=cfg['model_id'],model_revision=cfg['model_revision'],tokenizer_revision=cfg['model_revision'],sae_framework_revision=cfg['sparsify_commit'],
        git_head_at_run=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status_porcelain=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).splitlines()))
    write(run/'status.json',dict(status='RUNNING',updated_utc=now))
    (run/'stdout.log').touch();(run/'stderr.log').touch();(run/'metrics.raw.jsonl').touch()
    forwards=0;start=time.perf_counter();rows=[];status='FAIL';summary={}
    try:
        if cfg['split']!='development' or cfg['audit_opened']:raise ValueError('Source screen may only expose development')
        paths={'surface':ROOT/cfg['surface_path'],'factors':ROOT/cfg['factors_path'],'asset_config':ROOT/cfg['asset_config'],
               'environment_spec':ROOT/'.aris/compute/local-r006b1-env-spec.json'}
        for role in ('surface','factors'):
            if sha256(paths[role])!=cfg[role+'_sha256']:raise ValueError(f'{role} identity mismatch')
        if sha256(paths['environment_spec'])!='3129a184d787ae9be38ac6d8d97dbf5087e5c838c112473fe45f3862064bb60f':raise ValueError('Environment spec changed')
        for sae in cfg['saes']:
            paths[f'sae_{sae["seed"]}']=ROOT/sae['path']/'sae.safetensors'
            paths[f'sae_cfg_{sae["seed"]}']=ROOT/sae['path']/'cfg.json'
            if sha256(paths[f'sae_{sae["seed"]}'])!=sae['sha256']:raise ValueError('SAE identity mismatch')
        modeldir=Path(cfg['model_local_dir'])
        for name in ('config.json','tokenizer.json','tokenizer_config.json'):
            paths['model_'+name]=modeldir/name
        write(run/'inputs.json',dict(inputs=[entry(p,'CCAD saved input / model revision',role) for role,p in paths.items()]))
        all_prompts=make_prompts(cfg['design']);write(run/'task_prompts.json',dict(rows=all_prompts,reserved_forwarded=False))
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
        write(run/'tokenized_development.json',dict(rows=[dict(r,token_ids=ids,final_position=int(p)) for r,ids,p in zip(prompts,encoded,last)],
              continuation_ids=continuation_ids,tokenizer_revision=cfg['model_revision'],right_padding=True,reserved_tokenized=False))
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
        np.savez_compressed(run/'development_activations.npz',hidden=hidden,logprobs=base,**{f'codes_{s}':v for s,v in z.items()})
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
        specs=[('raw_hook_swap',None,None)]+[('source_query',s,a) for s,a in cfg['queries']]
        for method,s,a in specs:
            for axis in ('subject','attractor'):
                donor=swap_indices(prompts,axis)
                if method=='raw_hook_swap':natural=hidden-hidden[donor]
                else:
                    support,b=families[s,a];local=(z[s][:,support]-z[s][donor][:,support])@dec[s][support]
                    natural=(local@b)[:,None]*b[None,:]
                delta,scale=capped(natural,hidden,cfg['maximum_source_hook_fraction']);lp=np.empty_like(base)
                for ix in batches:lp[ix],_=forward(ix,delta[ix])
                changed=margins(lp,prompts)
                for i,p in enumerate(prompts):
                    rows.append(dict(run_id=cfg['run_id'],method=method,source_seed=s,source_atom=a,axis=axis,**p,
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
        write(run/'source_shortlist.json',dict(rule=cfg['selection'],ranking=ranking,selected=ranking[:2],target_endpoints_used=False,reserved_forwarded=False))
        by_condition={}
        for sn in (0,1):
            for an in (0,1):
                values=np.array([r['margins'] for r in baseline_rows if r['subject_number']==sn and r['attractor_number']==an])
                by_condition[f'{sn}/{an}']=dict(n=len(values),primary_correct=int((values[:,0]>0).sum()),primary_mean_margin=float(values[:,0].mean()),past_correct=int((values[:,1]>0).sum()))
        checks=dict(noop=noop<=1e-6,padding_equivalence=padding_ok,all_rows=len(rows)==len(prompts)*(1+2*len(specs)),
                    all_finite=all(np.isfinite(r['logprobs']).all() for r in rows),source_cap=max(r.get('hook_fraction',0) for r in rows)<=.10000001,reserved_not_forwarded=True)
        summary.update(checks=checks,baseline_by_condition=by_condition,source_ranking=ranking,model_forwards=forwards,numeric_seconds=time.perf_counter()-numeric_start,
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),max_noop=noop,max_unpadded_error=unpad_error,rows=len(rows),
            primary_correct=sum(r['margins'][0]>0 for r in baseline_rows),primary_total=len(baseline_rows),
            scope='Synthetic source-only development; no cross-seed correspondence claim or reserved results')
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
