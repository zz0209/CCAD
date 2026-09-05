"""Small task-conditioned paired asset, with text-only selection before encoding."""
from __future__ import annotations
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ccad.activation_contract import HookPointContract, extract_primary_hook_tensor
from ccad.artifacts import sha256, validate_run_directory
from ccad.data_manifest import canonical_sha256, paired_document_split
from run_r011s1_raw_hook_asset import entry, aggregate, write_json as write


def generate(cfg, excluded):
    forbidden={word for split in ('development','reserved') for role in ('subjects','attractors')
               for pair in excluded['design'][split][role] for word in pair}
    vocabulary=[word for role in ('subjects','attractors') for pair in cfg[role] for word in pair]
    if forbidden.intersection(vocabulary) or len(set(vocabulary))!=len(vocabulary):
        raise ValueError('Repeated or excluded lexical item')
    if cfg['prepositions']!=excluded['design']['development']['prepositions']:
        raise ValueError('Prepositions must remain original development set')
    documents=[]
    for subject in cfg['subjects']:
        for attractor in cfg['attractors']:
            for prep in cfg['prepositions']:
                spec=dict(subject=subject,attractor=attractor,preposition=prep,construction=cfg['construction'])
                ident=canonical_sha256(spec)
                split=paired_document_split(cfg['dataset_version'],ident,salt=cfg['split_salt'])
                rows=[]
                for an in (0,1):
                    for sn in (0,1):
                        text=cfg['construction'].format(subject=subject[sn],attractor=attractor[an],preposition=prep)
                        rows.append(dict(id=f'{ident}:{sn}:{an}',template=ident,split=split,subject_number=sn,
                            attractor_number=an,text=text,text_sha256=canonical_sha256(text),label_source='programmatic_synthetic_not_human'))
                documents.append(dict(id=ident,split=split,rows=rows))
    eligible=sorted((d for d in documents if d['split']=='discovery'),
                    key=lambda d:canonical_sha256([cfg['selection_salt'],d['id']]))
    selected=eligible[:cfg['template_count']]
    if len(selected)!=cfg['template_count']:raise ValueError('Insufficient discovery templates')
    return documents,[r for d in selected for r in d['rows']]


class HookCaptured(Exception):
    pass


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());asset_path=ROOT/cfg['asset_config'];asset=json.loads(asset_path.read_text())
    for key in ('model_id','model_local_dir','model_revision','model_license','hook_module_path','hook_hidden_size',
                'num_latents','k','saes','device','sparsify_source_dir','sparsify_overlay_dir','sparsify_commit'):
        cfg[key]=asset[key]
    run=ROOT/'runs'/cfg['run_id'];run.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    write(run/'config.resolved.json',cfg);now=datetime.now(timezone.utc).isoformat()
    code=[]
    for rel in ('scripts/run_f4_task_paired_asset.py','scripts/run_r011s1_raw_hook_asset.py',
                'src/ccad/activation_contract.py','src/ccad/artifacts.py','src/ccad/data_manifest.py'):
        path=ROOT/rel;snap=run/'source_snapshot'/rel;snap.parent.mkdir(parents=True,exist_ok=True);snap.write_bytes(path.read_bytes())
        code.append(dict(path=rel,sha256=sha256(path),bytes=path.stat().st_size,snapshot_path=f'source_snapshot/{rel}'))
    write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='fcc.task.paired.asset.v1',run_id=cfg['run_id'],run_parent='F4',
        purpose='Independent task-near paired discovery encoding',milestone='M4',evidence_level='synthetic_task_conditioned_asset_only',
        started_utc=now,started_local=datetime.now().astimezone().isoformat(),project_root=str(ROOT),
        config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(code),source_snapshot_required=True,
        audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='original independent means cancel in differences',
        threshold_source_split='text-only rules frozen before encoding',statistics_unit='complete lexical template, dependent number conditions',
        device=cfg['device'],seeds=[s['seed'] for s in cfg['saes']],resource_lease='cpu-heavy -> gpu-0 resource_manager.run',
        resource_lease_reason='bounded hook collection and five frozen SAE encodings',model_id=cfg['model_id'],
        model_revision=cfg['model_revision'],tokenizer_revision=cfg['model_revision'],sae_framework_revision=cfg['sparsify_commit'],
        git_head_at_run=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    write(run/'status.json',dict(status='RUNNING',updated_utc=now))
    for name in ('stdout.log','stderr.log','metrics.raw.jsonl'):(run/name).touch()
    status='FAIL';summary={};forwards=0;env=dict(python=sys.executable,python_version=platform.python_version(),platform=platform.platform(),numpy=np.__version__)
    try:
        if cfg['audit_opened']:raise ValueError('Discovery only')
        excluded_path=ROOT/cfg['excluded_task_config'];spec=ROOT/'.aris/compute/local-r006b1-env-spec.json'
        if sha256(spec)!='3129a184d787ae9be38ac6d8d97dbf5087e5c838c112473fe45f3862064bb60f':raise ValueError('Environment changed')
        paths={'asset_config':asset_path,'excluded_task_config':excluded_path,'environment_spec':spec}
        for sae in cfg['saes']:
            path=ROOT/sae['path']/'sae.safetensors'
            if sha256(path)!=sae['sha256']:raise ValueError('SAE identity changed')
            paths[f'sae_{sae["seed"]}']=path;paths[f'sae_cfg_{sae["seed"]}']=path.parent/'cfg.json'
        for name in ('config.json','tokenizer.json','tokenizer_config.json'):
            paths[f'model_{name}']=Path(cfg['model_local_dir'])/name
        write(run/'inputs.json',dict(inputs=[entry(path,'CCAD frozen input / model revision',role) for role,path in paths.items()]))
        docs,rows=generate(cfg,json.loads(excluded_path.read_text()))
        write(run/'text_manifest.json',dict(documents=docs,split_template_counts=dict(Counter(d['split'] for d in docs)),
            selected_ids=[r['id'] for r in rows],rule=cfg['scope'],non_discovery_encoded=False))
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_DISABLED='true',SPARSIFY_DISABLE_TRITON='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        sys.path[:0]=[cfg['sparsify_source_dir'],cfg['sparsify_overlay_dir']]
        import torch
        import transformers
        from sparsify.sparse_coder import SparseCoder
        env.update(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,sae=cfg['sparsify_commit'],gpu=torch.cuda.get_device_name())
        torch.set_num_threads(cfg['cpu_threads']);torch.use_deterministic_algorithms(True);torch.cuda.set_device(cfg['device'])
        tok=transformers.AutoTokenizer.from_pretrained(cfg['model_local_dir'],local_files_only=True)
        encoded=[tok.encode(r['text'],add_special_tokens=False) for r in rows];last=np.array([len(x)-1 for x in encoded]);width=int(last.max()+1)
        if width>64:raise ValueError('Unexpected prompt length')
        tokens=np.full((len(rows),width),tok.eos_token_id,dtype=np.int64);mask=np.zeros_like(tokens)
        for i,ids in enumerate(encoded):tokens[i,:len(ids)]=ids;mask[i,:len(ids)]=1
        write(run/'paired_rows.json',dict(rows=[dict(r,token_ids=ids,final_position=int(pos)) for r,ids,pos in zip(rows,encoded,last)],
            pairing='adjacent rows: subject singular minus plural, attractor fixed',text_manifest_sha256=sha256(run/'text_manifest.json')))
        model=transformers.AutoModelForCausalLM.from_pretrained(cfg['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to(cfg['device'])
        saes={s['seed']:SparseCoder.load_from_disk(ROOT/s['path'],device=cfg['device']).eval() for s in cfg['saes']}
        hook=model.get_submodule(cfg['hook_module_path']);contract=HookPointContract(cfg['hook_module_path'],5,'resid_post',cfg['hook_hidden_size'])
        numeric=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        def capture(ix,unpad=False,stop=True):
            nonlocal forwards
            if forwards>=cfg['budget_forwards'] or time.perf_counter()-numeric>cfg['budget_seconds']:raise TimeoutError('Frozen budget exceeded')
            w=int(last[ix[0]]+1) if unpad else width;rr=torch.arange(len(ix),device=cfg['device']);pos=torch.tensor(last[ix],device=cfg['device']);out={}
            def save(m,args,value):
                out['hidden']=extract_primary_hook_tensor(value,contract)[rr,pos].detach().clone()
                if stop:raise HookCaptured()
            handle=hook.register_forward_hook(save)
            try:
                try:
                    with torch.no_grad():model(torch.tensor(tokens[ix,:w],device=cfg['device']),attention_mask=torch.tensor(mask[ix,:w],device=cfg['device']),use_cache=False)
                except HookCaptured:
                    if not stop:raise
            finally:handle.remove()
            forwards+=1;return out['hidden'].cpu().numpy()
        hidden=np.empty((len(rows),cfg['hook_hidden_size']),dtype=np.float32)
        for i in range(0,len(rows),cfg['batch_size']):
            ix=np.arange(i,min(i+cfg['batch_size'],len(rows)));hidden[ix]=capture(ix)
        checks=[]
        for i in (int(last.argmin()),int(last.argmax())):
            h=capture(np.array([i]),unpad=True)
            checks.append(dict(kind='unpadded',index=i,relative_l2=float(np.linalg.norm(h-hidden[i])/np.linalg.norm(hidden[i]))))
        ix=np.arange(cfg['batch_size']);full=capture(ix,stop=False)
        checks.append(dict(kind='normal_forward',relative_l2=float(np.linalg.norm(full-hidden[ix])/np.linalg.norm(hidden[ix]))))
        if any(c['relative_l2']>cfg['padding_relative_tolerance'] for c in checks):raise ValueError(f'Hook mismatch: {checks}')
        data={'hidden':hidden};observations=[]
        for seed,sae in saes.items():
            with torch.no_grad():sp=sae.encode(torch.tensor(hidden,device=cfg['device']))
            z=np.zeros((len(rows),cfg['num_latents']),dtype=np.float32);np.put_along_axis(z,sp.top_indices.cpu().numpy(),sp.top_acts.cpu().numpy(),axis=1)
            data[f'codes_{seed}']=z
            observations.append(dict(run_id=cfg['run_id'],seed=seed,split='discovery',rows=len(rows),mean_l0=float(np.mean(np.sum(z!=0,axis=1))),finite=bool(np.isfinite(z).all())))
        np.savez_compressed(run/'paired_activations.npz',**data)
        with (run/'metrics.raw.jsonl').open('w') as out:
            for row in observations:out.write(json.dumps(row)+'\n')
        summary.update(checks=checks,all_finite=all(r['finite'] for r in observations),rows=len(rows),pairs=len(rows)//2,
            numerical_seconds=time.perf_counter()-numeric,preparation_seconds=numeric-started,peak_vram_bytes=torch.cuda.max_memory_allocated(),
            paired_cache_sha256=sha256(run/'paired_activations.npz'),paired_rows_sha256=sha256(run/'paired_rows.json'),
            mean_l0_by_seed={r['seed']:r['mean_l0'] for r in observations},audit_encoded=False,task_endpoints_read=False)
        status='PASS' if summary['all_finite'] and np.isfinite(hidden).all() else 'FAIL'
    except Exception:
        error=traceback.format_exc();(run/'stderr.log').write_text(error);summary['error']=error;print(error,flush=True)
    env.setdefault('cuda','not_initialized');env.setdefault('torch','not_initialized');env.setdefault('transformers','not_initialized');env.setdefault('sae','not_initialized')
    write(run/'environment.json',env)
    summary.update(model_forwards=forwards,wall_seconds=time.perf_counter()-started,metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),
        generator_script_path=Path(__file__).relative_to(ROOT).as_posix(),generator_script_sha256=sha256(Path(__file__)))
    write(run/'metrics.summary.json',summary);write(run/'status.json',dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat()))
    check=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=check.ok,errors=check.errors))
    result=dict(status=status,contract_pass=check.ok,**summary);(run/'stdout.log').write_text(json.dumps(result)+'\n');print(json.dumps(result),flush=True)
    return 0 if status=='PASS' and check.ok else 1


if __name__=='__main__':raise SystemExit(main())
