"""Raw material screen for separable number/time changes and their composition."""
import argparse,json,os,platform,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import numpy as np
from run_r011s1_raw_hook_asset import ROOT,entry,aggregate,write_json as write
from run_functional_material import logprob
from composition_panel import make_panel
from ccad.artifacts import sha256,validate_run_directory
from ccad.activation_contract import HookPointContract,extract_primary_hook_tensor,replace_primary_hook_tensor
from composition_runtime import realized_delta_norm


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args();cfg=json.loads(args.config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False)
    start=time.perf_counter();inputs=[];metrics=[];checks={};error=None;env={};forwards=0;write(run/'config.resolved.json',cfg);code=[]
    for rel in ['scripts/run_composition_material.py','scripts/composition_runtime.py','scripts/composition_panel.py','scripts/run_functional_material.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/factor_correspondence.py','src/ccad/nip_baselines.py','src/ccad/proposal.py','src/ccad/artifacts.py','src/ccad/activation_contract.py']:
        p=ROOT/rel;q=run/'source_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='composition.material.v1',run_id=cfg['run_id'],run_parent='SEVEN_R3',purpose=cfg['purpose'],milestone='compositional-explanation',evidence_level='controlled_development',started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(code),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='fixed centering cancels in donor differences',threshold_source_split='complete authored material before model outcomes',statistics_unit='lexical block and time cue; repeated factorial conditions dependent',device='cuda:0',seeds=[],resource_lease='gpu-0 resource_manager.run',resource_lease_reason=cfg['budget']))
    if cfg.get('evidence_level'):
        manifest=json.loads((run/'manifest.json').read_text());manifest.update(evidence_level=cfg['evidence_level'],audit_opened=cfg.get('audit_opened',False),run_parent=cfg.get('run_parent','SEVEN_R3'));write(run/'manifest.json',manifest)
    for fn in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/fn).touch()
    write(run/'status.json',dict(status='RUNNING'))
    def checked(path):
        p=Path(path);p=p if p.is_absolute() else ROOT/p;inputs.append(entry(p,'existing pinned CCAD assets or authored config','input'));return p
    def record(row):
        line=json.dumps(row)
        with (run/'metrics.raw.jsonl').open('a') as f:f.write(line+'\n')
        metrics.append(row)
    def progress(stage,**kw):
        result=dict(stage=stage,elapsed=time.perf_counter()-start,sequence_forwards=forwards,**kw);write(run/'progress.json',result);print(json.dumps(result),flush=True)
    try:
        import torch,transformers
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high');torch.cuda.reset_peak_memory_stats()
        checked(args.config);modeldir=Path(cfg['model_local_dir'])
        for fn in ['config.json','tokenizer.json','model.safetensors']:checked(modeldir/fn)
        tokenizer=transformers.AutoTokenizer.from_pretrained(modeldir,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
        model=transformers.AutoModelForCausalLM.from_pretrained(modeldir,local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().cuda();model.config.use_cache=False
        rows,pairs=make_panel(cfg);n=len(rows);bs=cfg['batch_size'];tokenrows=[]
        for row in rows:
            full=tokenizer.encode(row['text'],add_special_tokens=False);ends=[]
            for j in range(len(row['spans'])):
                prefix=tokenizer.encode(''.join(row['spans'][:j+1]),add_special_tokens=False);assert full[:len(prefix)]==prefix;ends.append(len(prefix)-1)
            tokenrows.append(dict(id=row['id'],ids=full,region_ends=ends))
        names=cfg.get('position_names',['time','number','distractor'])
        aliases=cfg.get('position_aliases',{})
        positions=np.array([[tr['region_ends'][r['positions_by_region'][aliases.get(name,name)]] for name in names] for r,tr in zip(rows,tokenrows)]);label_tokens=[tokenizer.encode(s,add_special_tokens=False) for s in cfg['labels']];assert all(len(s)==1 for s in label_tokens);labels=np.array([s[0] for s in label_tokens]);checks['token_alignment']=True
        write(run/'panel.json',dict(rows=rows,pairs=pairs,token_rows=tokenrows,labels=cfg['labels'],label_ids=labels.tolist(),position_names=names,position_aliases=aliases,scope=cfg['scope']))
        captures={l:np.empty((n,len(names),model.config.hidden_size),np.float32) for l in cfg['layers']}
        def forward(ids,layer=None,delta=None,capture=False):
            nonlocal forwards
            if time.perf_counter()-start>cfg['budget_seconds']:raise TimeoutError('Raw composition screen budget')
            enc=tokenizer([rows[i]['text'] for i in ids],add_special_tokens=False,padding=True,return_tensors='pt').to('cuda');ix=torch.arange(len(ids),device='cuda');last=enc.attention_mask.sum(1)-1;handles=[]
            def make_hook(l):
                contract=HookPointContract(f'gpt_neox.layers.{l}',l,'resid_post',model.config.hidden_size)
                def hook(mod,args,out):
                    h=extract_primary_hook_tensor(out,contract)
                    if capture:captures[l][ids]=h[ix[:,None],torch.as_tensor(positions[ids],device='cuda')].detach().cpu().numpy()
                    if delta is not None and layer==l:
                        h=h.clone()
                        for slot in range(len(names)):h[ix,torch.as_tensor(positions[ids,slot],device='cuda')]+=torch.as_tensor(delta[:,slot],device='cuda')
                        return replace_primary_hook_tensor(out,h,contract)
                    return out
                return hook
            for l in cfg['layers'] if capture else [layer] if layer is not None else []:handles.append(model.get_submodule(f'gpt_neox.layers.{l}').register_forward_hook(make_hook(l)))
            try:
                with torch.no_grad():
                    h=model.gpt_neox(**enc,use_cache=False).last_hidden_state;lp=logprob(model.get_output_embeddings()(h[ix,last]).float().cpu().numpy())
            finally:
                for handle in handles:handle.remove()
            forwards+=len(ids);return lp
        base=np.concatenate([forward(np.arange(off,min(off+bs,n)),capture=True) for off in range(0,n,bs)])
        np.savez_compressed(run/'raw_cache.npz',positions=positions,baseline_logprobs=base.astype(np.float32),**{f'layer{l}':v for l,v in captures.items()})
        def readout(lp):
            small=lp[labels];num=float(np.logaddexp(small[1],small[3])-np.logaddexp(small[0],small[2]));tense=float(np.logaddexp(small[2],small[3])-np.logaddexp(small[0],small[1]));return int(np.argmax(small)),num,tense
        for r,lp in zip(rows,base):
            label,num,tense=readout(lp);record(dict(kind='baseline',row_id=r['id'],block=r['block'],cue_id=r['cue_id'],template=r['template'],number=r['number'],past=r['past'],distractor=r['distractor'],label=label,expected=r['expected_label_index'],correct=label==r['expected_label_index'],number_logodds=num,past_logodds=tense,label_logprobs=lp[labels].tolist()))
        checks['noop_exact']=np.array_equal(forward(np.arange(bs),layer=cfg['layers'][0],delta=np.zeros_like(captures[cfg['layers'][0]][:bs])),base[:bs]);progress('RAW_CAPTURED',rows=n)
        donors={f:np.array([p[f] for p in pairs]) for f in ['number','time','joint']}
        specs=[('number_subject','number','number',None),('time_cue','time','time',None),('time_subject','time','number',None),('joint_separate_add','joint','both','add'),('joint_subject_add','joint','number','add'),('joint_subject_full','joint','number','full'),('joint_separate_full','joint','both','full')]
        if 'final' in names:specs.extend([('number_final','number','final',None),('time_final','time','final',None),('joint_final_add','joint','final','add'),('joint_final_full','joint','final','full')])
        if 'raw_operations' in cfg:specs=[s for s in specs if s[0] in cfg['raw_operations']]
        def measure(layer,name,factor,delta,**identity):
            for off in range(0,n,bs):
                ix=np.arange(off,min(off+bs,n));lp=forward(ix,layer=layer,delta=delta[ix])
                for k,j in enumerate(ix):
                    r=rows[j];d=int(donors[factor][j]);label,num,tense=readout(lp[k]);ref=base[d]
                    record(dict(kind='source_intervention' if identity else 'intervention',row_id=int(j),donor=d,block=r['block'],cue_id=r['cue_id'],template=r['template'],number=r['number'],past=r['past'],distractor=r['distractor'],layer=layer,method=name,factor=factor,expected=rows[d]['expected_label_index'],label=label,correct=label==rows[d]['expected_label_index'],number_logodds=num,past_logodds=tense,label_logprobs=lp[k,labels].tolist(),kl_to_full_donor=max(0.,float(np.sum(np.exp(ref)*(ref-lp[k])))),delta_norm=realized_delta_norm(delta[j],positions[j]),slot_delta_norm=float(np.linalg.norm(delta[j])),**identity))
        for layer,h in captures.items():
            change={f:h[ix]-h for f,ix in donors.items()}
            for name,factor,site,kind in specs:
                delta=np.zeros_like(h)
                if kind is None:slot=names.index(site);delta[:,slot]=change[factor][:,slot]
                elif site in ['number','final']:
                    slot=names.index(site);delta[:,slot]=change['joint'][:,slot] if kind=='full' else change['number'][:,slot]+change['time'][:,slot]
                else:
                    delta[:,0]=change['joint' if kind=='full' else 'time'][:,0];delta[:,1]=change['joint' if kind=='full' else 'number'][:,1]
                measure(layer,name,factor,delta)
                progress('OPERATION_COMPLETE',layer=layer,method=name)
        if cfg.get('sae_checkpoints'):
            from sparsify import SparseCoder
            from ccad.factor_correspondence import compact_source
            layer=cfg['sae_layer'];h=captures[layer];discovery=np.array([r['block'] in cfg['discovery_blocks'] and r['cue_id'] in cfg['discovery_cues'] for r in rows]);source_rows=[]
            checks['discovery_lexically_disjoint']=not ({noun for b in cfg['discovery_blocks'] for noun in cfg['noun_pairs'][b]}&{noun for b in range(len(cfg['noun_pairs'])) if b not in cfg['discovery_blocks'] for noun in cfg['noun_pairs'][b]})
            for spec in cfg['sae_checkpoints']:
                path=Path(spec['path']);checked(path/'sae.safetensors');checked(path/'cfg.json');sae=SparseCoder.load_from_disk(path,device='cuda').eval()
                with torch.no_grad():
                    acts,inds,_=sae.encode(torch.as_tensor(h[:,:2].reshape(-1,h.shape[-1]),device='cuda'));z=torch.zeros((len(acts),sae.num_latents),device='cuda').scatter_(1,inds,acts).cpu().numpy().reshape(n,2,-1);decoder=sae.W_dec.detach().cpu().numpy()
                del sae;np.savez_compressed(run/f"seed{spec['seed']}_codes.npz",time_z=z[:,0],number_z=z[:,1])
                if cfg.get('encode_only'):
                    progress('ENCODE_ONLY_COMPLETE',seed=spec['seed']);continue
                variants={name:np.zeros_like(h,dtype=np.float64) for name in ['source_native16','source_native32','source_full_sae','source_pca1']};arrays={}
                for factor,slot in [('number',1),('time',0)]:
                    dz=z[donors[factor],slot]-z[:,slot];sign=np.array([1 if rows[i]['number' if factor=='number' else 'past'] else -1 for i in donors[factor]])
                    source=compact_source(dz,decoder,discovery,sign,16);expanded=compact_source(dz,decoder,discovery,sign,32)
                    variants['source_native16'][:,slot]=source['coordinates']@source['basis'].T;variants['source_native32'][:,slot]=expanded['coordinates']@expanded['basis'].T;variants['source_full_sae'][:,slot]=dz@decoder
                    _,_,v=np.linalg.svd(source['coordinates'][discovery],full_matrices=False);variants['source_pca1'][:,slot]=source['coordinates']@v[:1].T@v[:1]@source['basis'].T
                    source_rows.append(dict(seed=spec['seed'],factor=factor,members=source['support'].tolist(),expanded_members=expanded['support'].tolist(),span_rank=source['rank'],process_singular_values=source['discovery_singular_values'].tolist(),span_error=source['span_error']))
                    for key in ['support','basis','coefficients','coordinates']:arrays[factor+'_'+key]=source[key]
                np.savez_compressed(run/f"seed{spec['seed']}_source_coordinates.npz",**arrays)
                for name,all_delta in variants.items():
                    for factor,slot in [('number',1),('time',0),('joint',None)]:
                        delta=all_delta.copy()
                        if slot is not None:delta[:,1-slot]=0
                        measure(layer,name,factor,delta,seed=spec['seed'])
                progress('SOURCE_COMPLETE',seed=spec['seed'])
            write(run/'source_components.json',dict(rows=source_rows,discovery_blocks=cfg['discovery_blocks'],discovery_cues=cfg['discovery_cues']))
        checks['all_rows']=len(metrics)==n*(1+len(cfg['layers'])*len(specs)+(0 if cfg.get('encode_only') else len(cfg.get('sae_checkpoints',[]))*12));checks['finite']=all(np.isfinite(v) for r in metrics for v in r.values() if isinstance(v,float));checks['unique']=len(metrics)==len({(r['kind'],r.get('layer'),r.get('seed'),r.get('method'),r.get('factor'),r['row_id']) for r in metrics})
        env=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),peak_vram_bytes=torch.cuda.max_memory_allocated(),cpu_threads=2)
    except Exception as exc:error=f'{type(exc).__name__}: {exc}';(run/'stderr.log').write_text(traceback.format_exc())
    status='PASS' if error is None and checks and all(checks.values()) else 'FAIL';summary=dict(status=status,error=error,checks=checks,rows=len(metrics),wall_seconds=time.perf_counter()-start,sequence_forwards=forwards,scope=cfg['scope'],metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path='scripts/run_composition_material.py',generator_script_sha256=sha256(run/'source_snapshot/scripts/run_composition_material.py'))
    write(run/'inputs.json',dict(inputs=inputs));write(run/'environment.json',env);write(run/'metrics.summary.json',summary);write(run/'status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat()));write(run/'stdout.log',summary);v=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=v.ok,errors=list(v.errors)));print(json.dumps(dict(summary=summary,contract_ok=v.ok,errors=list(v.errors))),flush=True);return 0 if status=='PASS' and v.ok else 1


if __name__=='__main__':raise SystemExit(main())
