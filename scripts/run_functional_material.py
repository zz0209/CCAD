"""Model competence and causally active material under controlled grammar changes.

Reads released CausalGym data without importing its code. The independently
authored crossed panel fixes lemma and output labels while toggling two factors.
"""
import argparse,gc,hashlib,json,os,platform,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                  HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from run_r011s1_raw_hook_asset import ROOT,entry,aggregate,write_json as write
from ccad.artifacts import sha256,validate_run_directory
from ccad.activation_contract import HookPointContract,extract_primary_hook_tensor,replace_primary_hook_tensor
import numpy as np


def crossed_panel(cfg):
    rows=[];lookup={}
    nouns=cfg['noun_pairs']
    for block,(singular,plural) in enumerate(nouns):
        other=(5*block+3)%len(nouns)
        if other==block:other=(other+1)%len(nouns)
        for template in cfg['templates']:
            for subject in range(2):
                for distractor in range(2):
                    a,b=nouns[block][subject],nouns[other][distractor]
                    if template=='pp':spans=['<|endoftext|>The',' '+a,' near the',' '+b]
                    elif template=='subject_relative':spans=['<|endoftext|>The',' '+a,' that praised the',' '+b]
                    elif template=='object_relative':spans=['<|endoftext|>The',' '+a,' that the',' '+b,' praised']
                    else:raise ValueError(template)
                    i=len(rows);lookup[block,template,subject,distractor]=i
                    rows.append(dict(id=i,block=block,template=template,subject_number=subject,
                        distractor_number=distractor,subject_lemma=nouns[block][0],distractor_lemma=nouns[other][0],
                        spans=spans,text=''.join(spans),subject_region=1,distractor_region=3))
    pairs=[]
    for row in rows:
        a=row['subject_number'];b=row['distractor_number'];block=row['block'];template=row['template']
        for factor in ['subject','distractor']:
            key=(block,template,1-a,b) if factor=='subject' else (block,template,a,1-b)
            pairs.append(dict(id=len(pairs),base=row['id'],donor=lookup[key],factor=factor,
                changed_region=row[factor+'_region'],block=block,template=template))
    return rows,pairs


def logprob(x):
    x=x.astype(np.float64);x=x-x.max(axis=1,keepdims=True)
    return x-np.log(np.exp(x).sum(axis=1,keepdims=True))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    cfg=json.loads(ap.parse_args().config.read_text(encoding='utf-8'))
    run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False)
    start=time.perf_counter();write(run/'config.resolved.json',cfg)
    code=[]
    for rel in ['scripts/run_functional_material.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/activation_contract.py','src/ccad/artifacts.py']:
        p=ROOT/rel;q=run/'source_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
        code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='functional.material.v1',run_id=cfg['run_id'],run_parent='SEVEN_R1',
        purpose=cfg['purpose'],milestone='material-selection-for-functional-composition',evidence_level='controlled_development',
        started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),
        code_snapshot_hash=aggregate(code),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,
        mean_constants_source_split='not_applicable_direct_raw_donor',threshold_source_split='fixed_before_outputs',
        statistics_unit='public reciprocal pairs; crossed lexical blocks and syntax templates',device='cuda:0',seeds=[],
        resource_lease='gpu-0 manager;one auxiliary CPU thread',resource_lease_reason=cfg['budget']))
    for f in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/f).touch()
    write(run/'status.json',dict(status='RUNNING'))
    inputs=[];metrics=[];checks={};env={};error=None;forwards=0;model_results=[]
    def record(value):
        serialized=json.dumps(value)
        with (run/'metrics.raw.jsonl').open('a',encoding='utf-8') as f:f.write(serialized+'\n')
        metrics.append(value)
    def progress(stage,**kw):
        v=dict(stage=stage,seconds=time.perf_counter()-start,sequence_forwards=forwards,**kw)
        write(run/'progress.json',v);print(json.dumps(v),flush=True)
    try:
        import torch,transformers
        from transformers import AutoTokenizer,AutoModelForCausalLM
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
        env=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
            transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),cpu_threads=1)
        dataset=Path(cfg['dataset_dir'])/(cfg['dataset_split']+'.json')
        inputs.extend([entry(dataset,'aryaman/causalgym '+cfg['dataset_revision'],'public development data',cfg['dataset_license']),
            entry(Path(cfg['dataset_dir'])/'README.md','official CausalGym dataset card','dataset documentation',cfg['dataset_license'])])
        public=json.loads(dataset.read_text());tasks=sorted(set(r['task'] for r in public))
        rows,pairs=crossed_panel(cfg)
        write(run/'crossed_inputs.json',dict(rows=rows,pairs=pairs,rule=cfg['crossed_rule'],label_pair=cfg['label_pair']))
        public_prompts=[]
        for i,row in enumerate(public):
            for side,other in [('base','src'),('src','base')]:
                public_prompts.append(dict(public_pair=i,side=side,task=row['task'],text=''.join(row[side]),
                    expected=row[side+'_label'],other=row[other+'_label'],type=row[side+'_type']))
        write(run/'public_inputs.json',dict(dataset=cfg['dataset_revision'],split=cfg['dataset_split'],rows=public_prompts))
        for spec in cfg['models']:
            model_started=time.perf_counter();name=spec['name'];path=Path(spec['path'])
            mcfg=json.loads((path/'config.json').read_text());dim=mcfg['hidden_size'];layers=spec['layers']
            for fn in ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']:
                inputs.append(entry(path/fn,spec['id']+' '+spec['revision'],'model/tokenizer metadata','Apache-2.0'))
            weight=path/'model.safetensors'
            if not weight.exists():weight=path/'pytorch_model.bin'
            # The weight digest is evaluated on the file actually loaded; no cache copy.
            weight_entry=entry(weight,spec['id']+' '+spec['revision'],'model weights','Apache-2.0');inputs.append(weight_entry)
            if spec.get('model_lfs_sha256'):
                assert weight_entry['sha256']==spec['model_lfs_sha256'] and weight_entry['bytes']==spec['model_expected_bytes']
            tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
            if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
            model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=False,
                dtype=torch.float32,attn_implementation='eager').to('cuda:0').eval()
            modules={l:model.get_submodule(f'gpt_neox.layers.{l}') for l in layers}
            contracts={l:HookPointContract(f'gpt_neox.layers.{l}',l,'resid_post',dim) for l in layers}

            def encode(texts):
                encoded=tokenizer(texts,add_special_tokens=False,padding=True,return_tensors='np')
                ids=encoded['input_ids'];mask=encoded['attention_mask'];last=mask.sum(axis=1)-1
                return ids,mask,last

            def forward(texts,capture_positions=None,patch=None):
                nonlocal forwards
                if time.perf_counter()-start>cfg['budget_seconds']:raise TimeoutError('Material compute budget exceeded')
                ids,mask,last=encode(texts);ix=torch.arange(len(texts),device='cuda:0');pos=torch.as_tensor(last,device='cuda:0')
                handles=[];captured={}
                for l in layers:
                    if capture_positions is None and (patch is None or patch[0]!=l):continue
                    def hook(module,args,output,l=l):
                        h=extract_primary_hook_tensor(output,contracts[l])
                        if capture_positions is not None:
                            p=torch.as_tensor(capture_positions,device='cuda:0')
                            captured[l]=h[ix[:,None],p].detach().cpu().numpy().copy()
                        if patch is not None and patch[0]==l:
                            h=h.clone();p=torch.as_tensor(patch[1],device='cuda:0')
                            h[ix,p]=torch.as_tensor(patch[2],device='cuda:0',dtype=h.dtype)
                            return replace_primary_hook_tensor(output,h,contracts[l])
                        return output
                    handles.append(modules[l].register_forward_hook(hook))
                try:
                    with torch.no_grad():
                        output=model.gpt_neox(torch.as_tensor(ids,device='cuda:0'),attention_mask=torch.as_tensor(mask,device='cuda:0'),use_cache=False).last_hidden_state
                        logits=model.get_output_embeddings()(output[ix,pos]);lp=logprob(logits.cpu().numpy())
                finally:
                    for h in handles:h.remove()
                forwards+=len(texts)
                return lp,captured

            bs=cfg['batch_size'];label_ids={}
            for label in sorted({p[k] for p in public_prompts for k in ['expected','other']}|set(cfg['label_pair'])):
                tok=tokenizer.encode(label,add_special_tokens=False)
                if len(tok)!=1:raise ValueError('non-single-token label '+repr(label))
                label_ids[label]=tok[0]
            for offset in range(0,len(public_prompts),bs):
                batch=public_prompts[offset:offset+bs];lp,_=forward([p['text'] for p in batch])
                for j,p in enumerate(batch):
                    a,b=label_ids[p['expected']],label_ids[p['other']]
                    record(dict(kind='public_baseline',model=name,**p,expected_logprob=float(lp[j,a]),other_logprob=float(lp[j,b]),
                        correct=bool(lp[j,a]>lp[j,b]),full_prediction=tokenizer.decode([int(lp[j].argmax())]),full_correct=bool(lp[j].argmax()==a)))
                if offset%(bs*40)==0:progress('public_baseline',model=name,completed=offset+len(batch),total=len(public_prompts))
            positions=[];token_rows=[]
            for row in rows:
                ids=tokenizer.encode(row['text'],add_special_tokens=False);ends=[];prefix=''
                for span in row['spans']:
                    prefix+=span;pids=tokenizer.encode(prefix,add_special_tokens=False)
                    if pids!=ids[:len(pids)]:raise ValueError('Region boundary changes tokenization')
                    ends.append(len(pids)-1)
                positions.append([ends[row['subject_region']],ends[row['distractor_region']],ends[-1]])
                for label in cfg['label_pair']:
                    if tokenizer.encode(row['text']+label,add_special_tokens=False)!=ids+[label_ids[label]]:
                        raise ValueError('Continuation tokenization is not concatenative')
                token_rows.append(dict(row_id=row['id'],tokens=ids,region_ends=ends,token_text=tokenizer.convert_ids_to_tokens(ids)))
            positions=np.array(positions);raw={l:np.empty((len(rows),3,dim),dtype=np.float32) for l in layers};baseline=[]
            for offset in range(0,len(rows),bs):
                batch=rows[offset:offset+bs];lp,cap=forward([p['text'] for p in batch],positions[offset:offset+len(batch)])
                baseline.append(lp)
                for l in layers:raw[l][offset:offset+len(batch)]=cap[l]
            baseline=np.concatenate(baseline);labels=[label_ids[s] for s in cfg['label_pair']]
            np.save(run/f'{name}_crossed_baseline_logprobs.npy',baseline.astype(np.float32))
            np.savez_compressed(run/f'{name}_crossed_raw.npz',positions=positions,**{f'layer{l}':raw[l] for l in layers})
            write(run/f'{name}_token_rows.json',dict(rows=token_rows,label_ids=labels,sites=['subject','distractor','last']))
            for i,row in enumerate(rows):
                margin=float(baseline[i,labels[1]]-baseline[i,labels[0]])
                record(dict(kind='crossed_baseline',model=name,row_id=i,block=row['block'],template=row['template'],
                    subject_number=row['subject_number'],distractor_number=row['distractor_number'],plural_margin=margin,
                    correct=bool((margin>0)==bool(row['subject_number'])),is_probability=float(np.exp(baseline[i,labels[0]])),
                    are_probability=float(np.exp(baseline[i,labels[1]])),full_prediction=tokenizer.decode([int(baseline[i].argmax())])))
            first=rows[:bs];texts=[x['text'] for x in first]
            reference,_=forward(texts)
            noop,_=forward(texts,patch=(layers[1],positions[:len(first),2],raw[layers[1]][:len(first),2]))
            checks[name+'_same_batch_noop_exact']=bool(np.array_equal(reference,noop))
            padding=[]
            for i in [0,int(np.argmax([len(x['tokens']) for x in token_rows]))]:
                single,_=forward([rows[i]['text']]);padding.append(float(np.max(np.abs(single[0]-baseline[i]))))
            checks[name+'_padding_bound']=max(padding)<cfg['numerical_padding_logprob_tolerance']
            for l in layers:
                for site in cfg['patch_sites']:
                    for offset in range(0,len(pairs),bs):
                        batch=pairs[offset:offset+bs];ri=np.array([p['base'] for p in batch]);di=np.array([p['donor'] for p in batch])
                        slot=np.array([0 if p['factor']=='subject' else 1 for p in batch]) if site=='changed_region_end' else np.full(len(batch),2)
                        patchdata=raw[l][di,slot];lp,_=forward([rows[i]['text'] for i in ri],patch=(l,positions[ri,slot],patchdata))
                        for j,p in enumerate(batch):
                            i=int(ri[j]);d=int(di[j]);original=baseline[i];full=baseline[d]
                            delta=float((lp[j,labels[1]]-lp[j,labels[0]])-(original[labels[1]]-original[labels[0]]))
                            fulldelta=float((full[labels[1]]-full[labels[0]])-(original[labels[1]]-original[labels[0]]))
                            orient=1 if rows[d]['subject_number'] else -1
                            kl=lambda a,b:max(0.,float(np.sum(np.exp(a)*(a-b))))
                            record(dict(kind='crossed_patch',model=name,pair_id=p['id'],block=p['block'],template=p['template'],
                                factor=p['factor'],layer=l,site=site,base=i,donor=d,plural_margin_change=delta,
                                donor_oriented_margin_change=orient*delta,full_donor_margin_change=fulldelta,
                                donor_label_correct=bool((lp[j,labels[1]]>lp[j,labels[0]])==bool(rows[d]['subject_number'])),
                                base_label_correct=bool((lp[j,labels[1]]>lp[j,labels[0]])==bool(rows[i]['subject_number'])),
                                kl_to_full_donor=kl(full,lp[j]),kl_to_base=kl(original,lp[j]),full_donor_kl_to_base=kl(full,original),
                                is_probability=float(np.exp(lp[j,labels[0]])),are_probability=float(np.exp(lp[j,labels[1]])),
                                patched_vector_norm=float(np.linalg.norm(patchdata[j]-raw[l][i,slot[j]]))))
                    progress('crossed_patches',model=name,layer=l,site=site)
            model_results.append(dict(model=name,layers=layers,hidden_size=dim,num_layers=mcfg['num_hidden_layers'],
                seconds=time.perf_counter()-model_started,padding_logprob_max_errors=padding,
                public_pairs=len(public),crossed_inputs=len(rows),crossed_pairs=len(pairs)))
            del model;gc.collect();torch.cuda.empty_cache()
        checks['all_public_rows']=sum(x['kind']=='public_baseline' for x in metrics)==len(cfg['models'])*2*len(public)
        checks['all_crossed_rows']=sum(x['kind']=='crossed_patch' for x in metrics)==sum(len(m['layers']) for m in cfg['models'])*len(cfg['patch_sites'])*len(pairs)
        checks['finite']=all(np.isfinite(v) for x in metrics for v in x.values() if isinstance(v,float))
        env=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
            transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),peak_vram_bytes=torch.cuda.max_memory_allocated(),cpu_threads=1)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';(run/'stderr.log').write_text(traceback.format_exc(),encoding='utf-8')
    status='PASS' if error is None and checks and all(checks.values()) else 'FAIL'
    summary=dict(status=status,error=error,checks=checks,models=model_results,rows=len(metrics),sequence_forwards=forwards,
        wall_seconds=time.perf_counter()-start,scope=cfg['scope'],metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),
        generator_script_path='scripts/run_functional_material.py',generator_script_sha256=sha256(run/'source_snapshot/scripts/run_functional_material.py'))
    write(run/'inputs.json',dict(inputs=inputs));write(run/'environment.json',env);write(run/'metrics.summary.json',summary)
    write(run/'stdout.log',summary);write(run/'status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat()))
    validation=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=validation.ok,errors=list(validation.errors)))
    print(json.dumps(dict(summary=summary,contract_ok=validation.ok,errors=list(validation.errors))),flush=True)
    return 0 if status=='PASS' and validation.ok else 1


if __name__=='__main__':raise SystemExit(main())
