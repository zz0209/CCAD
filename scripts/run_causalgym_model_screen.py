"""Run the original CausalGym pair/token/odds protocol through model adapters.

The public reference Batch is loaded from a pinned, privately retained source.
No benchmark examples, labels, region alignments or score definitions are changed.
This development screen is not the full benchmark or final confirmation.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, platform, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                  OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from run_r011s1_raw_hook_asset import ROOT, entry, aggregate, write_json as write
from ccad.artifacts import sha256, validate_run_directory


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False)
    started=datetime.now(timezone.utc).isoformat();timer=time.perf_counter();cpu_start=time.process_time()
    write(run/'config.resolved.json',cfg)
    codes=[]
    for rel in ['scripts/run_causalgym_model_screen.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py']:
        p=ROOT/rel;dst=run/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
        codes.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json',dict(files=codes,aggregate_sha256=aggregate(codes),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='causalgym.model.screen.v1',run_id=cfg['run_id'],run_parent='FINAL_THREE_R19',
        purpose=cfg['purpose'],milestone='M4',evidence_level='official_train_protocol_development',started_utc=started,
        project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(codes),source_snapshot_required=True,
        audit_opened=False,candidate_family_frozen=False,mean_constants_source_split='not used',threshold_source_split='no outcome filtering',
        statistics_unit='task and reciprocal pair; model family descriptive',device='cuda:0',seeds=[],
        resource_lease='gpu-0 via resource_manager.run',resource_lease_reason='Two-model full forward/backward development screen'))
    for name in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/name).touch()
    write(run/'status.json',dict(status='RUNNING',updated_utc=started))
    inputs=[];env={};records=[];checks={};error=None;array_outputs=[]
    def log(event,**values):
        line=json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),event=event,**values))
        with (run/'stdout.log').open('a') as f:f.write(line+'\n')
        print(line,flush=True)
    try:
        import numpy as np
        import torch
        import transformers
        from transformers import AutoTokenizer,AutoModelForCausalLM
        torch.set_num_threads(2);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True)
        env=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,
                 gpu=torch.cuda.get_device_name(),matmul_precision=torch.get_float32_matmul_precision(),threads=2)
        source=ROOT/cfg['official_data_module'];inputs.append(entry(source,'CausalGym pinned original code','reference_batch','private source study'))
        spec=importlib.util.spec_from_file_location('ccad_causalgym_reference_data',source);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
        path=Path(cfg['train_json']);inputs.append(entry(path,'aryaman/causalgym '+cfg['dataset_revision'],'original_train','MIT'))
        data=json.loads(path.read_text());all_tasks=sorted(set(r['task'] for r in data));checks['official_task_count']=len(all_tasks)
        if len(all_tasks)!=29:raise ValueError(f'Expected original 29 tasks, got {len(all_tasks)}')
        selected=[dict(row,original_index=i) for task in cfg['tasks'] for i,row in [(i,r) for i,r in enumerate(data) if r['task']==task][:cfg['rows_per_task']]]
        write(run/'panel.json',dict(rows=selected,all_official_tasks=all_tasks,selection='first original rows of prespecified three development tasks',test_read=False))
        for mc in cfg['models']:
            tag=mc['name'];directory=Path(mc['path']);torch.cuda.reset_peak_memory_stats();tmodel=time.perf_counter()
            for name in ['config.json','tokenizer.json','model.safetensors']:
                inputs.append(entry(directory/name,mc['model_id']+' '+mc['revision'],tag+'_'+name,mc['license']))
            tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
            model=AutoModelForCausalLM.from_pretrained(directory,local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to('cuda:0')
            model.requires_grad_(False);model.config.use_cache=False
            modules={layer:model.get_submodule(mc['module_prefix']+str(layer)) for layer in mc['layers']}
            dim=model.config.hidden_size;counter={'sequences':0,'tokens':0};cache={}
            def full(batch,delta=None,edit_layer=None,edit_positions=None,gradient=False,capture=False):
                hooks=[];captured={}
                for layer,module in modules.items():
                    def hook(mod,inp,out,layer=layer):
                        value=out[0] if isinstance(out,(tuple,list)) else out
                        if gradient and layer==min(modules):value=value.detach().requires_grad_(True)
                        if layer==edit_layer:
                            value=value.clone();value[torch.arange(len(edit_positions),device=value.device),edit_positions]+=delta
                        if capture or gradient:captured[layer]=value
                        if isinstance(out,tuple):return (value,)+out[1:]
                        if isinstance(out,list):return [value]+out[1:]
                        return value
                    hooks.append(module.register_forward_hook(hook))
                try:
                    with torch.set_grad_enabled(gradient):
                        out=model(**batch,use_cache=False,output_hidden_states=capture and not gradient)
                        ids=torch.arange(len(batch['input_ids']),device='cuda:0');last=batch['attention_mask'].sum(1)-1
                        logits=out.logits[ids,last]
                    counter['sequences']+=len(ids);counter['tokens']+=int(batch['attention_mask'].sum())
                    return logits,captured,out.hidden_states if capture and not gradient else None
                finally:
                    for h in hooks:h.remove()
            for offset in range(0,len(selected),cfg['batch_size']):
                rows=selected[offset:offset+cfg['batch_size']]
                pairs=[ref.Pair(r['base'],r['src'],r['base_type'],r['src_type'],r['base_label'],r['src_label']) for r in rows]
                batch=ref.Batch(pairs,tokenizer,'cuda:0');align=batch.compute_pos('last')
                changed=[max(j for j,(a,b) in enumerate(zip(r['base'],r['src'])) if a!=b) for r in rows]
                regions={'changed':changed,'last': [len(r['base'])-1 for r in rows]}
                for i,r in enumerate(rows):
                    if len(tokenizer(''.join(r['base']))['input_ids'])!=sum(len(s) for s in batch.alignment_base[i]):
                        raise ValueError('Original region token alignment disagrees with full GPT tokenization')
                with torch.no_grad():
                    base_logits,bh,oracle=full(batch.base,capture=True)
                    donor_logits,dh,_=full(batch.src,capture=True)
                for layer in modules:
                    discrepancy=float((bh[layer]-oracle[layer+1]).abs().max())
                    checks[tag+'_hook_'+str(layer)]=max(checks.get(tag+'_hook_'+str(layer),0.),discrepancy)
                    if discrepancy!=0:raise ValueError(f'Wrong block hook {tag}/{layer}: {discrepancy}')
                ii=torch.arange(len(rows),device='cuda:0');base_margin=base_logits[ii,batch.src_labels]-base_logits[ii,batch.base_labels]
                grad_logits,gh,_=full(batch.base,gradient=True)
                checks[tag+'_gradient_forward_max_difference']=max(checks.get(tag+'_gradient_forward_max_difference',0.),float((grad_logits.detach()-base_logits).abs().max()))
                margin=grad_logits[ii,batch.src_labels]-grad_logits[ii,batch.base_labels]
                grads=torch.autograd.grad(margin.sum(),list(gh.values()))
                grads=dict(zip(gh,grads))
                payload={}
                for layer in modules:
                    for position,which in regions.items():
                        bp=torch.tensor([align[1][i][j][0] for i,j in enumerate(which)],device='cuda:0')
                        sp=torch.tensor([align[0][i][j][0] for i,j in enumerate(which)],device='cuda:0')
                        if min(int(bp.min()),int(sp.min()))<0:raise ValueError('Null intervention region in screen')
                        delta=dh[layer][ii,sp]-bh[layer][ii,bp]
                        gradient=grads[layer][ii,bp]
                        predicted=(gradient*delta).sum(1)
                        lp,_,_=full(batch.base,delta=delta,edit_layer=layer,edit_positions=bp)
                        actual=lp[ii,batch.src_labels]-lp[ii,batch.base_labels]-base_margin
                        for i,r in enumerate(rows):
                            result=dict(model=tag,task=r['task'],row=r['original_index'],layer=layer,position=position,
                                base_correct=bool(base_margin[i]<0),donor_correct=bool(donor_logits[i,batch.src_labels[i]]>donor_logits[i,batch.base_labels[i]]),
                                iia=bool(lp[i,batch.src_labels[i]]>lp[i,batch.base_labels[i]]),
                                base_margin=float(base_margin[i]),log_odds_ratio=float(actual[i]),linear_prediction=float(predicted[i]),
                                delta_norm=float(delta[i].norm()),gradient_norm=float(gradient[i].norm()))
                            records.append(result)
                            with (run/'metrics.raw.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
                        key=f'l{layer}_{position}'
                        for k,v in [('base',bh[layer][ii,bp]),('donor',dh[layer][ii,sp]),('gradient',gradient)]:payload[key+'_'+k]=v.detach().cpu().numpy()
                np.savez_compressed(run/f'{tag}_batch_{offset}.npz',row_ids=np.array([r['original_index'] for r in rows]),**payload)
                del bh,dh,gh,grads,oracle,grad_logits,margin,payload
                if offset==0 or offset+cfg['batch_size']>=len(selected):log('MODEL_SCREEN_PROGRESS',model=tag,rows=min(offset+cfg['batch_size'],len(selected)),total=len(selected))
            log('MODEL_SCREEN_COMPLETE',model=tag,wall_seconds=time.perf_counter()-tmodel,peak_cuda_bytes=torch.cuda.max_memory_allocated(),**counter)
            env[tag]=dict(parameters=model.num_parameters(),hidden_size=dim,layers=model.config.num_hidden_layers,
                          peak_cuda_bytes=torch.cuda.max_memory_allocated(),wall_seconds=time.perf_counter()-tmodel,**counter)
            del model,tokenizer,modules,cache;import gc;gc.collect();torch.cuda.empty_cache()
        summary=[]
        for model in cfg['models']:
            for task in cfg['tasks']:
                for layer in model['layers']:
                    for position in ['changed','last']:
                        rows=[r for r in records if (r['model'],r['task'],r['layer'],r['position'])==(model['name'],task,layer,position)]
                        summary.append(dict(model=model['name'],task=task,layer=layer,position=position,n=len(rows),
                            base_accuracy=float(np.mean([r['base_correct'] for r in rows])),iia=float(np.mean([r['iia'] for r in rows])),
                            log_odds_ratio=float(np.mean([r['log_odds_ratio'] for r in rows])),
                            linear_mae=float(np.mean([abs(r['linear_prediction']-r['log_odds_ratio']) for r in rows]))))
        write(run/'screen_summary.json',dict(rows=summary,checks=checks))
    except BaseException as exc:
        error=repr(exc);(run/'stderr.log').write_text(traceback.format_exc());log('FAIL',error=error)
    finally:
        status='PASS' if error is None else 'FAIL';write(run/'inputs.json',dict(inputs=inputs));write(run/'environment.json',env)
        write(run/'metrics.summary.json',dict(run_status=status,metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path='scripts/run_causalgym_model_screen.py',generator_script_sha256=codes[0]['sha256'],metrics={'rows':len(records),'wall_seconds':time.perf_counter()-timer,'process_cpu_seconds':time.process_time()-cpu_start},checks=checks,error=error,
            scope='Prespecified three original train tasks; exact original Batch, last-region alignment, first-token labels and log odds. New full benchmark/test not run.'))
        write(run/'status.json',dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),error=error))
        validation=validate_run_directory(run);write(run/'artifact_validation.json',dict(ok=validation.ok,errors=validation.errors))
    return 0 if error is None and validation.ok else 1

if __name__=='__main__':raise SystemExit(main())
