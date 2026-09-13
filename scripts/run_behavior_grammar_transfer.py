"""Transfer source-selected native behavioral groups on original BLiMP pairs.

All source selection uses the first64 pairs of three fixed paradigms. Target
operations and the remaining original pairs are evaluated after selection is
saved. Dictionaries and correspondence coefficients are frozen natural fits.
"""
from __future__ import annotations
import os,sys,json,time,platform,argparse,traceback
from pathlib import Path
from datetime import datetime,timezone
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from run_causalgym_multisite import MultisiteWork,ROOT,write
from ccad.artifacts import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);args=p.parse_args();cfg=json.loads(args.config.read_text())
    w=MultisiteWork(cfg,args.config,['scripts/run_behavior_grammar_transfer.py','scripts/run_causalgym_multisite.py','scripts/run_causalgym_native_transfer.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py','src/ccad/activation_contract.py'])
    error=None
    def checked(path):return w.checked(ROOT/Path(path))
    def log(stage,**kw):
        w.progress(stage,**kw)
        with (w.run/'events.jsonl').open('a') as f:f.write(json.dumps(dict(stage=stage,written_at_utc=datetime.now(timezone.utc).isoformat(),**kw))+'\n')
    def budget():
        if time.perf_counter()-w.wall_start>cfg['budget_seconds']:raise TimeoutError('Declared grammar-transfer budget exceeded')
    try:
        import numpy as np,torch,transformers
        w.torch=torch;w.device=torch.device('cuda:0');torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high')
        base=ROOT/cfg['group_run'];gc=json.loads(checked(base/'config.resolved.json').read_text());ref=ROOT/gc['reference_run'];rc=json.loads(checked(ref/'config.resolved.json').read_text());tr=ROOT/rc['training_run'];tc=json.loads(checked(tr/'config.resolved.json').read_text())
        assert json.loads(checked(base/'status.json').read_text())['status']=='PASS'
        sys.path.append(tc['dictionary_source_dir']);sys.path.append(tc['dictionary_overlay_dir'])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
        for f in ['dictionary_learning/trainers/top_k.py','dictionary_learning/trainers/matryoshka_batch_top_k.py','LICENSE']:checked(Path(tc['dictionary_source_dir'])/f)
        for f in ['config.json','model.safetensors','tokenizer.json']:checked(Path(tc['model_local_dir'])/f)
        tokenizer=transformers.AutoTokenizer.from_pretrained(tc['model_local_dir'],local_files_only=True)
        model=transformers.AutoModelForCausalLM.from_pretrained(tc['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to(w.device);model.requires_grad_(False);model.config.use_cache=False
        module=model.get_submodule(tc['hook_module_path']);saes={};D={}
        snapshots=json.loads(checked(tr/'checkpoints.json').read_text())['checkpoints']
        for obj in cfg['objectives']:
            for seed in cfg['seeds']:
                snap=next(s for s in snapshots if s['objective']==obj and s['seed']==seed and s['step']==rc['checkpoint_step']);sp=checked(snap['path']);assert sha256(sp)==snap['sha256']
                state=torch.load(sp,map_location=w.device,weights_only=True)
                ae=AutoEncoderTopK(1024,tc['dict_size'],tc['k']) if obj=='topk' else MatryoshkaBatchTopKSAE(1024,tc['dict_size'],tc['k'],state['group_sizes'].cpu().tolist())
                ae=ae.to(w.device);ae.load_state_dict(state);ae.eval();ae.requires_grad_(False);saes[obj,seed]=ae;D[obj,seed]=ae.decoder.weight.T if obj=='topk' else ae.W_dec
        queries=json.loads(checked(base/'query_results.json').read_text())['queries'];groups={}
        for q in queries:
            with np.load(checked(base/(q['query']+'_groups.npz'))) as a:groups[q['query']]={k:a[k] for k in a.files}
        source_manifest=json.loads(checked(cfg['data_manifest']).read_text());panel=[]
        if cfg.get('official_implementation_manifest'):checked(cfg['official_implementation_manifest'])
        for task in cfg['tasks']:
            fi=next(v for v in source_manifest['files'] if Path(v['path']).name==task+'.jsonl');path=checked(fi['path']);assert sha256(path)==fi['sha256']
            rr=[json.loads(s) for s in path.read_text().splitlines()];assert len(rr)==1000
            for row in rr:
                assert row['UID']==task and row['one_prefix_method']
                good=tokenizer.encode(row['sentence_good']);bad=tokenizer.encode(row['sentence_bad']);prefix=tokenizer.encode(row['one_prefix_prefix'])
                if cfg.get('official_boundary_tokens',False):
                    good=[tokenizer.eos_token_id]+good+[tokenizer.eos_token_id]
                    bad=[tokenizer.eos_token_id]+bad+[tokenizer.eos_token_id]
                    prefix=[tokenizer.eos_token_id]+prefix
                assert prefix and good[:len(prefix)]==prefix and bad[:len(prefix)]==prefix
                assert max(len(good),len(bad))<=cfg['max_length'] and len(prefix)<min(len(good),len(bad))
                panel.append(dict(task=task,row_id=int(row['pairID']),good=good,bad=bad,position=len(prefix)-1,
                    sentence_good=row['sentence_good'],sentence_bad=row['sentence_bad'],split='source_selection' if int(row['pairID'])<cfg['selection_pairs'] else 'held'))
        write(w.run/'panel.json',dict(rows=panel,official_boundary_tokens=cfg.get('official_boundary_tokens',False),scope='Original good/bad sentences, pairIDs and explicit original prefix; no truncation, filtering or reciprocal augmentation. When configured, official EOS boundaries are included in the score. All rows parsed/tokenized before inference. Only selection rows enter source selection.'))
        manifest=json.loads((w.run/'manifest.json').read_text());manifest.update(schema_version='behavior.grammar.transfer.v1',mean_constants_source_split='Absolute native code deletions; no mean subtraction in these interventions',statistics_unit='Original lexical pairs and shared SAE seed identities within three fixed paradigms',candidate_family_frozen=True);write(w.run/'manifest.json',manifest)
        w.environment=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),threads=2,model=tc['model_id'],hook=tc['hook_module_path'],max_length=cfg['max_length'],source_grammar_forwards_before_freeze=0,target_grammar_forwards_before_freeze=0)
        max_hidden=0.;forwards_by_phase={}
        def forward(records,delta=None,phase='clean'):
            nonlocal max_hidden
            actual=len(records);padded=records+[records[0]]*(cfg['batch_pairs']-actual)
            ids=torch.full((len(padded)*2,cfg['max_length']),tokenizer.eos_token_id,dtype=torch.long,device=w.device);length=[];positions=[]
            for i,r in enumerate(padded):
                for j,k in enumerate(['good','bad']):
                    ids[2*i+j,:len(r[k])]=torch.tensor(r[k],device=w.device);length.append(len(r[k]));positions.append(r['position'])
            pos=torch.tensor(positions,device=w.device);idx=torch.arange(len(ids),device=w.device);cache={}
            if delta is not None:
                if actual<len(padded):delta=torch.cat([delta,torch.zeros((len(padded)-actual,1024),device=w.device)])
                delta=delta.repeat_interleave(2,dim=0)
            def hook(m,i,out):
                h=out[0] if isinstance(out,tuple) else out;cache['h']=h[idx,pos].detach()
                if delta is None:return out
                hh=h.clone();hh[idx,pos]-=delta
                return (hh,)+out[1:] if isinstance(out,tuple) else hh
            handle=module.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    logits=model(ids,use_cache=False).logits
                    per=logits[:,:-1].log_softmax(-1).gather(-1,ids[:,1:,None]).squeeze(-1)
                    mask=torch.arange(cfg['max_length']-1,device=w.device)[None,:]<(torch.tensor(length,device=w.device)-1)[:,None]
                    total=(per*mask).double().sum(1).reshape(-1,2)
                    lp=logits[idx,pos].double().log_softmax(-1).reshape(-1,2,logits.shape[-1])[:,0]
            finally:handle.remove()
            hidden=cache['h'].reshape(-1,2,1024);err=float((hidden[:,0]-hidden[:,1]).abs().max());max_hidden=max(max_hidden,err);assert err<cfg['hidden_atol'],err
            w.sequence_forwards+=len(ids);w.token_forwards+=int(ids.numel());forwards_by_phase[phase]=forwards_by_phase.get(phase,0)+len(ids)
            return total[:actual,0]-total[:actual,1],hidden[:actual,0],lp[:actual]
        def capture(records):
            hs=[];clean=[]
            for i in range(0,len(records),cfg['batch_pairs']):
                a,h,_=forward(records[i:i+cfg['batch_pairs']]);hs.append(h);clean.append(a)
            h=torch.cat(hs);z={}
            with torch.no_grad():
                for key,ae in saes.items():z[key]=torch.cat([ae.encode(h[i:i+512]) for i in range(0,len(h),512)])
            return dict(rows=records,hidden=h,clean=torch.cat(clean),codes=z)
        def source_delta(q,data,ids):
            g=groups[q['query']];z=data['codes'][q['objective'],q['source_seed']];ix=torch.as_tensor(g['source_members'],device=w.device);gate=torch.as_tensor(g['source_gate'],device=w.device,dtype=z.dtype)
            return (z[ids][:,ix]*gate)@D[q['objective'],q['source_seed']][ix]
        dev=capture([r for r in panel if r['split']=='source_selection']);source_stats=[]
        for q in queries:
            all_delta=source_delta(q,dev,torch.arange(len(dev['rows']),device=w.device));margins=[]
            for i in range(0,len(dev['rows']),cfg['batch_pairs']):
                local=dev['rows'][i:i+cfg['batch_pairs']];m,_,_=forward(local,all_delta[i:i+len(local)],phase='source_selection');margins.extend(m.cpu().tolist())
                for j,r in enumerate(local):
                    w.record(kind='source_selection',task=r['task'],row_id=r['row_id'],component=r['task']+':'+str(r['row_id']),mode=q['query'],operation='delete',method='source_group',seed=q['source_seed'],target_seed=q['target_seed'],objective=q['objective'],split='source_selection',margin=float(m[j]),clean_margin=float(dev['clean'][i+j]))
            for task in cfg['tasks']:
                at=[i for i,r in enumerate(dev['rows']) if r['task']==task]
                source_stats.append(dict(query=q['query'],objective=q['objective'],source_seed=q['source_seed'],target_seed=q['target_seed'],component_id=q['component_id'],task=task,
                    mean_margin_decrement=float(np.mean([float(dev['clean'][i])-margins[i] for i in at])),clean_accuracy=float(np.mean([float(dev['clean'][i])>0 for i in at])),source_accuracy=float(np.mean([margins[i]>0 for i in at]))))
            log('SOURCE_GROUP_EVALUATED',query=q['query']);budget()
        selected=[]
        for obj in cfg['objectives']:
            for seed in cfg['seeds']:
                for task in cfg['tasks']:
                    candidates=[r for r in source_stats if r['objective']==obj and r['source_seed']==seed and r['task']==task]
                    for r in candidates:
                        controls=[c['mean_margin_decrement'] for c in source_stats if c['query']==r['query'] and c['task']!=task]
                        r['source_selectivity_score']=r['mean_margin_decrement']-float(np.mean(controls))
                    best=max(candidates,key=lambda r:(r['source_selectivity_score'],-r['component_id']));selected.append(dict(best,selection_task=task))
        write(w.run/'source_selection.json',dict(all_candidates=source_stats,selected=selected,rule=cfg['source_rule']))
        (w.run/'source_selection.raw.jsonl').write_bytes((w.run/'metrics.raw.jsonl').read_bytes())
        freeze=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),config_sha256=sha256(w.run/'config.resolved.json'),source_selection_sha256=sha256(w.run/'source_selection.json'),source_raw_sha256=sha256(w.run/'source_selection.raw.jsonl'),code_sha256=sha256(w.run/'code_hashes.json'),selected=selected,held_model_outcomes_consumed=0,target_interventions_consumed=0)
        write(w.run/'SOURCE_SELECTION_FREEZE.json',freeze);log('SOURCE_SELECTION_FROZEN',selected=len(selected),freeze_sha256=sha256(w.run/'SOURCE_SELECTION_FREEZE.json'))
        w.environment['source_grammar_forwards_before_freeze']=forwards_by_phase['source_selection'];w.checks['source_selection_before_target_and_held']=True
        del dev
        held=capture([r for r in panel if r['split']=='held']);lookup={q['query']:q for q in queries};raw_readers={}
        def readers(q):
            key=q['query']
            if key not in raw_readers:
                g=groups[key]
                with np.load(checked(ref/f"natural_discovery_{q['objective']}_seed{q['target_seed']}.npz")) as a:
                    z=torch.zeros(tuple(a['shape']),device=w.device);z[torch.as_tensor(a['rows'].astype('int64'),device=w.device),torch.as_tensor(a['columns'].astype('int64'),device=w.device)]=torch.as_tensor(a['values'],device=w.device)
                with np.load(checked(ref/'natural_discovery_states.npz')) as a:keep=(a['packed_positions']%128>=gc['behavior']['minimum_context_position'])&(a['packed_positions']%128<127)
                # Exactly the same rows as the parent fit, including its full-
                # producer-batch restriction, are saved with the NMF profiles.
                with np.load(checked(base/f"{q['objective']}_s{q['source_seed']}_behavior_nmf.npz")) as a:fit_rows=a['discovery_indices']
                X=z[fit_rows][:,g['target_members']].double();A=X.T@X/len(X);A+=gc['behavior']['raw_ridge_fraction']*A.diag().mean()*torch.eye(A.shape[0],device=w.device,dtype=torch.float64)
                B=torch.as_tensor(g['extra_raw_reader_target_pool_ridge'],device=w.device,dtype=torch.float64)
                C=torch.linalg.cholesky(A).T@B;_,_,Vh=torch.linalg.svd(C,full_matrices=False)
                raw_readers[key]={'raw_full':B.float(),'raw_rank1':(B@Vh[:1].T@Vh[:1]).float(),'raw_rank2':(B@Vh[:2].T@Vh[:2]).float()}
                np.savez_compressed(w.run/(key+'_raw_readers.npz'),**{k:v.cpu().numpy() for k,v in raw_readers[key].items()})
            return raw_readers[key]
        for selection in selected:
            q=lookup[selection['query']];g=groups[q['query']];obj=q['objective'];s=q['source_seed'];t=q['target_seed'];target_codes=held['codes'][obj,t];tp=torch.as_tensor(g['target_members'],device=w.device)
            rr=readers(q);indices=[i for i,r in enumerate(held['rows']) if r['task']==selection['task'] or r['row_id']<cfg['selection_pairs']+cfg['control_pairs']]
            index=torch.tensor(indices,device=w.device);src=source_delta(q,held,index);zt=target_codes[index][:,tp];dt=D[obj,t][tp]
            deltas=dict(source_group=src,group64=(zt*torch.as_tensor(g['extra_gate_same_budget_group'],device=w.device,dtype=zt.dtype))@dt,
                assignment64=(zt*torch.as_tensor(g['extra_gate_assignment_joint_refit'],device=w.device,dtype=zt.dtype))@dt,
                assignment_one_scale=(zt*torch.as_tensor(g['extra_gate_assignment_one_scale'],device=w.device,dtype=zt.dtype))@dt,
                **{name:zt@B for name,B in rr.items()})
            for off in range(0,len(indices),cfg['batch_pairs']):
                loc=indices[off:off+cfg['batch_pairs']];records=[held['rows'][i] for i in loc];clean,hh,cleanlp=forward(records);replay=float((hh-held['hidden'][loc]).abs().max());assert replay<cfg['hidden_atol'],replay
                outputs={name:forward(records,delta[off:off+len(loc)],phase='held_'+name) for name,delta in deltas.items()}
                sm,_,slp=outputs['source_group'];effects=(slp.exp()*(slp-cleanlp)).sum(1)
                for method,(margin,_,lp) in outputs.items():
                    divergence=(slp.exp()*(slp-lp)).sum(1)
                    for j,r in enumerate(records):
                        w.record(kind='frozen_transfer',task=r['task'],row_id=r['row_id'],component=r['task']+':'+str(r['row_id']),mode=selection['task']+'|'+q['query'],operation='delete',method=method,seed=s,target_seed=t,objective=obj,selection_task=selection['task'],query=q['query'],split='held_positive' if r['task']==selection['task'] else 'held_control',
                            margin=float(margin[j]),clean_margin=float(clean[j]),source_margin=float(sm[j]),margin_error=abs(float(margin[j]-sm[j])),source_kl=float(divergence[j]),source_effect_kl=float(effects[j]),accuracy=bool(margin[j]>0),source_accuracy=bool(sm[j]>0),clean_accuracy=bool(clean[j]>0),edit_norm=float(deltas[method][off+j].norm()))
                budget()
            log('FROZEN_TRANSFER_COMPLETE',query=q['query'],selected_for=selection['task'],positive_pairs=1000-cfg['selection_pairs'],control_pairs=2*cfg['control_pairs'])
        # These hypotheses come from source-only natural examples in the
        # preceding round. Their identities and all authored counterfactuals
        # are fixed in the input manifest before this run starts.
        if cfg.get("run_controlled_probes",True):
            probe_spec=json.loads(checked(cfg['probe_manifest']).read_text());probe_rows=[]
            for i,r in enumerate(probe_spec['rows']):
                tokens=tokenizer.encode(r['prefix']);assert 16<=len(tokens)<cfg['max_length']
                label_ids=[]
                for token in r['response_tokens']:
                    ti=tokenizer.encode(token);assert len(ti)==1,(token,ti);label_ids+=ti
                probe_rows.append(dict(r,task='probe_'+r['category'],row_id=i,good=tokens+[tokenizer.eos_token_id],bad=tokens+[tokenizer.eos_token_id],position=len(tokens)-1,response_ids=label_ids))
            probe=capture(probe_rows);write(w.run/'probe_panel.json',dict(rows=probe_rows,hypotheses=probe_spec['hypotheses']))
            for query in probe_spec['queries']:
                q=lookup[query];g=groups[query];obj=q['objective'];s=q['source_seed'];t=q['target_seed'];idx=torch.arange(len(probe_rows),device=w.device)
                src=source_delta(q,probe,idx);tp=torch.as_tensor(g['target_members'],device=w.device);zt=probe['codes'][obj,t][:,tp];dt=D[obj,t][tp];rr=readers(q)
                deltas=dict(source_group=src,group64=(zt*torch.as_tensor(g['extra_gate_same_budget_group'],device=w.device,dtype=zt.dtype))@dt,
                    assignment64=(zt*torch.as_tensor(g['extra_gate_assignment_joint_refit'],device=w.device,dtype=zt.dtype))@dt,**{name:zt@B for name,B in rr.items()})
                for off in range(0,len(probe_rows),cfg['batch_pairs']):
                    records=probe_rows[off:off+cfg['batch_pairs']];_,_,cleanlp=forward(records);outputs={name:forward(records,delta[off:off+len(records)],phase='controlled_probe')[2] for name,delta in deltas.items()};slp=outputs['source_group']
                    for method,lp in outputs.items():
                        for j,r in enumerate(records):
                            lab=r['response_ids'];w.record(kind='controlled_probe',task=r['task'],row_id=r['row_id'],component=r['intro_id']+':'+r['stem'],mode=query,operation='delete_probe',method=method,seed=s,target_seed=t,objective=obj,query=query,split='new_controlled_context',category=r['category'],intro_id=r['intro_id'],stem=r['stem'],
                                response_probability=float(lp[j,lab].exp().sum()),clean_response_probability=float(cleanlp[j,lab].exp().sum()),source_response_probability=float(slp[j,lab].exp().sum()),source_kl=float((slp[j].exp()*(slp[j]-lp[j])).sum()),source_effect_kl=float((slp[j].exp()*(slp[j]-cleanlp[j])).sum()),edit_norm=float(deltas[method][off+j].norm()))
                    budget()
                log('CONTROLLED_PROBE_COMPLETE',query=query,contexts=len(probe_rows))
        w.checks['prefix_hidden_equal_and_replayed']=max_hidden<cfg['hidden_atol'];w.environment.update(maximum_prefix_hidden_error=max_hidden,forwards_by_phase=forwards_by_phase)
    except BaseException:error=traceback.format_exc()
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
