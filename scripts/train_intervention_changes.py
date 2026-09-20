from pathlib import Path
import argparse, hashlib, io, json, sys, time, traceback, zipfile
import numpy as np
import torch
import transformers
from run_causalgym_multisite import MultisiteWork, write
from run_shift_explanation import source_groups
from train_shift_dictionaries import site_module
from ccad.intervention_transport import transport_delta, refine_columns, pursuit_columns
from adaptive_native_execution import realize


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args(); c=json.loads(args.config.read_text())
    sys.path.extend([c['dictionary_source_dir'],c['dictionary_overlay_dir']])
    from dictionary_learning.trainers.top_k import AutoEncoderTopK
    files=['scripts/train_intervention_changes.py','scripts/run_causalgym_multisite.py',
           'scripts/run_r011s1_raw_hook_asset.py','scripts/run_shift_explanation.py',
           'scripts/train_shift_dictionaries.py','src/ccad/intervention_transport.py','src/ccad/artifacts.py',
           'scripts/adaptive_native_execution.py']
    w=MultisiteWork(c,args.config,files); handles=[]; error=None
    try:
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest'); torch.cuda.set_device(c['device'])
        torch.cuda.reset_peak_memory_stats(); w.torch=torch; w.device=torch.device(c['device'])
        w.checked(Path(c['dictionary_source_dir'])/'dictionary_learning/trainers/top_k.py','Pinned TopK','MIT')
        w.environment=dict(python=sys.executable,torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__)
        source=json.loads(w.checked(c['source_manifest'],'Published human decisions','MIT').read_text())
        groups,_=source_groups(w.checked(c['notebook'],'Published grouping','MIT'),source['members'])
        sites=list(source['members']); bank=np.load(w.checked(c['source_parameters']))
        human={s:{k:torch.tensor(bank[s+'__'+k],device=w.device) for k in ['encoder','encoder_bias','decoder','center']} for s in sites}
        gb=np.load(w.checked(c['grammar_source_parameters']))
        grammar={k:torch.tensor(gb[k],device=w.device) for k in ['encoder','encoder_bias','decoder','center']}
        initial={}; targets={}
        for s in sites:
            initial[s]=torch.load(w.checked(Path(c['target_directory'])/f'{s}_seed{c["target_seed"]}.pt'),map_location='cpu',weights_only=True)
            t=AutoEncoderTopK(512,len(initial[s]['encoder.weight']),int(initial[s]['k'])).to(w.device)
            t.load_state_dict(initial[s]); t.requires_grad_(False); targets[s]=t
        for f in ['model.safetensors','config.json','tokenizer.json']: w.checked(Path(c['model_local_dir'])/f)
        model=transformers.AutoModelForCausalLM.from_pretrained(c['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').to(w.device).eval()
        model.requires_grad_(False); model.config.use_cache=False
        tok=transformers.AutoTokenizer.from_pretrained(c['model_local_dir'],local_files_only=True)
        answer=tok.encode(' to',add_special_tokens=False); assert len(answer)==1
        probe=np.load(w.checked(Path(c['frozen_source_run'])/'probe.npz'))
        pw=torch.tensor(probe['weight'],device=w.device); pb=torch.tensor(probe['bias'],device=w.device)
        external=json.loads(w.checked(c['human_panel']).read_text())['rows']; hr=[]
        for prof in sorted({r['profession'] for r in external}):
            for gender in [0,1]:
                cell=[r for r in external if r['split']==c.get('human_split','test') and r['profession']==prof and r['gender']==gender]
                hr.extend(sorted(cell,key=lambda r:r['document_sha256'])[:c['human_per_cell']])
        gp=json.loads(w.checked(c['grammar_panel']).read_text()); gr=gp['rows'][:c['grammar_rows']]
        def tokenize_grammar(rows):
            for row in rows:
                row['tokens']=tok.encode(row['text'],add_special_tokens=False)
                if c.get('grammar_endpoint')=='answer_difference':
                    for key in ['clean_answer','patch_answer']:
                        value=tok.encode(row[key],add_special_tokens=False)
                        if len(value)!=1: raise ValueError('Agreement answer must be one token')
                        row[key+'_id']=value[0]
        tokenize_grammar(gr)
        hq={}
        for name,weights in c['human_queries'].items():
            hq[name]={s:[sum(weights[g] for g in groups if member in groups[g].get(s,[])) for member in source['members'][s]] for s in sites}
        assert all(v==1 for values in hq['full'].values() for v in values)
        hq.update(c.get('human_member_queries',{}))
        gq=gp['queries']; dataset='human'; mode='clean'; q={}; mask=None; pool=None; observed={}; diagnostics={}
        train_site=None; train_sp=None; train_reconstruction=None; train_local_loss=None
        train_sources={}; train_queries={}; train_terms=[]; metric_roots={}; source_roots={}; other_roots={}
        token_grams={}; token_counts={}; token_roots={}; embedding_cache={}; current_tokens=None
        hybrid_prefix=None; datasets=c.get('datasets',['human','grammar'])
        assert datasets and set(datasets)<= {'human','grammar'}
        def hook(site):
            def f(module,inputs,out):
                nonlocal pool,train_reconstruction,train_local_loss
                h=out[0] if isinstance(out,tuple) else out
                if mode=='clean': observed[site]=h.detach()
                elif mode=='metric':
                    if site=='embed': h=h.detach().requires_grad_(True)
                    observed[site]=h
                elif mode=='source_metric':
                    if site=='embed': h=h.detach().requires_grad_(True)
                    if dataset=='human' or site=='resid_4':
                        sp=human[site] if dataset=='human' else grammar
                        query=q[site] if dataset=='human' else q['resid_4']
                        zs=torch.relu((h-sp['center'])@sp['encoder'].T+sp['encoder_bias'])
                        h=h-(zs*query)@sp['decoder']
                    observed[site]=h
                elif dataset=='training' and site in train_sources:
                    train_sp=train_sources[site]; train_q=train_queries[site]
                    if mode=='source':
                        zs=torch.relu((h-train_sp['center'])@train_sp['encoder'].T+train_sp['encoder_bias'])
                        h=h-(zs*train_q)@train_sp['decoder']
                    elif mode=='tangent':
                        t=targets[site]
                        train_reconstruction=(t(h)-h).square().mean()/h.square().mean().clamp_min(1e-8)
                        capacity_first=c.get('capacity_first',False)
                        delta,_,_=transport_delta(h,t,train_sp,train_q,t.encoder.weight,2*len(train_q),not capacity_first,capacity_first)
                        zs=torch.relu((h-train_sp['center'])@train_sp['encoder'].T+train_sp['encoder_bias'])
                        requested=-(zs*train_q)@train_sp['decoder']
                        train_local_loss=(delta-requested).square().sum()/requested.square().sum().clamp_min(1e-10)
                        train_terms.append((train_reconstruction,train_local_loss))
                        h=h+delta
                    else: raise ValueError(mode)
                elif dataset=='human' or site=='resid_4':
                    if dataset=='training': return out
                    sp=human[site] if dataset=='human' else grammar
                    query=q[site] if dataset=='human' else q['resid_4']
                    zs=torch.relu((h-sp['center'])@sp['encoder'].T+sp['encoder_bias'])
                    u=-(zs*query)@sp['decoder']; t=targets[site]
                    allowance=c.get('member_budget',2*len(query))
                    if mode=='source' or (dataset=='human' and hybrid_prefix is not None and sites.index(site)>=hybrid_prefix): h=h+u
                    elif mode=='raw_readout':
                        rec=t.decode(t.encode(h)); zsr=torch.relu((rec-sp['center'])@sp['encoder'].T+sp['encoder_bias'])
                        h=h-(zsr*query)@sp['decoder']
                    elif mode=='tangent':
                        delta,_,_=transport_delta(h,t,sp,query,t.encoder.weight,allowance,True); h=h+delta
                    elif mode=='feasible_tangent':
                        delta,dz,columns=transport_delta(h,t,sp,query,t.encoder.weight,allowance,False,True)
                        z=t.encode(h); real=mask.bool()
                        if float((z+dz)[real].min()) < -2e-5: raise ValueError('Negative target code')
                        if int((dz[real]!=0).sum(-1).max()) > allowance: raise ValueError('Member allowance exceeded')
                        d=diagnostics.setdefault(site,dict(error=0.,energy=0.,changed=0.,newly_active=0.,states=0,min_code=0.))
                        d['error']+=float((delta-u)[real].square().sum()); d['energy']+=float(u[real].square().sum())
                        d['changed']+=int((dz[real]!=0).sum()); d['newly_active']+=int(((z[real]==0)&(dz[real]>0)).sum())
                        d['states']+=int(real.sum()); d['min_code']=min(d['min_code'],float((z+dz)[real].min()))
                        h=h+delta
                    elif mode in ['pursuit_common','pursuit_source_metric']:
                        root=source_roots[dataset][site] if mode=='pursuit_source_metric' else None
                        columns=pursuit_columns(h,t,sp,allowance,root,c['inverse_steps'],c['inverse_candidates'])
                        dz=columns@query; z=t.encode(h)
                        if float((z+dz)[mask.bool()].min()) < -2e-5: raise ValueError('Negative pursuit target code')
                        if int((dz[mask.bool()]!=0).sum(-1).max()) > allowance: raise ValueError('Pursuit member allowance exceeded')
                        h=h+dz@t.decoder.weight.T
                    elif mode in ['refined_conditional_source_metric','refined_cached_source_metric'] and site=='embed':
                        flat=h.reshape(-1,h.shape[-1]); token_ids=current_tokens.flatten()
                        delta=torch.zeros_like(flat)
                        active=zs.reshape(len(flat),-1).sum(-1)>0
                        for token in torch.unique(token_ids[active]).tolist():
                            take=(token_ids==token).nonzero().flatten()
                            if token not in embedding_cache:
                                state=flat[take[:1]]
                                if not torch.allclose(flat[take],state.expand(len(take),-1),atol=1e-6,rtol=0):
                                    raise ValueError('Token-conditioned embedding requires identical states')
                                _,_,cc=transport_delta(state,t,sp,query,t.encoder.weight,allowance,False)
                                root=token_roots[token] if mode=='refined_conditional_source_metric' and token in token_roots else source_roots[dataset][site]
                                cc=refine_columns(state,t,sp,cc,allowance,c['inverse_steps'],metric_root=root,accelerate=c.get('accelerate_refinement',False))
                                embedding_cache[token]=(cc[0],t.encode(state)[0])
                            cc,zz=embedding_cache[token]
                            dz=cc@query
                            if float((zz+dz).min()) < -2e-5: raise ValueError('Negative conditional target code')
                            delta[take]=(dz@t.decoder.weight.T).expand(len(take),-1)
                        h=h+delta.reshape_as(h)
                    elif mode in ['refined_common','refined_fisher','refined_source_metric','refined_other_metric','refined_conditional_source_metric','refined_cached_source_metric']:
                        _,_,columns=transport_delta(h,t,sp,query,t.encoder.weight,allowance,False)
                        root=source_roots[dataset][site] if mode in ['refined_source_metric','refined_conditional_source_metric','refined_cached_source_metric'] else other_roots[dataset][site] if mode=='refined_other_metric' else metric_roots[site] if mode=='refined_fisher' else None
                        refined=refine_columns(h,t,sp,columns,allowance,c['inverse_steps'],metric_root=root,accelerate=c.get('accelerate_refinement',False))
                        dz=refined@query; z=t.encode(h)
                        if float((z+dz)[mask.bool()].min()) < -2e-5: raise ValueError('Negative refined target code')
                        if int((dz[mask.bool()]!=0).sum(-1).max()) > allowance: raise ValueError('Refined member allowance exceeded')
                        h=h+dz@t.decoder.weight.T
                    else:
                        z=t.encode(h); dz=t.encode(h+u)-z
                        if mode=='inverse_budget':
                            delta,coeff,ix,_=realize(u,dz,t.decoder.weight.T,members=allowance,
                                steps=c['inverse_steps'],refine_steps=c['inverse_steps'],
                                batch_size=128,candidate_limit=c['inverse_candidates'],current_codes=z)
                            dz=torch.zeros_like(z).scatter(-1,ix,coeff)
                        elif mode=='finite_budget':
                            score=dz.abs()*t.decoder.weight.norm(dim=0)
                            keep=torch.zeros_like(score).scatter(-1,score.topk(min(allowance,score.shape[-1]),dim=-1).indices,1.)
                            dz=dz*keep
                        elif mode!='finite_full': raise ValueError(mode)
                        delta=dz@t.decoder.weight.T
                        d=diagnostics.setdefault(site,dict(error=0.,energy=0.,changed=0.,states=0,min_code=0.))
                        real=mask.bool(); d['error']+=float((delta-u)[real].square().sum()); d['energy']+=float(u[real].square().sum())
                        d['changed']+=int((dz[real]!=0).sum()); d['states']+=int(real.sum()); d['min_code']=min(d['min_code'],float((z+dz)[real].min()))
                        h=h+delta
                if site=='resid_4': pool=(h*mask[...,None]).sum(1)/mask.sum(1)[...,None]
                return (h,*out[1:]) if isinstance(out,tuple) else h
            return f
        for s in sites: handles.append(site_module(model,s).register_forward_hook(hook(s)))
        def forward(rows):
            nonlocal mask,current_tokens
            size=max(len(r['tokens']) for r in rows)
            ids=torch.zeros((len(rows),size),device=w.device,dtype=torch.long); mask=torch.zeros_like(ids)
            for j,r in enumerate(rows):
                ids[j,:len(r['tokens'])]=torch.tensor(r['tokens'],device=w.device); mask[j,:len(r['tokens'])]=1
            current_tokens=ids
            hidden=model.gpt_neox(ids,attention_mask=mask,use_cache=False).last_hidden_state
            w.sequence_forwards+=len(rows); w.token_forwards+=int(mask.sum())
            if dataset=='training': return hidden
            if dataset=='human': return (pool@pw.T+pb).squeeze(-1)
            last=hidden[torch.arange(len(rows),device=w.device),mask.sum(1)-1]
            logits=model.get_output_embeddings()(last)
            if c.get('grammar_endpoint')=='answer_difference':
                clean_ids=torch.tensor([r['clean_answer_id'] for r in rows],device=w.device)
                patch_ids=torch.tensor([r['patch_answer_id'] for r in rows],device=w.device)
                batch=torch.arange(len(rows),device=w.device)
                return logits[batch,clean_ids]-logits[batch,patch_ids]
            return torch.log_softmax(logits,dim=-1)[:,answer[0]]
        responses={}; representations={}; diagnostic_rows=[]; quality=[]
        @torch.no_grad()
        def evaluate(label,execution):
            nonlocal mode,dataset,q,diagnostics,embedding_cache
            embedding_cache={}
            mode=execution; output_label=label if hybrid_prefix is None else f'{label}_prefix{hybrid_prefix}'
            for ds,rows,queries in [('human',hr,hq),('grammar',gr,gq)]:
                if ds not in datasets: continue
                if label=='task_adapted_tangent':
                    for s in sites: targets[s].load_state_dict(initial[s])
                    paths=c['task_adapted_reference']['human'] if ds=='human' else {'resid_4':c['task_adapted_reference']['grammar']}
                    for s,path in paths.items():
                        targets[s].load_state_dict(torch.load(w.checked(path),map_location=w.device,weights_only=True))
                dataset=ds; values=[]; pooled_queries=[]
                for name,v in queries.items():
                    q={s:torch.tensor(x,device=w.device,dtype=torch.float32) for s,x in (v.items() if ds=='human' else [('resid_4',v)])}
                    diagnostics={}; chunks=[]; pooled_chunks=[]
                    off=0
                    while off<len(rows):
                        count=min(c['eval_batch_size'],len(rows)-off)
                        if c.get('eval_token_budget'):
                            while count>1 and count*max(len(r['tokens']) for r in rows[off:off+count])>c['eval_token_budget']: count-=1
                        chunks.append(forward(rows[off:off+count]).cpu().numpy()); off+=count
                        if ds=='human' and c.get('save_pooled',False): pooled_chunks.append(pool.detach().cpu().numpy())
                    if pooled_chunks: pooled_queries.append(np.concatenate(pooled_chunks))
                    values.append(np.concatenate(chunks)); diagnostic_rows.append(dict(method=output_label,dataset=ds,query=name,sites=diagnostics))
                    if c.get('query_progress',False): w.progress('QUERY_EVALUATION',dataset=ds,method=output_label,query=name,completed=len(values),total=len(queries),rows=len(rows))
                responses[ds+'__'+output_label]=np.stack(values)
                np.savez_compressed(w.run/'responses.npz',**responses)
                if pooled_queries:
                    representations[output_label]=np.stack(pooled_queries)
                    np.savez_compressed(w.run/'pooled.npz',**representations)
                w.progress('EVALUATION',dataset=ds,method=output_label,queries=len(queries),rows=len(rows))
        # 自然文本和训练动作在功能结果计算之前确定。
        tokens=np.memmap(w.checked(c['paired_tokens']),dtype='<u2',mode='r').reshape(-1,128)
        used=np.arange(c['natural_offset'],c['natural_offset']+c['natural_sequences'])
        if c.get('natural_cache'):
            natural=torch.load(w.checked(c['natural_cache']),map_location='cpu',weights_only=True)
        else:
            collected={s:[] for s in sites}; mode='clean'; dataset='human'
            with torch.no_grad():
                for off in range(0,len(used),4):
                    ids=used[off:off+4]; forward([{'tokens':tokens[i].astype('int64').tolist()} for i in ids])
                    for s in sites: collected[s].append(observed[s].reshape(-1,512).cpu())
            natural={s:torch.cat(v) for s,v in collected.items()}; del collected
            torch.save(natural,w.run/'natural_states.pt')
        with torch.no_grad():
            for s in sites:
                xx=natural[s][-c['quality_states']:].to(w.device); zz=targets[s].encode(xx); rr=targets[s].decode(zz)
                quality.append(dict(method='initial',site=s,fve=float(1-(rr-xx).square().sum()/(xx-xx.mean(0)).square().sum()),l0=float((zz>0).sum(-1).float().mean())))
        write(w.run/'membership.json',dict(natural_sequences=used.tolist(),human_rows=hr,grammar_rows=gr,
            human_queries=hq,grammar_queries=gq,human_query_order=list(hq),grammar_query_order=list(gq),
            site_order=sites,source_function_training=False))
        if c.get('fisher_sequences',0):
            grams={s:torch.zeros((512,512),device=w.device) for s in sites}
            gen=torch.Generator(device=w.device).manual_seed(c['training_seed'])
            metric_rows=used[:c['fisher_sequences']]; dataset='training'; mode='metric'
            for off in range(0,len(metric_rows),2):
                for draw in range(c['fisher_draws']):
                    rows=[{'tokens':tokens[i,:64].astype('int64').tolist()} for i in metric_rows[off:off+2]]
                    hidden=forward(rows); logits=model.get_output_embeddings()(hidden)
                    logp=torch.log_softmax(logits,dim=-1)
                    choices=torch.multinomial(logp.detach().exp().flatten(0,1),1,generator=gen).reshape(logp.shape[:-1])
                    objective=logp.gather(-1,choices[...,None]).sum()
                    gradients=torch.autograd.grad(objective,[observed[s] for s in sites])
                    for s,g in zip(sites,gradients):
                        flat=g.detach().reshape(-1,512); grams[s]+=flat.T@flat
                w.progress('NATURAL_FISHER',completed=min(off+2,len(metric_rows)),total=len(metric_rows),draws=c['fisher_draws'])
            for s,gram in grams.items():
                metric=gram/(torch.trace(gram)/512).clamp_min(1e-12)
                metric=(1-c['fisher_shrinkage'])*metric+c['fisher_shrinkage']*torch.eye(512,device=w.device)
                metric_roots[s]=torch.linalg.cholesky(metric)
            torch.save({s:v.cpu() for s,v in metric_roots.items()},w.run/'fisher_roots.pt')
            del grams,gradients,hidden,logits,logp,objective,observed
            observed={}
        if c.get('source_metric_rows',0):
            hp=json.loads(w.checked(Path(c['frozen_source_run'])/'panel.json').read_text())['rows']
            if c.get('source_metric_membership'):
                old_fit=set(json.loads(w.checked(c['source_metric_membership']).read_text())['fit_documents'])
                source_rows=sorted([r for r in hp if r['document_sha256'] in old_fit],key=lambda r:r['document_sha256'])[:c['source_metric_rows']]
                if len(source_rows)!=c['source_metric_rows']: raise ValueError('Insufficient source program fit contexts')
            else:
                source_rows=[]
                for y in [0,1]:
                    for g in [0,1]:
                        source_rows+=sorted([r for r in hp if r['split']=='train' and r['label']==y and r['gender']==g],key=lambda r:r['document_sha256'])[:c['source_metric_rows']//4]
            fit_grammar=json.loads(w.checked(c['grammar_fit_panel']).read_text())['rows'][:c['source_metric_rows']]
            tokenize_grammar(fit_grammar)
            if {r['document_sha256'] for r in source_rows}&{r['document_sha256'] for r in hr}: raise ValueError('Source metric and human evaluation overlap')
            if {r['text'] for r in fit_grammar}&{r['text'] for r in gr}: raise ValueError('Source metric and grammar evaluation overlap')
            write(w.run/'source_metric_membership.json',dict(human=source_rows,grammar=fit_grammar,path_fractions=c['source_metric_fractions'],target_responses_used=False))
            for ds,rr,ss in [('human',source_rows,sites),('grammar',fit_grammar,['resid_4'])]:
                if ds not in datasets: continue
                grams={s:torch.zeros((512,512),device=w.device) for s in ss}
                dataset=ds; mode='source_metric'
                metric_generator=torch.Generator(device=w.device).manual_seed(c['training_seed']+101)
                for fraction in c['source_metric_fractions']:
                    q={s:torch.full((len(human[s]['encoder']) if ds=='human' else len(grammar['encoder']),),fraction,device=w.device) for s in ss}
                    for off in range(0,len(rr),4):
                        response=forward(rr[off:off+4])
                        representation_metric=ds=='human' and c.get('human_profile_endpoint')=='pooled_representation'
                        projections=c.get('profile_projections',8) if representation_metric else 1
                        for draw in range(projections):
                            if representation_metric:
                                direction=torch.randn(pool.shape[-1],device=w.device,generator=metric_generator)
                                objective=(pool@direction).sum()
                            else: objective=response.sum()
                            gradients=torch.autograd.grad(objective,[observed[s] for s in ss],retain_graph=draw+1<projections)
                            for s,g in zip(ss,gradients):
                                flat=g.detach()[mask.bool()]; grams[s]+=flat.T@flat/projections
                                if s=='embed' and ds=='human' and c.get('conditional_profile',False):
                                    tokens_here=current_tokens[mask.bool()]
                                    clean_embed=torch.nn.functional.embedding(current_tokens,model.gpt_neox.embed_in.weight)[mask.bool()]
                                    sp=human['embed']
                                    active=torch.relu((clean_embed-sp['center'])@sp['encoder'].T+sp['encoder_bias']).sum(-1)>0
                                    for token in torch.unique(tokens_here[active]).tolist():
                                        selected=(tokens_here==token)&active
                                        gg=flat[selected]
                                        addition=(gg.T@gg/projections).cpu()
                                        if token not in token_grams:
                                            token_grams[token]=torch.zeros_like(addition)
                                            token_counts[token]=0
                                        token_grams[token]+=addition
                                        token_counts[token]+=int(selected.sum())/projections
                    w.progress('SOURCE_METRIC',dataset=ds,fraction=fraction,rows=len(rr))
                source_roots[ds]={}
                for s,gram in grams.items():
                    if torch.trace(gram)<=0: raise ValueError('Source response has zero gradient energy')
                    metric=gram/(torch.trace(gram)/512)
                    metric=(1-c['fisher_shrinkage'])*metric+c['fisher_shrinkage']*torch.eye(512,device=w.device)
                    source_roots[ds][s]=torch.linalg.cholesky(metric)
                    if ds=='human' and s=='embed' and c.get('conditional_profile',False):
                        count=sum(len(r['tokens']) for r in rr)*len(c['source_metric_fractions'])
                        global_mean=gram/count
                        prior=len(c['source_metric_fractions'])
                        for token,accumulated in token_grams.items():
                            conditional=(accumulated.to(w.device)+prior*global_mean)/(token_counts[token]+prior)
                            conditional=conditional/(torch.trace(conditional)/512).clamp_min(1e-12)
                            conditional=(1-c['fisher_shrinkage'])*conditional+c['fisher_shrinkage']*torch.eye(512,device=w.device)
                            token_roots[token]=torch.linalg.cholesky(conditional)
            torch.save({ds:{s:v.cpu() for s,v in vv.items()} for ds,vv in source_roots.items()},w.run/'source_metric_roots.pt')
            if token_roots:
                torch.save({token:v.cpu() for token,v in token_roots.items()},w.run/'source_metric_tokens.pt')
                write(w.run/'source_metric_token_counts.json',dict(counts=token_counts,prior_observations=len(c['source_metric_fractions']),unobserved='global mean through the same prior estimator',endpoint=c.get('human_profile_endpoint','original_classifier')))
            del grams,gradients,response
            observed={}
        if c.get('source_metric_cache'):
            source_roots=torch.load(w.checked(c['source_metric_cache']),map_location=w.device,weights_only=True)
        if c.get('other_source_metric_cache'):
            other_roots=torch.load(w.checked(c['other_source_metric_cache']),map_location=w.device,weights_only=True)
        if c.get('conditional_profile_cache'):
            token_roots=torch.load(w.checked(c['conditional_profile_cache']),map_location=w.device,weights_only=True)
        evaluate('none','clean'); evaluate('source','source')
        for prefix in c.get('hybrid_prefixes',[None]):
            hybrid_prefix=prefix
            for execution in c['executions']: evaluate('initial_'+execution,execution)
            if c.get('task_adapted_reference'):
                evaluate('task_adapted_tangent','tangent')
                for s in sites: targets[s].load_state_dict(initial[s])
        hybrid_prefix=None
        with zipfile.ZipFile(w.checked(c['source_archive'],'Public full source dictionaries','MIT')) as archive:
            for variant in c['variants']:
                for s in sites: targets[s].load_state_dict(initial[s])
                for s in c['train_sites']:
                    t=targets[s]; t.requires_grad_(False); t.encoder.requires_grad_(True)
                    if c['train_decoder']: t.decoder.requires_grad_(True)
                    xall=natural[s].to(w.device); xfit=xall[:-c['quality_states']]; xval=xall[-c['quality_states']:]
                    folder='embed' if s=='embed' else s.split('_')[0]+'_out_layer'+s.split('_')[1]
                    member=f'dictionaries/pythia-70m-deduped/{folder}/10_32768/ae.pt'
                    raw=archive.read(member); sd=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
                    excluded=set(source['members'][s]) | (set(c['grammar_members']) if s=='resid_4' else set())
                    available=np.array([i for i in range(len(sd['encoder.weight'])) if i not in excluded])
                    rng=np.random.default_rng(c['training_seed']+sites.index(s)); members=rng.choice(available,min(c['action_bank_size'],len(available)),replace=False)
                    enc=sd['encoder.weight'][members].to(w.device); bias=sd['encoder.bias'][members].to(w.device)
                    dec=sd['decoder.weight'][:,members].T.to(w.device); center=sd['bias'].to(w.device)
                    write(w.run/f'bank_{s}.json',dict(members=members.tolist(),excluded=sorted(excluded),archive_member=member,archive_member_sha256=hashlib.sha256(raw).hexdigest()))
                    del sd,raw
                    optimizer=torch.optim.AdamW([p for p in t.parameters() if p.requires_grad],lr=c['learning_rate'],weight_decay=0.)
                    gen=torch.Generator(device=w.device).manual_seed(c['training_seed']); seen=torch.zeros(len(members),device=w.device,dtype=torch.long)
                    for step in range(c['steps']):
                        ids=torch.randint(len(xfit),(c['state_batch'],),generator=gen,device=w.device); x=xfit[ids]
                        with torch.no_grad():
                            zs=torch.relu((x-center)@enc.T+bias)
                            # 每个状态从当前激活的源成员采样动作。
                            scores=torch.rand(zs.shape,generator=gen,device=w.device)*(zs>0)
                            chosen=scores.topk(c['actions_per_state'],dim=-1).indices
                            action_codes=zs.gather(1,chosen)*torch.rand((len(x),1),generator=gen,device=w.device)
                            u=-(action_codes[...,None]*dec[chosen]).sum(1)
                            seen.scatter_add_(0,chosen.flatten(),(action_codes>0).long().flatten())
                        rx=t(x); ry=t(x+u); scale=x.square().mean().clamp_min(1e-8)
                        recon=(rx-x).square().mean()/scale
                        if variant=='natural': loss=recon
                        elif variant=='endpoints': loss=.5*(recon+(ry-x-u).square().mean()/scale)
                        elif variant=='increments': loss=recon+c['change_weight']*(ry-rx-u).square().sum()/u.square().sum().clamp_min(1e-8)
                        else: raise ValueError(variant)
                        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(t.parameters(),1.); optimizer.step()
                        if c['train_decoder']:
                            with torch.no_grad(): t.decoder.weight.div_(t.decoder.weight.norm(dim=0).clamp_min(1e-10))
                        if not torch.isfinite(loss): raise FloatingPointError('Non-finite training objective')
                        if (step+1)%c['log_every']==0:
                            w.record(kind='training',task=s,method=variant,row_id=step+1,component='natural_excluded_actions',operation='fit',seed=c['target_seed'],loss=float(loss.detach()),recon=float(recon.detach()),members_seen=int((seen>0).sum()))
                            w.progress('TRAINING',site=s,method=variant,step=step+1,total=c['steps'],loss=float(loss.detach()),members_seen=int((seen>0).sum()))
                        if time.perf_counter()-w.wall_start>c['budget_seconds']: raise TimeoutError('Allocated driver budget reached')
                    with torch.no_grad():
                        zz=t.encode(xval); rr=t.decode(zz)
                        quality.append(dict(method=variant,site=s,fve=float(1-(rr-xval).square().sum()/(xval-xval.mean(0)).square().sum()),l0=float((zz>0).sum(-1).float().mean()),members_seen=int((seen>0).sum())))
                    dest=w.run/variant; dest.mkdir(exist_ok=True); torch.save(t.state_dict(),dest/f'{s}_seed{c["target_seed"]}.pt')
                    t.requires_grad_(False); del optimizer,xall,xfit,xval,enc,bias,dec,center; torch.cuda.empty_cache()
                for execution in c['executions']: evaluate(variant+'_'+execution,execution)
            if c.get('generic_program_steps',0):
                for s in sites: targets[s].load_state_dict(initial[s]); targets[s].requires_grad_(True)
                action_banks={}
                for s in sites:
                    folder='embed' if s=='embed' else s.split('_')[0]+'_out_layer'+s.split('_')[1]
                    member=f'dictionaries/pythia-70m-deduped/{folder}/10_32768/ae.pt'
                    raw=archive.read(member); sd=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
                    excluded=set(source['members'][s]) | (set(c['grammar_members']) if s=='resid_4' else set())
                    available=np.array([i for i in range(len(sd['encoder.weight'])) if i not in excluded])
                    rng=np.random.default_rng(c['training_seed']+sites.index(s)); members=rng.choice(available,min(c['action_bank_size'],len(available)),replace=False)
                    action_banks[s]=dict(encoder=sd['encoder.weight'][members].to(w.device),encoder_bias=sd['encoder.bias'][members].to(w.device),decoder=sd['decoder.weight'][:,members].T.to(w.device),center=sd['bias'].to(w.device))
                    write(w.run/f'bank_{s}.json',dict(members=members.tolist(),excluded=sorted(excluded),archive_member=member,archive_member_sha256=hashlib.sha256(raw).hexdigest()))
                    del sd,raw
                parameters=[p for t in targets.values() for p in t.parameters() if p.requires_grad]
                optimizer=torch.optim.AdamW(parameters,lr=c['learning_rate'],weight_decay=0.)
                gen=torch.Generator(device=w.device).manual_seed(c['training_seed'])
                fit_sequences=used[:-max(1,c['quality_states']//128)]
                for step in range(c['generic_program_steps']):
                    dataset='training'; mode='clean'; train_site=sites[step%len(sites)]
                    sample=torch.randint(len(fit_sequences),(c['program_batch_sequences'],),generator=gen,device=w.device).cpu().tolist()
                    rr=[{'tokens':tokens[fit_sequences[i],:c['program_length']].astype('int64').tolist()} for i in sample]
                    with torch.no_grad():
                        clean_hidden=forward(rr); train_sources={}; train_queries={}
                        selected_sites=sites if c.get('program_scope','single')=='joint' else [train_site]
                        for s in selected_sites:
                            sb=action_banks[s]; hh=observed[s]
                            activity=torch.relu((hh-sb['center'])@sb['encoder'].T+sb['encoder_bias']).sum((0,1))
                            eligible=(activity>0).nonzero().flatten()
                            if len(eligible)<c['actions_per_state']: raise ValueError('Insufficient active source actions')
                            selected=eligible[torch.randperm(len(eligible),generator=gen,device=w.device)[:c['actions_per_state']]]
                            train_sources[s]={k:(v if k=='center' else v[selected]) for k,v in sb.items()}
                            train_queries[s]=.5+.5*torch.rand(c['actions_per_state'],generator=gen,device=w.device)
                        mode='source'; teacher=forward(rr); energy=(teacher-clean_hidden).square().mean()
                        if energy<1e-10: raise ValueError('Source program has insufficient effect')
                    mode='tangent'; train_terms=[]; student=forward(rr)
                    train_reconstruction=torch.stack([v[0] for v in train_terms]).mean()
                    train_local_loss=torch.stack([v[1] for v in train_terms]).mean()
                    downstream=(student-teacher).square().mean()/energy
                    program=train_local_loss if c.get('program_objective','downstream')=='local' else downstream
                    loss=program+c['reconstruction_weight']*train_reconstruction
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters,1.); optimizer.step()
                    with torch.no_grad():
                        for s in selected_sites:
                            t=targets[s]; t.decoder.weight.div_(t.decoder.weight.norm(dim=0).clamp_min(1e-10))
                    if not torch.isfinite(loss): raise FloatingPointError('Non-finite program loss')
                    if (step+1)%c['log_every']==0:
                        w.record(kind='training',task=train_site,method='generic_program',row_id=step+1,component='excluded_natural_actions',operation='fit',seed=c['target_seed'],loss=float(loss.detach()),program=float(program.detach()),energy=float(energy),recon=float(train_reconstruction.detach()))
                        w.progress('PROGRAM_TRAINING',step=step+1,total=c['generic_program_steps'],site=train_site,loss=float(loss.detach()),program=float(program.detach()))
                    if (step+1) in c['program_checkpoints']:
                        label='generic_program_'+str(step+1); dest=w.run/label; dest.mkdir()
                        for s,t in targets.items(): torch.save(t.state_dict(),dest/f'{s}_seed{c["target_seed"]}.pt')
                        evaluate(label,'feasible_tangent' if c.get('capacity_first',False) else 'tangent')
                        with torch.no_grad():
                            for s in sites:
                                xx=natural[s][-c['quality_states']:].to(w.device); zz=targets[s].encode(xx); rec=targets[s].decode(zz)
                                quality.append(dict(method=label,site=s,fve=float(1-(rec-xx).square().sum()/(xx-xx.mean(0)).square().sum()),l0=float((zz>0).sum(-1).float().mean())))
                        write(w.run/'quality.json',quality)
                    if time.perf_counter()-w.wall_start>c['budget_seconds']: raise TimeoutError('Allocated driver budget reached')
        write(w.run/'quality.json',quality); write(w.run/'execution_diagnostics.json',diagnostic_rows)
        summary=[]
        for ds in datasets:
            src=responses[ds+'__source']; clean=responses[ds+'__none']; energy=np.mean((src-clean)**2,axis=1)
            for key,values in responses.items():
                if not key.startswith(ds+'__') or key in [ds+'__source',ds+'__none']: continue
                err=np.mean((values-src)**2,axis=1)
                per=np.sqrt(err/np.maximum(energy,1e-12)); item=dict(dataset=ds,method=key.split('__')[1],mean_query_nrmse=float(per.mean()),global_nrmse=float(np.sqrt(err.mean()/energy.mean())),per_query=per.tolist(),source_energy=energy.tolist())
                summary.append(item); w.record(kind='response_error',task=ds,method=item['method'],row_id='all',component='fixed_function_and_development_panel',operation='all_queries',seed=c['target_seed'],nrmse=item['mean_query_nrmse'])
        write(w.run/'results.json',summary)
        w.checks.update(finite=all(np.isfinite(v).all() for v in responses.values()),complete_human_members=sum(map(len,source['members'].values()))==55,excluded_test_functions=True)
    except Exception: error=traceback.format_exc()
    finally:
        for h in handles: h.remove()
    return w.finish(error)


if __name__=='__main__': raise SystemExit(main())
