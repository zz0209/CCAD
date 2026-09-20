"""Compare response-space coverage while preserving the same source program.

Frozen source annotations and LM define the teacher. Configuration selects
natural or source-use contexts and complete, member, or semantic-part requests.
TopK architecture and decoder normalization reuse the pinned MIT implementation.
"""
from pathlib import Path
import argparse, json, sys, time, traceback
import numpy as np
from run_causalgym_multisite import MultisiteWork, write
from run_shift_explanation import source_groups
from train_shift_dictionaries import site_module
from run_shift_transfer import input_member_delta


def project_rows(a):
    """Euclidean projection onto nonnegative rows with total at most one."""
    import torch
    with torch.no_grad():
        a.clamp_(min=0)
        s=a.sort(dim=1,descending=True).values
        v=(s.cumsum(1)-1)/torch.arange(1,a.shape[1]+1,device=a.device)
        k=(s>v).sum(1).clamp_min(1)-1
        tau=v.gather(1,k[:,None]).clamp_min(0)
        a.copy_((a-tau).clamp_min(0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    args=p.parse_args();c=json.loads(args.config.read_text())
    w=MultisiteWork(c,args.config,['scripts/train_shift_response_space.py','scripts/run_shift_transfer.py','scripts/run_shift_explanation.py','scripts/train_shift_dictionaries.py','scripts/run_causalgym_multisite.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py'])
    hooks=[];error=None
    try:
        import torch,transformers
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest');torch.cuda.set_device(c['device'])
        torch.cuda.reset_peak_memory_stats();w.torch=torch;w.device=torch.device(c['device'])
        sys.path.extend([c['dictionary_source_dir'],c['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        w.environment=dict(python=sys.executable,torch=torch.__version__,numpy=np.__version__,transformers=transformers.__version__,cpu_threads=2)
        for file in ['dictionary_learning/trainers/top_k.py','LICENSE']:
            w.checked(Path(c['dictionary_source_dir'])/file,'Unchanged TopK implementation','MIT')
        source=json.loads(w.checked(c['source_manifest'],'Original human source decisions','MIT').read_text())
        if c.get('request_panel'):
            import hashlib
            assert hashlib.sha256(w.checked(c['request_panel'],'Request coordinates frozen before execution').read_bytes()).hexdigest()==c['request_panel_sha256']
        groups,_=source_groups(w.checked(c['notebook'],'Original source annotations','MIT'),source['members'])
        sites=list(source['members']);sb=np.load(w.checked(c['source_parameters'],'Published source parameters','MIT'))
        sp={s:{k:torch.tensor(sb[s+'__'+k],device=w.device) for k in ['encoder','encoder_bias','decoder','center']} for s in sites}
        old=np.load(w.checked(Path(c['relation_run'])/'relation.npz','Frozen source-member relation'))
        baselines={}
        initial={};targets={};indices={};coefficients={};q={};source_gains={};write_matrices={}
        for s in sites:
            path=w.checked(Path(c['target_directory'])/f'{s}_seed{c["target_seed"]}.pt','Initial natural-data target dictionary')
            initial[s]=torch.load(path,map_location='cpu',weights_only=True)
            baselines[s]={k:torch.tensor(old[s+'__'+k],device=w.device,dtype=torch.long if k=='candidates' else torch.float32) for k in ['geometry','geometry_gain','raw','candidates']}
            sae=AutoEncoderTopK(512,initial[s]['encoder.weight'].shape[0],int(initial[s]['k'])).to(w.device)
            sae.load_state_dict(initial[s]);sae.requires_grad_(s in c['adapt_sites']);targets[s]=sae
            a=torch.tensor(old[s+'__native'],device=w.device,dtype=torch.float32)
            indices[s]=(a.sum(1)>0).nonzero().flatten()
            coefficients[s]=torch.nn.Parameter(a[indices[s]].clone())
            q[s]=torch.ones(a.shape[1],device=w.device)
            source_gains[s]=torch.nn.Parameter(torch.ones(a.shape[1],device=w.device),requires_grad=False)
            write_matrices[s]=torch.nn.Parameter(sae.encoder.weight.detach().clone(),requires_grad=False)
        for f in ['config.json','tokenizer.json','model.safetensors']:
            w.checked(Path(c['model_local_dir'])/f,'Pinned Pythia70M','Apache-2.0')
        model=transformers.AutoModelForCausalLM.from_pretrained(c['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to(w.device)
        model.requires_grad_(False);model.config.use_cache=False
        mode='clean';observed={};mask=None;pooled=None
        def hook(site):
            def f(module,inputs,out):
                nonlocal pooled
                x=out[0] if isinstance(out,tuple) else out
                if mode=='clean':observed[site]=x.detach()
                elif mode=='source':
                    ss=sp[site];zs=torch.relu((x-ss['center'])@ss['encoder'].T+ss['encoder_bias'])
                    x=x-(zs*q[site])@ss['decoder']
                elif mode=='student':
                    t=targets[site];ix=indices[site]
                    x=x-(t.encode(x)[...,ix]*(coefficients[site]@q[site]))@t.decoder.weight[:,ix].T
                elif mode.startswith('transport_'):
                    ss={**sp[site]}
                    basis=-(write_matrices[site]@ss['decoder'].T)
                    ss['transport_basis']=basis;ss['fixed_response_basis']=basis
                    operation='input_tangent_budget' if 'active' in mode else 'input_fixed_budget'
                    delta,_=input_member_delta(x,targets[site],ss,q[site],operation,c['members_per_source']*len(q[site]),mask)
                    x=x+delta
                elif mode in ('input_initial','tangent_gain','tangent_mixed'):
                    ss=sp[site]
                    if mode=='tangent_gain': ss={**ss,'decoder':ss['decoder']*source_gains[site][:,None]}
                    delta,_=input_member_delta(x,targets[site],ss,q[site],'input_tangent_budget',c['members_per_source']*len(q[site]),mask)
                    x=x+delta
                elif mode in ('geometry','geometry_gain','raw'):
                    t=targets[site];z=t.encode(x);b=baselines[site]
                    if mode=='raw':
                        x=x-(z[...,b['candidates']]@b['raw']*q[site])@sp[site]['decoder']
                    else:
                        x=x-(z*(b[mode]@q[site]))@t.decoder.weight.T
                elif mode=='raw_reconstruction':
                    delta,_=input_member_delta(x,targets[site],sp[site],q[site],mode,c['members_per_source']*len(q[site]),mask)
                    x=x+delta
                elif mode=='clean_reconstruction' and site in c['adapt_sites']:
                    x=targets[site](x)
                if site=='resid_4':pooled=(x*mask[...,None]).sum(1)/mask.sum(1)[...,None]
                return (x,*out[1:]) if isinstance(out,tuple) else x
            return f
        for s in sites:hooks.append(site_module(model,s).register_forward_hook(hook(s)))
        token_path=w.checked(c['paired_tokens'],'Natural discovery text; no biography labels','ODC-By-1.0')
        tokens=np.memmap(token_path,dtype='<u2',mode='r').reshape(-1,128)
        start=c.get('token_offset_sequences',0);count=c['steps']*c['batch_sequences']
        available=np.arange(start,len(tokens)-4)
        assert len(available)>0
        order=np.random.default_rng(c['training_seed']).permutation(available)
        used=np.resize(order,count)
        nat=np.array(tokens[used],dtype='int64')
        write(w.run/'training_membership.json',dict(sequence_indices=used.tolist(),unique_sequences=len(set(used.tolist())),repeats_allowed=True,calibration_sequences=[0,64],quality_sequences=[len(tokens)-4,len(tokens)]))
        panel=json.loads(w.checked(Path(c['frozen_source_run'])/'panel.json','Already exposed development panel').read_text())
        dev=[]
        for y in [0,1]:
            for g in [0,1]:
                dev.extend(sorted([r for r in panel['rows'] if r['split']=='dev' and r['label']==y and r['gender']==g],key=lambda r:r['document_sha256'])[:c['development_per_group']])
        program_rows=[]
        if c.get('program_contexts')=='source_development':
            excluded={r['document_sha256'] for r in dev}
            if c.get('program_evaluation_exclude_per_group'):
                # Preserve original training membership when evaluating fewer
                # of the already excluded development documents.
                for y in [0,1]:
                    for g in [0,1]:
                        excluded.update(r['document_sha256'] for r in sorted([r for r in panel['rows'] if r['split']=='dev' and r['label']==y and r['gender']==g],key=lambda r:r['document_sha256'])[:c['program_evaluation_exclude_per_group']])
            program_rows=sorted([r for r in panel['rows'] if r['split']=='dev' and r['document_sha256'] not in excluded],key=lambda r:r['document_sha256'])
            assert len(program_rows)>64
            write(w.run/'program_context_membership.json',dict(documents=[r['document_sha256'] for r in program_rows],calibration_documents=[r['document_sha256'] for r in program_rows[:64]],fit_documents=[r['document_sha256'] for r in program_rows[64:]],labels_used=False,evaluation_disjoint=True))
        if c.get('evaluation_panel'):
            external=json.loads(w.checked(c['evaluation_panel'],c.get('evaluation_description','Previously exposed independent-task panel, now development')).read_text())
            dev=[]
            for profession in sorted({r['profession'] for r in external['rows']}):
                for gender in [0,1]:
                    cell=[r for r in external['rows'] if r['split']==c.get('evaluation_split','test') and r['profession']==profession and r['gender']==gender]
                    dev.extend(sorted(cell,key=lambda r:r['document_sha256'])[:c['evaluation_per_cell']])
            assert not {r['document_sha256'] for r in dev}&{r['document_sha256'] for r in program_rows}
        write(w.run/'evaluation_membership.json',dict(split=c.get('evaluation_evidence','exposed_development'),rows=dev))
        probe=np.load(w.checked(Path(c['frozen_source_run'])/'probe.npz','Original source head defining reusable responses and evaluation'))
        pw=torch.tensor(probe['weight'],device=w.device);pb=torch.tensor(probe['bias'],device=w.device)
        def forward(ids):
            nonlocal mask
            result=model.gpt_neox(ids,attention_mask=mask,use_cache=False).last_hidden_state
            w.sequence_forwards+=len(ids);w.token_forwards+=ids.numel()
            return result,pooled
        def program_batch(rows):
            nonlocal mask
            length=max(len(r['tokens']) for r in rows)
            ids=torch.zeros((len(rows),length),device=w.device,dtype=torch.long);mask=torch.zeros_like(ids)
            for j,row in enumerate(rows):
                v=row['tokens'];ids[j,:len(v)]=torch.tensor(v,device=w.device);mask[j,:len(v)]=1
            return ids
        def token_mse(delta):
            return (delta.square()*mask[...,None]).sum()/(mask.sum()*delta.shape[-1])
        def set_query(name):
            for s in sites:
                if name in c.get('member_queries', {}):
                    q[s]=torch.tensor(c['member_queries'][name][s],device=w.device,dtype=torch.float32)
                elif name in c.get('dose_queries', {}):
                    weights=c['dose_queries'][name]
                    q[s]=torch.tensor([sum(weights[g] for g in groups if i in groups[g].get(s,[])) for i in source['members'][s]],device=w.device,dtype=torch.float32)
                else:
                    q[s]=torch.tensor([name=='full' or any(i in groups[g].get(s,[]) for g in name.split('+')) for i in source['members'][s]],device=w.device,dtype=torch.float32)
        @torch.no_grad()
        def evaluate(name):
            nonlocal mode,mask
            mode=name if name.startswith('transport_') or name in ('source','geometry','geometry_gain','raw','raw_reconstruction','input_initial','tangent_gain','tangent_mixed') else ('clean' if name=='none' else 'student')
            for query in (['full'] if name=='none' else c['queries']):
                set_query(query);values=np.empty(len(dev),dtype='float32');pooled_values=np.empty((len(dev),512),dtype='float32')
                order=sorted(range(len(dev)),key=lambda i:len(dev[i]['tokens']))
                off=0
                while off<len(order):
                    count=min(c['eval_batch_size'],len(order)-off)
                    if c.get('eval_token_budget'):
                        while count>1 and count*len(dev[order[off+count-1]]['tokens'])>c['eval_token_budget']:count-=1
                    ix=order[off:off+count];off+=count;length=max(len(dev[i]['tokens']) for i in ix)
                    ids=torch.zeros((len(ix),length),device=w.device,dtype=torch.long);mask=torch.zeros_like(ids)
                    for j,i in enumerate(ix):
                        v=dev[i]['tokens'];ids[j,:len(v)]=torch.tensor(v,device=w.device);mask[j,:len(v)]=1
                    _,po=forward(ids);values[ix]=(po@pw.T+pb).squeeze(-1).cpu().numpy();pooled_values[ix]=po.cpu().numpy()
                np.save(w.run/f'{name}__{query}__pooled.npy',pooled_values)
                for row,value in zip(dev,values):
                    w.record(kind='classification' if 'label' in row else 'response',task='profession',row_id=row['row_id'],component=row['document_sha256'],method=name,operation=query,seed=c['target_seed'],target_seed=c['target_seed'],split='dev',label=row.get('label'),gender=row['gender'],prediction=int(value>0),logit=float(value))
                w.progress('EVALUATION',method=name,query=query)
        evaluate('none');evaluate('source');evaluate('initial')
        for baseline in c.get('evaluate_baselines',[]):
            evaluate(baseline)
        if c.get('evaluate_checkpoints'):
            for variant,folder in c['evaluate_checkpoints'].items():
                folder=Path(folder)
                for s in c['adapt_sites']:
                    targets[s].load_state_dict(torch.load(w.checked(folder/f'{s}_seed{c["target_seed"]}.pt'),map_location=w.device,weights_only=True))
                relation=np.load(w.checked(folder/'relation.npz'))
                for s in sites:
                    a=torch.tensor(relation[s+'__native'],device=w.device,dtype=torch.float32)
                    coefficients[s].data.copy_(a[indices[s]])
                if variant=='tangent_gain':
                    loaded=np.load(w.checked(folder/'source_gains.npz'))
                    for s in sites:source_gains[s].data.copy_(torch.tensor(loaded[s],device=w.device))
                if variant.startswith('transport_'):
                    loaded=torch.load(w.checked(folder/'write_matrices.pt'),map_location=w.device,weights_only=True)
                    for s in sites:write_matrices[s].data.copy_(loaded[s])
                evaluate(variant)
            w.checks['completed_fixed_checkpoints']=True
            return w.finish(None)
        # One common normalization for every arm. Near-zero-effect batches must
        # not receive arbitrarily larger weight than informative source actions.
        calibration=[];pooled_effects=[];calgen=torch.Generator(device=w.device).manual_seed(917)
        with torch.no_grad():
            for off in range(0,64,c['batch_sequences']):
                if program_rows:ids=program_batch(program_rows[off:off+c['batch_sequences']])
                else:
                    ids=torch.tensor(np.array(tokens[off:off+c['batch_sequences']],dtype='int64'),device=w.device);mask=torch.ones_like(ids)
                mode='clean';ch,cp=forward(ids)
                for full in [False,True]:
                    for s in sites:q[s]=torch.ones_like(q[s]) if full else (torch.rand(q[s].shape,generator=calgen,device=w.device)<.5).float()
                    mode='source';th,tp=forward(ids)
                    pooled_effects.append((tp-cp).detach())
                    calibration.append([float(token_mse(th-ch)),float((tp-cp).square().mean()),float(((tp-cp)@pw.T).square().mean())])
        scales=np.maximum(np.mean(calibration,axis=0),1e-8).tolist()
        effects=torch.cat(pooled_effects)
        covariance=effects.T@effects/len(effects)
        damping=c['response_covariance_ridge']*torch.trace(covariance)/covariance.shape[0]
        metric=torch.linalg.inv(covariance+damping*torch.eye(covariance.shape[0],device=w.device))
        metric=(metric+metric.T)/2
        metric_energy=torch.einsum('bi,ij,bj->b',effects,metric,effects).mean().clamp_min(1e-8)
        np.savez_compressed(w.run/'source_response_metric.npz',covariance=covariance.cpu().numpy(),metric=metric.cpu().numpy(),effects=effects.cpu().numpy(),damping=float(damping),normalizer=float(metric_energy),eigenvalues=torch.linalg.eigvalsh(covariance).cpu().numpy())
        del pooled_effects
        write(w.run/'loss_scales.json',dict(final_hidden=scales[0],pooled_hidden=scales[1],source_response=scales[2],calibration_contexts=c.get('program_contexts','natural'),calibration_indices=[0,64],requests='32 paired full and Bernoulli-member calibrations; source only; common across arms'))
        params=[p for s in c['adapt_sites'] for p in targets[s].parameters()]
        base_coeff={s:a.detach().clone() for s,a in coefficients.items()}
        bulk=Path(c['bulk_output_dir']);bulk.mkdir(parents=True,exist_ok=False)
        checkpoints=[]
        for variant in c['variants']:
            independent=variant.startswith('transport_')
            for s in sites:
                targets[s].load_state_dict(initial[s]);coefficients[s].data.copy_(base_coeff[s])
                coefficients[s].requires_grad_(not independent and not variant.startswith('tangent_'))
                source_gains[s].data.fill_(1.);source_gains[s].requires_grad_(variant=='tangent_gain')
                write_matrices[s].data.copy_(targets[s].encoder.weight);write_matrices[s].requires_grad_(independent and s in c['adapt_sites'])
            for s in c['adapt_sites']:targets[s].requires_grad_(not independent and variant not in ('parts_relation','tangent_gain'))
            groups_opt=[dict(params=[p for p in params if p.requires_grad]+[v for v in write_matrices.values() if v.requires_grad],lr=c['dictionary_lr'])]
            groups_opt.append(dict(params=[a for a in coefficients.values() if a.requires_grad]+[g for g in source_gains.values() if g.requires_grad],lr=c['relation_lr']))
            optim=torch.optim.AdamW(groups_opt,weight_decay=0.)
            generator=torch.Generator(device=w.device).manual_seed(c['training_seed'])
            for step in range(c['steps']):
                ids=torch.tensor(nat[step*c['batch_sequences']:(step+1)*c['batch_sequences']],device=w.device)
                mask=torch.ones_like(ids);mode='clean'
                with torch.no_grad():clean_h,clean_pool=forward(ids);clean={s:observed[s].clone() for s in c['adapt_sites']}
                if independent or variant in ('head_parts','pooled_parts','white_parts','pooled_whole','parts_relation','head_continuous','pooled_continuous','head_mixed','pooled_mixed','tangent_gain','tangent_mixed'):
                    if program_rows:
                        fit=program_rows[64:];take=[fit[(step*c['batch_sequences']+j)%len(fit)] for j in range(c['batch_sequences'])]
                        ids=program_batch(take);mode='clean'
                        with torch.no_grad():clean_h,clean_pool=forward(ids)
                    if variant.endswith(('_continuous','_mixed')) or variant=='tangent_gain':
                        if (variant.endswith('_mixed') or variant=='tangent_gain') and step%2==0:
                            # Balance interior participation against exact
                            # deletion corners at the same total update count.
                            bits=1+(step//2)%((1<<len(groups))-1)
                            group_weights=torch.tensor([(bits>>i)&1 for i in range(len(groups))],device=w.device,dtype=torch.float32)
                        else:
                            group_weights=torch.rand(len(groups),generator=generator,device=w.device)
                        for s in sites:
                            membership=torch.tensor([[i in groups[g].get(s,[]) for g in groups] for i in source['members'][s]],device=w.device,dtype=torch.float32)
                            q[s]=membership@group_weights
                    elif variant!='pooled_whole' and c.get('request_training')=='semantic_singletons':
                        set_query(list(groups)[step%len(groups)])
                    else:
                        for s in sites:q[s]=torch.ones_like(q[s])
                    mode='source'
                    with torch.no_grad():teacher_h,teacher_pool=forward(ids)
                    mode=variant if independent or variant.startswith('tangent_') else 'student';student_h,student_pool=forward(ids)
                    energy,pe=scales[:2]
                    state_loss=(token_mse(student_h-teacher_h)/energy+(student_pool-teacher_pool).square().mean()/pe)/2
                    response_energy=scales[2]
                    response_loss=((student_pool-teacher_pool)@pw.T).square().mean()/response_energy
                    weight=c.get('source_response_weight',0.)
                    if independent or variant in ('head_parts','head_continuous','head_mixed','tangent_gain','tangent_mixed'):
                        program=(1-weight)*state_loss+weight*response_loss
                    elif variant=='white_parts':
                        error_pool=student_pool-teacher_pool
                        program=torch.einsum('bi,ij,bj->b',error_pool,metric,error_pool).mean()/metric_energy
                    else:
                        program=(student_pool-teacher_pool).square().mean()/pe
                elif variant=='clean_e2e':
                    mode='clean_reconstruction';student_h,student_pool=forward(ids)
                    program=((student_h-clean_h).square().mean()/clean_h.square().mean().clamp_min(1e-8)+(student_pool-clean_pool).square().mean()/clean_pool.square().mean().clamp_min(1e-8))/2
                else:program=torch.zeros((),device=w.device)
                recon=sum((targets[s](clean[s])-clean[s]).square().mean()/clean[s].square().mean().clamp_min(1e-8) for s in c['adapt_sites'])/len(c['adapt_sites'])
                loss=program+c['reconstruction_weight']*recon
                optim.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_([p for group in optim.param_groups for p in group['params']],1.)
                optim.step()
                with torch.no_grad():
                    for s in c['adapt_sites']:
                        if targets[s].decoder.weight.requires_grad:targets[s].decoder.weight.div_(targets[s].decoder.weight.norm(dim=0).clamp_min(1e-10))
                    for a in coefficients.values():project_rows(a)
                    for g in source_gains.values():g.clamp_(min=0)
                assert torch.isfinite(loss)
                if (step+1)%16==0:
                    w.record(kind='training',task='natural_program',component=variant,row_id=step+1,method=variant,step=step+1,loss=float(loss.detach()),program=float(program.detach()),reconstruction=float(recon.detach()))
                    w.progress('TRAINING',method=variant,step=step+1,total_steps=c['steps'],loss=float(loss.detach()),program=float(program.detach()),reconstruction=float(recon.detach()),peak_cuda_bytes=torch.cuda.max_memory_allocated())
                if time.perf_counter()-w.wall_start>c['budget_seconds']:raise TimeoutError('Program adaptation reached allocated wall budget')
            dest=bulk/variant;dest.mkdir()
            for s in c['adapt_sites']:torch.save(targets[s].state_dict(),dest/f'{s}_seed{c["target_seed"]}.pt')
            if independent:
                torch.save({s:v.detach().cpu() for s,v in write_matrices.items()},dest/'write_matrices.pt')
                w.checks['all_dictionaries_unchanged_'+variant]=all(torch.equal(targets[s].state_dict()[k].cpu(),v) for s in sites for k,v in initial[s].items())
            export={}
            for s in sites:
                for key in ['native','geometry','geometry_gain','raw','candidates']:export[s+'__'+key]=old[s+'__'+key].copy()
                export[s+'__native'][indices[s].cpu().numpy()]=coefficients[s].detach().cpu().numpy()
            np.savez_compressed(dest/'relation.npz',**export)
            if variant=='tangent_gain':np.savez_compressed(dest/'source_gains.npz',**{s:g.detach().cpu().numpy() for s,g in source_gains.items()})
            checkpoints.append(dict(method=variant,directory=str(dest),adapt_sites=c['adapt_sites'],steps=c['steps']))
            write(w.run/'checkpoints.json',dict(checkpoints=checkpoints))
            evaluate(variant)
            # Comparable natural-state quality on a separate, previously exposed slice.
            mode='clean';ids=torch.tensor(np.array(tokens[-4:],dtype='int64'),device=w.device);mask=torch.ones_like(ids)
            with torch.no_grad():
                forward(ids)
                for s in c['adapt_sites']:
                    x=observed[s];z=targets[s].encode(x);rec=targets[s].decode(z)
                    w.record(kind='quality',task='natural_reconstruction',component=s,row_id=s,method=variant,site=s,fve=float(1-(rec-x).square().sum()/(x-x.mean((0,1))).square().sum()),l0=float((z>0).sum(-1).float().mean()),decoder_norm_error=float((targets[s].decoder.weight.norm(dim=0)-1).abs().max()))
            del optim;torch.cuda.empty_cache()
        w.checks.update(all_variants=len(checkpoints)==len(c['variants']),capacity=all(float(a.detach().sum(1).max())<=1.00001 for a in coefficients.values()),nonnegative=all(float(a.detach().min())>=0 for a in coefficients.values()))
    except Exception:error=traceback.format_exc()
    finally:
        for hook in hooks:hook.remove()
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
