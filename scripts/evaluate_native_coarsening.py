"""Actual natural-context ablations for already fitted cross-seed groups."""
from __future__ import annotations
import json,time
from pathlib import Path
from run_r011s1_raw_hook_asset import ROOT
from ccad.artifacts import sha256


def evaluate(cfg,run,reference,rc,results,checked,write,log):
    import numpy as np,torch,transformers
    # The paired-state producer uses this matmul policy. Match it when
    # replaying physical states; the small quadratic fit uses highest instead.
    torch.set_float32_matmul_precision('high')
    timer=time.perf_counter();f=cfg['functional'];tr=ROOT/rc['training_run']
    tc=json.loads(checked(tr/'config.resolved.json').read_text())
    snaps=json.loads(checked(tr/'checkpoints.json').read_text())['checkpoints']
    for fn in ['config.json','tokenizer.json','model.safetensors']:checked(Path(tc['model_local_dir'])/fn)
    tokenizer=transformers.AutoTokenizer.from_pretrained(tc['model_local_dir'],local_files_only=True)
    model=transformers.AutoModelForCausalLM.from_pretrained(tc['model_local_dir'],local_files_only=True,
        dtype=torch.float32,attn_implementation='eager').eval().to('cuda:0');model.requires_grad_(False);model.config.use_cache=False
    module=model.get_submodule(tc['hook_module_path'])
    pm=json.loads(checked(rc['paired_manifest']).read_text());p=checked(pm['outputs']['calibration']['path'])
    assert sha256(p)==pm['outputs']['calibration']['sha256']
    tokens=np.fromfile(p,dtype='<u2').reshape(-1,128)
    with np.load(checked(reference/'natural_calibration_states.npz')) as a:
        packed=a['packed_positions'];hidden=torch.as_tensor(a['hidden'],device='cuda:0')
    Z={};D={};forwards=0;rows=[];max_replay=0.
    def codes(split,obj,seed):
        key=(split,obj,seed)
        if key not in Z:
            with np.load(checked(reference/f'natural_{split}_{obj}_seed{seed}.npz')) as a:
                z=torch.zeros(tuple(a['shape']),device='cuda:0')
                z[torch.as_tensor(a['rows'].astype('int64'),device='cuda:0'),torch.as_tensor(a['columns'].astype('int64'),device='cuda:0')]=torch.as_tensor(a['values'],device='cuda:0');Z[key]=z
        return Z[key]
    def decoder(obj,seed):
        if (obj,seed) not in D:
            snap=next(s for s in snaps if s['objective']==obj and s['seed']==seed and s['step']==rc['checkpoint_step'])
            p=checked(snap['path']);assert sha256(p)==snap['sha256']
            state=torch.load(p,map_location='cuda:0',weights_only=True)
            D[obj,seed]=(state['decoder.weight'].T if obj=='topk' else state['W_dec']).contiguous()
        return D[obj,seed]
    def output(ids,positions,delta=None):
        nonlocal forwards,max_replay
        actual=len(ids)
        if actual<rc['batch_size']:
            padding=rc['batch_size']-actual
            ids=torch.cat([ids,ids[:1].expand(padding,-1)])
            positions=torch.cat([positions,positions[:1].expand(padding)])
            if delta is not None:delta=torch.cat([delta,torch.zeros_like(delta[:1]).expand(padding,-1)])
        idx=torch.arange(len(ids),device='cuda:0');cache={}
        def hook(m,i,out):
            x=out[0] if isinstance(out,tuple) else out;cache['hidden']=x[idx,positions].detach()
            if delta is None:return out
            x=x.clone();x[idx,positions]-=delta
            return (x,)+out[1:] if isinstance(out,tuple) else x
        h=module.register_forward_hook(hook)
        try:
            # Double-precision normalization resolves the very small effects
            # on uniform contexts without changing physical interventions.
            with torch.no_grad():
                logits=model(ids,use_cache=False).logits
                out=logits[idx,positions].double().log_softmax(-1)
                future=logits[idx,(positions+f['future_offset']).clamp(max=127)].double().log_softmax(-1) if f.get('future_offset') else None
        finally:h.remove()
        forwards+=len(ids)
        return out[:actual],cache['hidden'][:actual],future[:actual] if future is not None else None
    for r in results:
        obj=r['objective'];s=r['source_seed'];t=r['target_seed'];key=r['query']
        with np.load(checked(run/(key+'_groups.npz'))) as a:group={k:a[k] for k in a.files}
        zs=codes('calibration',obj,s);zt=codes('calibration',obj,t);ds=decoder(obj,s);dt=decoder(obj,t)
        sp=torch.as_tensor(group['source_members'],device='cuda:0');tp=torch.as_tensor(group['target_members'],device='cuda:0')
        gs=torch.as_tensor(group['source_gate'],dtype=zs.dtype,device='cuda:0');gt=torch.as_tensor(group['target_gate'],dtype=zs.dtype,device='cuda:0')
        best=torch.as_tensor(group['best_atom_target_gate'],dtype=zs.dtype,device='cuda:0')
        atomt=torch.as_tensor(group['atom_target_gate'],dtype=zs.dtype,device='cuda:0')
        # Both strata are fixed from source activation or a deterministic grid.
        full_blocks=(len(tokens)//rc['batch_size'])*rc['batch_size']
        allowed=np.where((packed%128<127)&(packed%128>=f.get('minimum_context_position',0))&(packed//128<full_blocks))[0]
        context_score=zs[allowed,r['anchor']]
        if f.get('source_context_rule')=='group_activation':context_score=(zs[allowed][:,sp]*gs).sum(1)
        order=torch.argsort(context_score,descending=True,stable=True).cpu().numpy()
        chosen=[];seen=set()
        for i in np.asarray(allowed)[order]:
            block=int(packed[i]//128)
            if block not in seen:chosen.append((int(i),'source_active'));seen.add(block)
            if len(chosen)==f['active_contexts']:break
        grid=allowed[np.linspace(0,len(allowed)-1,f['uniform_contexts'],dtype=int)]
        chosen.extend((int(i),'uniform') for i in grid if int(i) not in {j for j,_ in chosen})
        select=torch.tensor([i for i,_ in chosen],device='cuda:0');n=len(select)
        source=(zs[select][:,sp]*gs)@ds[sp];target=(zt[select][:,tp]*gt)@dt[tp]
        qdisc=(codes('discovery',obj,t)[:,tp]*gt)@dt[tp]
        if f.get('minimum_context_position',0):
            with np.load(checked(reference/'natural_discovery_states.npz')) as a:
                discovery_keep=np.where((a['packed_positions']%128>=f['minimum_context_position'])&(a['packed_positions']%128<127))[0]
            qdisc=qdisc[discovery_keep]
        torch.manual_seed(417);_,_,basis=torch.pca_lowrank(qdisc,q=4,center=False,niter=3)
        deltas=dict(source_group=source,target_group=target,target_best_atom=(zt[select][:,tp]*best)@dt[tp],
            target_group_rank1=(target@basis[:,:1])@basis[:,:1].T,
            target_group_rank2=(target@basis[:,:2])@basis[:,:2].T,
            target_group_shuffled=target.roll(1,0),
            source_anchor=zs[select,r['anchor'],None]*ds[r['anchor']],
            target_for_anchor=(zt[select][:,tp]*atomt)@dt[tp])
        for name,gate in group.items():
            if name.startswith('extra_gate_'):
                value=torch.as_tensor(gate,dtype=zt.dtype,device='cuda:0')
                deltas[name.removeprefix('extra_gate_')]=(zt[select][:,tp]*value)@dt[tp]
            if name.startswith('extra_raw_reader_'):
                value=torch.as_tensor(gate,dtype=zt.dtype,device='cuda:0')
                deltas[name.removeprefix('extra_raw_reader_')]=zt[select][:,tp]@value
        if f.get('wrong_group_control'):
            peers=[q for q in results if q['objective']==obj and q['source_seed']==s and q['target_seed']==t]
            peer=peers[(next(i for i,q in enumerate(peers) if q['query']==key)+1)%len(peers)]
            with np.load(checked(run/(peer['query']+'_groups.npz'))) as a:
                wp=torch.as_tensor(a['target_members'],device='cuda:0');wg=torch.as_tensor(a['target_gate'],device='cuda:0',dtype=zt.dtype)
            wrong=(zt[select][:,wp]*wg)@dt[wp]
            deltas['wrong_group_matched_norm']=wrong*(target.norm(dim=1)/wrong.norm(dim=1).clamp_min(1e-12))[:,None]
        np.savez_compressed(run/(key+'_functional_edits.npz'),calibration_indices=select.cpu().numpy(),
            packed_positions=packed[select.cpu().numpy()],strata=np.array([stratum for _,stratum in chosen]),
            **{name:val.cpu().numpy() for name,val in deltas.items()})
        for off in range(0,n,f['batch_size']):
            local=select[off:off+f['batch_size']];positions=torch.tensor(packed[local.cpu().numpy()]%128,device='cuda:0')
            ids=torch.tensor(tokens[packed[local.cpu().numpy()]//128].astype('int64'),device='cuda:0')
            clean,h,clean_future=output(ids,positions);error=float((h-hidden[local]).abs().max());max_replay=max(max_replay,error)
            assert error<f['hidden_replay_atol'],(key,error)
            packed_outputs={name:output(ids,positions,delta[off:off+len(local)]) for name,delta in deltas.items()}
            outputs={name:value[0] for name,value in packed_outputs.items()}
            teacher=outputs['source_group'];anchor_teacher=outputs['source_anchor'];source_effect=(teacher.exp()*(teacher-clean)).sum(1)
            for name,lp in outputs.items():
                kl=(teacher.exp()*(teacher-lp)).sum(1);act=(lp.exp()*(lp-clean)).sum(1)
                anchor_kl=(anchor_teacher.exp()*(anchor_teacher-lp)).sum(1)
                for j,ix in enumerate(local.cpu().tolist()):
                    pos=int(positions[j]);label=int(ids[j,pos+1])
                    row=dict(query=key,objective=obj,source_seed=s,target_seed=t,anchor=r['anchor'],
                        calibration_index=ix,packed_position=int(packed[ix]),context_block=int(packed[ix]//128),
                        stratum=chosen[off+j][1],method=name,source_kl=float(kl[j]),
                        source_effect_kl=float(source_effect[j]),effect_kl=float(act[j]),anchor_source_kl=float(anchor_kl[j]),
                        source_group_nll=-float(teacher[j,label]),nll=-float(lp[j,label]),clean_nll=-float(clean[j,label]),
                        edit_norm=float(deltas[name][off+j].norm()),source_norm=float(source[off+j].norm()))
                    if f.get('future_offset') and pos+f['future_offset']<127:
                        future_pos=pos+f['future_offset'];flabel=int(ids[j,future_pos+1])
                        teacher_future=packed_outputs['source_group'][2][j];lp_future=packed_outputs[name][2][j]
                        row.update(future_offset=f['future_offset'],source_future_kl=float((teacher_future.exp()*(teacher_future-lp_future)).sum()),
                            source_future_effect_kl=float((teacher_future.exp()*(teacher_future-clean_future[j])).sum()),
                            future_nll_error=abs(float(teacher_future[flabel]-lp_future[flabel])))
                    rows.append(row)
                    with (run/'functional.raw.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
            if off==0:
                examples=[]
                for j in range(min(3,len(local))):
                    shift=teacher[j]-clean[j];top=torch.topk(shift.abs(),10).indices.cpu().tolist()
                    probability_shift=teacher[j].exp()-clean[j].exp();mass_top=torch.topk(probability_shift.abs(),10).indices.cpu().tolist()
                    examples.append(dict(calibration_index=int(local[j]),context=tokenizer.decode(ids[j,:int(positions[j])+1].cpu().tolist()),
                        next_token=tokenizer.decode([int(ids[j,int(positions[j])+1])]),
                        packed_position=int(packed[int(local[j])]),
                        largest_source_probability_changes=[dict(token=tokenizer.decode([token]),clean_probability=float(clean[j,token].exp()),
                            source_probability=float(teacher[j,token].exp()),target_probability=float(outputs['target_group'][j,token].exp())) for token in mass_top],
                        largest_source_log_probability_changes=[dict(token=tokenizer.decode([token]),source_change=float(shift[token]),
                            target_change=float(outputs['target_group'][j,token]-clean[j,token])) for token in top]))
                write(run/(key+'_natural_examples.json'),dict(examples=examples))
        summaries={}
        for name in deltas:
            rr=[x for x in rows if x['query']==key and x['method']==name]
            summaries[name]={stratum:dict(rows=len(v),source_kl=float(np.mean([x['source_kl'] for x in v])),
                source_effect_kl=float(np.mean([x['source_effect_kl'] for x in v])),
                pooled_relative_kl=float(sum(x['source_kl'] for x in v)/max(sum(x['source_effect_kl'] for x in v),1e-12)))
                for stratum in ['source_active','uniform'] if (v:=[x for x in rr if x['stratum']==stratum])}
        log('FUNCTIONAL_QUERY_COMPLETE',query=key,summary=summaries)
        write(run/'functional.summary.json',dict(rows=len(rows),queries=len({x['query'] for x in rows}),
            methods=list(deltas),raw_sha256=sha256(run/'functional.raw.jsonl'),max_hidden_replay=max_replay))
        if time.perf_counter()-timer>f['budget_seconds']:raise TimeoutError('Functional development budget exceeded')
    return dict(forward_sequences=forwards,forward_tokens=forwards*128,wall_seconds=time.perf_counter()-timer,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),max_hidden_replay=max_replay,rows=len(rows))
