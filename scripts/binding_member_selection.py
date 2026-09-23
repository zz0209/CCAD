"""Select members of a fixed role-update bank using reusable response information."""
import json,time
import numpy as np
from run_causalgym_multisite import write,ROOT
from ccad.artifacts import sha256


def composition_fields(torch, ids, rows, codecs, means, cache, members=64):
    device=means.device
    roles=torch.tensor([0,1,2,0,1,2],device=device)
    sign=torch.tensor([1,1,1,-1,-1,-1],device=device)
    masks={'entity':[1,1,0,1,1,0],'attribute':[0,0,1,0,0,1]}
    positions=torch.tensor([rows[i]['positions'] for i in ids],device=device)
    index=torch.arange(len(ids),device=device)
    fields={family:{part:{} for part in masks} for family in ['raw','source','target']}
    codes={family:{part:{} for part in masks} for family in ['source','target']}
    def retain(value,decoder):
        selected=(value.abs()*decoder.norm(dim=1)).topk(members,dim=-1).indices
        return torch.zeros_like(value).scatter(-1,selected,value.gather(-1,selected))
    with torch.no_grad():
        for layer in range(len(means)):
            for part,mask in masks.items():
                delta=(means[layer,roles]*sign[:,None]*torch.tensor(mask,device=device)[:,None])[None].expand(len(ids),-1,-1)
                for family in fields:fields[family][part][layer]=delta
            if layer not in codecs:continue
            asset=codecs[layer]
            hidden=cache[layer][index[:,None],positions]
            zs=asset['s'].encode(hidden.flatten(0,1)).reshape(len(ids),6,-1)
            zt=asset['t'].encode(hidden.flatten(0,1)).reshape(len(ids),6,-1)
            for part in masks:
                raw=fields['raw'][part][layer]
                cs=retain(asset['s'].encode((hidden+raw).flatten(0,1)).reshape_as(zs)-zs,asset['ds'])
                source=cs@asset['ds']
                ct=retain(asset['t'].encode((hidden+source).flatten(0,1)).reshape_as(zt)-zt,asset['dt'])
                fields['source'][part][layer]=source
                fields['target'][part][layer]=ct@asset['dt']
                codes['source'][part][layer]=(cs,zs)
                codes['target'][part][layer]=(ct,zt)
        for family in codes:
            codes[family]['both']={layer:(codes[family]['entity'][layer][0]+codes[family]['attribute'][layer][0],
                                         codes[family]['entity'][layer][1]) for layer in codecs}
    return fields,codes


def evaluate_composition(torch, batch_data, ids, rows, fields, forward, control, gradient=False):
    layers=sorted(fields['entity'])
    states={'none':{l:torch.zeros_like(fields['entity'][l]) for l in layers},
            'entity':fields['entity'],'attribute':fields['attribute'],
            'both':{l:fields['entity'][l]+fields['attribute'][l] for l in layers}}
    device=states['none'][layers[0]].device
    index=torch.arange(len(ids),device=device)
    original=torch.tensor([rows[i]['answer_id'] for i in ids],device=device)
    swapped=torch.tensor([rows[i]['swap_answer_id'] for i in ids],device=device)
    responses={}
    for name,delta in states.items():
        control(dict(name='composition_request',layers=layers,role_fields=delta),'both')
        lp=forward(batch_data,gradient=gradient)
        responses[name]=dict(margin=lp[index,original]-lp[index,swapped],
                             original_log_probability=lp[index,original],
                             swap_log_probability=lp[index,swapped],prediction=lp.argmax(-1))
    return responses,states


def record_composition(c,w,torch,model,rows,codecs,means,cache,batch,forward,control):
    cfg=c['composition_record'];members=cfg.get('candidate_members',64)
    assert codecs and members==64
    families=cfg.get('families',['raw','source','target'])
    assert families and len(set(families))==len(families) and set(families)<={'raw','source','target'}
    selected=[]
    for split,limit in cfg['worlds_by_split'].items():
        worlds=sorted({r['component'] for r in rows if r['split']==split})[:limit]
        assert len(worlds)==limit and limit>0
        selected.extend(r['row_id'] for r in rows if r['split']==split and r['component'] in worlds)
    assert selected
    arrays=w.run/'composition_arrays';arrays.mkdir()
    roles=torch.tensor([0,1,2,0,1,2],device=w.device)
    sign=torch.tensor([1,1,1,-1,-1,-1],device=w.device)
    background=means[:,roles]*sign[None,:,None]
    np.savez_compressed(arrays/'raw_background.npz',delta=background.cpu().numpy(),
                        entity_mask=np.array([1,1,0,1,1,0]),attribute_mask=np.array([0,0,1,0,0,1]))
    manifest=dict(row_ids=selected,states=['none','entity','attribute','both'],families=families,
        sae_layers=sorted(codecs),raw_background_layers=[l for l in range(len(means)) if l not in codecs],
        candidate_members=members,margin='Fixed original answer log probability minus swapped answer log probability',
        role_order=['entity0','entity0_next','attribute0','entity1','entity1_next','attribute1'],
        coefficient_axes=['row','role','candidate'],delta_axes=['state','row','role','hidden'],
        state_order=['entity','attribute','both'],zero_state='All hidden updates zero',
        joint='Execute the saved entity plus attribute updates without re-encoding or capacity adjustment',
        support='Padded support is -1 with coefficient zero; baseline_code uses the same support',
        context='Restore original context before each block; the query stream remains live',
        source_seed=c['source_seed'],target_seed=c['target_seed'],batches=[])
    write(w.run/'composition_protocol.json',manifest)
    max_noop=0.;max_joint=0.
    for offset in range(0,len(selected),c['batch_size']):
        ids=selected[offset:offset+c['batch_size']];b=batch(ids)
        control(capture_states=True);clean=forward(b)
        fields,codes=composition_fields(torch,ids,rows,codecs,means,cache,members)
        payload=dict(row_ids=np.array(ids),components=np.array([rows[i]['component'] for i in ids]),
                     positions=np.array([rows[i]['positions'] for i in ids]),
                     answer_id=np.array([rows[i]['answer_id'] for i in ids]),
                     swap_answer_id=np.array([rows[i]['swap_answer_id'] for i in ids]))
        index=torch.arange(len(ids),device=w.device)
        expected=torch.tensor(payload['answer_id'],device=w.device)
        swapped=torch.tensor(payload['swap_answer_id'],device=w.device)
        clean_margin=clean[index,expected]-clean[index,swapped]
        for family in families:
            responses,states=evaluate_composition(torch,b,ids,rows,fields[family],forward,control)
            max_noop=max(max_noop,float((responses['none']['margin']-clean_margin).abs().max()))
            interaction=responses['both']['margin']-responses['entity']['margin']-responses['attribute']['margin']+responses['none']['margin']
            payload[family+'_margin']=torch.stack([responses[s]['margin'] for s in manifest['states']]).cpu().numpy()
            for layer in codecs:
                delta=torch.stack([states[s][layer] for s in manifest['state_order']])
                max_joint=max(max_joint,float((delta[2]-delta[0]-delta[1]).abs().max()))
                payload[f'{family}_pre{layer}_delta']=delta.cpu().numpy()
            for state,response in responses.items():
                for j,i in enumerate(ids):
                    row=rows[i]
                    w.record(kind='conditional_composition',row_id=i,component=row['component'],split=row['split'],
                        template=row['template'],order=row['order'],query=row['query'],method=family,operation=state,
                        seed=c['source_seed'],target_seed=c['target_seed'],margin=float(response['margin'][j]),
                        original_log_probability=float(response['original_log_probability'][j]),
                        swap_log_probability=float(response['swap_log_probability'][j]),
                        original_answer_id=row['answer_id'],swap_answer_id=row['swap_answer_id'],
                        prediction=int(response['prediction'][j]),interaction=float(interaction[j]),
                        edit_norm=float(sum(v[j].square().sum() for v in states[state].values()).sqrt()))
        for family,parts in codes.items():
            for part,layers in parts.items():
                for layer,(coeff,base) in layers.items():
                    support=coeff.abs().topk(members,dim=-1).indices
                    assert int((coeff!=0).sum(-1).max())<=members
                    values=coeff.gather(-1,support);baseline=base.gather(-1,support)
                    prefix=f'{family}_{part}_pre{layer}'
                    payload[prefix+'_support']=support.masked_fill(values==0,-1).cpu().numpy()
                    payload[prefix+'_coefficients']=values.cpu().numpy()
                    payload[prefix+'_baseline_code']=baseline.masked_fill(values==0,0).cpu().numpy()
        path=arrays/f'batch_{offset:06d}.npz';np.savez_compressed(path,**payload)
        manifest['batches'].append(dict(path=str(path.relative_to(w.run)),row_ids=ids,bytes=path.stat().st_size,sha256=sha256(path)))
        write(w.run/'composition_protocol.json',manifest)
        w.progress('COMPOSITION_RECORD',completed_rows=offset+len(ids),total_rows=len(selected),families=families)
    write(w.run/'composition_checks.json',dict(noop_margin_max_error=max_noop,joint_update_max_error=max_joint,
        rows=len(selected),response_forwards_per_row=4*len(families),capture_forwards_per_row=1))
    assert max_noop<1e-5 and max_joint<1e-6
    w.checks.update(fixed_answer_margin=True,explicit_both_parts=True,composition_noop=True,additive_joint_update=True)


def run_selection(c,w,torch,model,rows,codecs,means,cache,batch,forward,control):
    cfg=c['member_selection'];device=w.device;layers=sorted(codecs);es=[0,1,3,4]
    roles=torch.tensor([0,1,2,0,1,2],device=device);sign=torch.tensor([1,1,1,-1,-1,-1],device=device)
    em=torch.tensor([1,1,0,1,1,0],device=device);full_layers=list(range(model.config.num_hidden_layers))
    calls={'source':0,'raw':0,'target':0,'finite_difference':0,'evaluation':0};native_max={}
    def sparse(z,d,k,score=None):
        values=z.abs()*d.norm(dim=1) if score is None else score
        values=values.masked_fill(z==0,-torch.inf)
        idx=values.topk(k,dim=-1).indices
        return torch.zeros_like(z).scatter(-1,idx,z.gather(-1,idx))
    def state_fields(ids):
        ix=torch.arange(len(ids),device=device);pos=torch.tensor([rows[i]['positions'] for i in ids],device=device)
        out={}
        with torch.no_grad():
            for l,a in codecs.items():
                h=cache[l][ix[:,None],pos];b=(means[l,roles]*sign[:,None]*em[:,None])[None].expand_as(h)
                zs=a['s'].encode(h.flatten(0,1)).reshape(len(ids),6,-1)
                cs=sparse(a['s'].encode((h+b).flatten(0,1)).reshape_as(zs)-zs,a['ds'],64)
                vs=cs@a['ds'];zt=a['t'].encode(h.flatten(0,1)).reshape(len(ids),6,-1)
                ct=sparse(a['t'].encode((h+vs).flatten(0,1)).reshape_as(zt)-zt,a['dt'],64)
                out[l]=dict(source=vs,raw=b,target=ct@a['dt'],cs=cs,ct=ct)
        return out
    def evaluate(b,fields,op,gradient=False):
        control(dict(name='member_request',layers=full_layers,entity_fields=fields),op)
        return forward(b,gradient=gradient)
    def margin(lp,ids,op):
        correct=torch.tensor([rows[i]['answer_id'] if op=='both' else rows[i]['swap_answer_id'] for i in ids],device=device)
        wrong=torch.tensor([rows[i]['swap_answer_id'] if op=='both' else rows[i]['answer_id'] for i in ids],device=device)
        return lp[torch.arange(len(ids),device=device),correct]-lp[torch.arange(len(ids),device=device),wrong]
    fit_worlds=sorted({r['component'] for r in rows if r['split']=='fit'})[:cfg['fit_worlds']]
    fit=[r['row_id'] for r in rows if r['component'] in fit_worlds and r['split']=='fit']
    bank={family:{op:{l:torch.zeros(6,model.config.hidden_size,device=device) for l in layers} for op in ['entity','both']} for family in ['source_cached','source_fullgrad','source_path','raw_path','target_path']}
    count=0;fd=None;bs=c['batch_size'];alphas=cfg['alphas'];weights=cfg['weights']
    frozen=cfg.get('frozen_bank')
    if frozen:
        path=w.checked(ROOT/frozen);assert sha256(path)==cfg['frozen_bank_sha256']
        saved=np.load(path)
        for family in bank:
            if family=='target_path':continue
            for op in bank[family]:
                for l in layers:bank[family][op][l]=torch.tensor(saved[f'{family}_{op}_pre{l}'],device=device)
        check=w.checked(ROOT/cfg['frozen_gradient_check']);assert sha256(check)==cfg['frozen_gradient_check_sha256']
        fd=json.loads(check.read_text());write(w.run/'gradient_check.json',dict(fd,reused_from=str(check)))
    for offset in range(0,len(fit),bs):
        ids=fit[offset:offset+bs];b=batch(ids);control(capture_states=True);forward(b);fields=state_fields(ids)
        for op in ['entity','both']:
            g0=None
            for family in (['target'] if frozen else ['source','raw','target']):
                for ai,(alpha,weight) in enumerate(zip(alphas,weights)):
                    if frozen and ai==0:continue
                    if ai==0 and g0 is not None:grads=g0
                    else:
                        leaf={l:(fields[l][family]*alpha).detach().requires_grad_(True) for l in layers}
                        lp=evaluate(b,leaf,op,True);m=margin(lp,ids,op)
                        grads=torch.autograd.grad(m.sum(),list(leaf.values()));calls[family]+=1
                        grads={l:g.detach() for l,g in zip(layers,grads)}
                        assert all(torch.isfinite(g).all() for g in grads.values())
                        if ai==0:g0=grads
                        if fd is None and family=='source' and op=='entity' and alpha==.5:
                            eps=.01;vp=evaluate(b,{l:fields[l][family]*(alpha+eps) for l in layers},op)
                            vm=evaluate(b,{l:fields[l][family]*(alpha-eps) for l in layers},op);calls['finite_difference']+=2
                            numerical=float((margin(vp,ids,op)-margin(vm,ids,op)).mean()/(2*eps))
                            analytical=float(sum((grads[l]*fields[l][family]).sum() for l in layers)/len(ids))
                            fd=dict(analytical=analytical,numerical=numerical,absolute_error=abs(analytical-numerical),epsilon=eps)
                            write(w.run/'gradient_check.json',fd)
                            assert abs(analytical-numerical)<.03*max(abs(analytical),abs(numerical))+.001,fd
                    for l in layers:bank[family+'_path'][op][l]+=weight*grads[l].sum(0)
                    if family=='source':
                        if ai==0:
                            for l in layers:bank['source_cached'][op][l]+=grads[l].sum(0)
                        if ai==len(alphas)-1:
                            for l in layers:bank['source_fullgrad'][op][l]+=grads[l].sum(0)
        count+=len(ids)
        if offset%(8*bs)==0:w.progress('MEMBER_RESPONSE_BANK',fit_rows=count,total_rows=len(fit),gradient_calls=calls)
    for family in (['target_path'] if frozen else bank):
        for op in bank[family]:
            for l in layers:bank[family][op][l]/=count
    if frozen:
        for op in bank['target_path']:
            for l in layers:bank['target_path'][op][l]+=weights[0]*bank['source_cached'][op][l]
    bank['source_pooled']={op:{l:(bank['source_path']['entity'][l]+bank['source_path']['both'][l])/2 for l in layers} for op in ['entity','both']}
    np.savez_compressed(w.run/'response_bank.npz',**{f'{family}_{op}_pre{l}':v.cpu().numpy() for family,ops in bank.items() for op,ll in ops.items() for l,v in ll.items()})
    write(w.run/'response_protocol.json',dict(fit_rows=count,fit_worlds=fit_worlds,alphas=alphas,weights=weights,
        rule='Average logit-margin derivatives by layer and semantic token role over both queried entities. Source and raw banks reusable; target bank uses the same number of path samples on actual target code updates. Alpha0 is shared exactly.',
        candidate='Same64 source-mediated target code differences per state. Select8/16/32 by signed coefficient times decoder-gradient projection, or geometric magnitude. No evaluation output enters selection.',
        prediction='Integral gradient bank projects each unseen partial update. Predicts finite margin gain from the no-entity-update baseline; linear approximation evaluated against actual nonlinear continuation.',
        composition='Source entity and full entity-plus-attribute paths are both observed in fit. Held-out tests are sparse member requests on different worlds, not unobserved full joint operations.',
        source_bank_reused=frozen,source_target_response_cost='With frozen bank, source/raw/cached selectors use zero new gradients; only the direct target comparator computes256 path-gradient batches. Its alpha0 contribution is exactly reusable from the source bank.',calls=dict(calls)))
    eval_ids=[r['row_id'] for r in rows if r['split']=='development'];methods=['geometry','source_cached','source_fullgrad','source_path','source_pooled','raw_path','target_path']
    for offset in range(0,len(eval_ids),bs):
        ids=eval_ids[offset:offset+bs];b=batch(ids);control(capture_states=True);base=forward(b);fields=state_fields(ids)
        for j,i in enumerate(ids):w.record(kind='baseline',row_id=i,component=rows[i]['component'],split='development',method='baseline',operation='none',seed=0,correct=bool(base[j].argmax()==rows[i]['answer_id']))
        for op in ['entity','both']:
            none={l:torch.zeros_like(fields[l]['source']) for l in layers};lp0=evaluate(b,none,op);m0=margin(lp0,ids,op);calls['evaluation']+=1
            specs=[('source_full',64,'source'),('target_full',64,'target')]+[(m,k,'target') for m in methods for k in cfg['budgets']]+[('source_selected',k,'source') for k in cfg['budgets']]
            for name,k,kind in specs:
                delta={};predicted=torch.zeros(len(ids),device=device);selected={}
                for l,a in codecs.items():
                    coeff=fields[l]['cs' if kind=='source' else 'ct'];d=a['ds' if kind=='source' else 'dt']
                    if name.endswith('_full'):cc=coeff
                    elif name=='geometry':cc=sparse(coeff,d,k)
                    else:
                        family='source_path' if name=='source_selected' else name
                        scores=coeff*(bank[family][op][l]@d.T)[None]
                        cc=sparse(coeff,d,k,scores)
                        predicted+=(cc*(bank[family][op][l]@d.T)[None]).sum((1,2))
                    delta[l]=cc@d;selected[l]=cc
                    native_max[name]=max(native_max.get(name,0),int((cc!=0).sum(-1).max()))
                lp=evaluate(b,delta,op);calls['evaluation']+=1;m=margin(lp,ids,op)
                for j,i in enumerate(ids):
                    r=rows[i];wanted=r['answer_id'] if op=='both' else r['swap_answer_id'];p=int(lp[j].argmax())
                    w.record(kind='member_selection',row_id=i,component=r['component'],split='development',template=r['template'],order=r['order'],query=r['query'],
                        method=name,mode=f'k{k}',members=k,operation=op,seed=c['source_seed'],target_seed=c['target_seed'],answer_id=p,expected_id=wanted,correct=p==wanted,
                        margin=float(m[j]),margin_without_entity_part=float(m0[j]),actual_margin_gain=float(m[j]-m0[j]),
                        predicted_margin_gain=float(predicted[j]) if name in bank or name=='source_selected' else None,
                        edit_norm=float(sum(v[j].square().sum() for v in delta.values()).sqrt()),actual_members=max(int((v[j]!=0).sum(-1).max()) for v in selected.values()))
        w.progress('MEMBER_CHOICE_EVALUATION',evaluated_rows=offset+len(ids),total_rows=len(eval_ids),gradient_calls=calls)
    write(w.run/'selection_cost.json',dict(calls=calls,max_members_by_method=native_max))
    w.checks.update(source_fit_worlds_only=all(rows[i]['split']=='fit' for i in fit),finite_difference_pass=fd is not None,
        allowed_member_budgets=all(x['actual_members']<=x['members'] for x in w.metrics if x['kind']=='member_selection'),response_fit_complete=count==len(fit))
