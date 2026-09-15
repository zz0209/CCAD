"""Select members of a fixed role-update bank using reusable response information."""
import json,time
import numpy as np
from run_causalgym_multisite import write,ROOT
from ccad.artifacts import sha256


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
