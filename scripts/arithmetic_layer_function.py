"""Compare the same carry intervention learner at different residual sites."""
import json
import time
import numpy as np
import torch
from analyze_carry_readouts import fit_reader
from run_causalgym_multisite import ROOT, write


def fit_and_evaluate(w,cfg,model,module,tok,rows,tokenrows,hidden,generate,budget,base):
    spec=cfg['layer_function'];batch=cfg['batch_size'];device=w.device
    panel=json.loads(w.checked(ROOT/cfg['frozen_rule_evaluation']['panel']).read_text())
    freeze=json.loads(w.checked(ROOT/cfg['frozen_rule_evaluation']['freeze']).read_text())
    from ccad.artifacts import sha256
    for item in freeze['inputs']:assert sha256(w.checked(ROOT/item['path']))==item['sha256']
    write(w.run/'RULE_PANEL.json',panel)
    fit_ids=[i for i,r in enumerate(rows) if r['split']=='fit']
    assert fit_ids==list(range(405)) and all(rows[i]['template']==0 for i in fit_ids)
    x=hidden[fit_ids,0];arity=torch.tensor([int('c' in rows[i]) for i in fit_ids],device=device)
    y=torch.tensor([float(rows[i]['carry']) for i in fit_ids],device=device)
    readouts=[];slopes=[]
    for a in range(2):
        use=arity==a;xx=x[use].double().clone();yy=y[use].double().clone()
        readouts.append(fit_reader(xx.cpu().numpy(),yy.cpu().numpy(),spec['ridge'])[0])
        totals=torch.tensor([rows[i]['total'] for i in fit_ids if int('c' in rows[i])==a],device=device)
        for total in totals.unique():
            jj=totals==total;xx[jj]-=xx[jj].mean(0);yy[jj]-=yy[jj].mean()
        slopes.append((yy[:,None]*xx).sum(0)/yy.square().sum())
    reader=torch.tensor(np.array(readouts),device=device,dtype=torch.float32)
    initial=torch.stack(slopes).float();scale=initial.norm(dim=1,keepdim=True).clamp_min(1e-6)/np.sqrt(x.shape[-1])
    param=torch.nn.Parameter(initial/scale);cap=spec['norm_multiple']*initial.norm(dim=1,keepdim=True)
    optimizer=torch.optim.Adam([param],lr=spec['lr']);ix=torch.arange(batch,device=device)
    with np.load(w.checked(ROOT/spec['schedule'])) as f:schedule=f['fit_schedule']
    assert schedule.shape==(256,batch,2)
    physical=max(len(tokenrows[i]) for i in fit_ids)+cfg['max_new_tokens']

    def objective(pairs,value):
        ii,jj=torch.tensor(pairs,device=device).T
        assert bool((arity[ii]==arity[jj]).all())
        tokens=torch.full((batch,physical),tok.eos_token_id,device=device,dtype=torch.long)
        mask=torch.zeros_like(tokens);positions=[];targets=[]
        for b,(i,j) in enumerate(pairs):
            expected=rows[i]['total']+10*(int(rows[j]['carry'])-int(rows[i]['carry']))
            answer=tok.encode(str(expected),add_special_tokens=False)
            assert 10<=expected<100 and len(answer)==2
            seq=tokenrows[i]+answer;tokens[b,:len(seq)]=torch.tensor(seq,device=device);mask[b,:len(seq)]=1
            positions.append(len(tokenrows[i])-1);targets.append(answer+tok.encode('\n',add_special_tokens=False))
        pos=torch.tensor(positions,device=device);target=torch.tensor(targets,device=device)
        def hook(_m,_a,out):
            h=out[0] if isinstance(out,tuple) else out;query=((x[jj]-h[ix,pos])*reader[arity[ii]]).sum(-1,keepdim=True)
            hh=h.clone();hh[ix,pos]+=query*(value*scale)[arity[ii]]
            return (hh,)+out[1:] if isinstance(out,tuple) else hh
        handle=module.register_forward_hook(hook)
        try:
            logits=model(tokens,attention_mask=mask,use_cache=False).logits[ix[:,None],pos[:,None]+torch.arange(3,device=device)]
            w.sequence_forwards+=batch;w.token_forwards+=batch*physical
            return torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.flatten())
        finally:handle.remove()

    started=time.perf_counter();trace=[]
    with torch.enable_grad():
        diagnostics=[]
        for precision in spec['gradient_precisions']:
            torch.set_float32_matmul_precision(precision)
            loss=objective(schedule[0],param);grad=torch.autograd.grad(loss,param)[0]
            coord=int(grad.flatten().abs().argmax())
            coordinate=torch.zeros_like(param);coordinate.flatten()[coord]=1
            direction=grad/grad.norm().clamp_min(1e-12)
            for name,vector,eps in [('coordinate',coordinate,.01),('coordinate',coordinate,.1),('direction',direction,.1)]:
                analytic=float((grad*vector).sum())
                plus=(param.detach()+eps*vector).requires_grad_(True)
                minus=(param.detach()-eps*vector).requires_grad_(True)
                high_loss=float(objective(schedule[0],plus).detach())
                low_loss=float(objective(schedule[0],minus).detach())
                finite=(high_loss-low_loss)/(2*eps)
                diagnostics.append(dict(precision=precision,direction=name,coordinate=coord,epsilon=eps,
                    base_loss=float(loss.detach()),plus_loss=high_loss,minus_loss=low_loss,
                    analytic=analytic,finite=finite,error=abs(finite-analytic)))
        torch.set_float32_matmul_precision(cfg['matmul_precision'])
        write(w.run/'GRADIENT_DIAGNOSTIC.json',diagnostics)
        selected=[d for d in diagnostics if d['precision']==cfg['matmul_precision']]
        assert len(selected)==3 and all(d['error']<.0002+.03*abs(d['analytic']) for d in selected),selected
        for step,pairs in enumerate(schedule):
            optimizer.zero_grad(set_to_none=True);loss=objective(pairs,param)
            assert bool(torch.isfinite(loss));loss.backward();optimizer.step()
            with torch.no_grad():param.mul_((cap/(param*scale).norm(dim=1,keepdim=True).clamp_min(1e-8)).clamp_max(1))
            if (step+1)%64==0:
                trace.append(dict(step=step+1,loss=float(loss.detach())));w.progress('LAYER_FUNCTION_FIT',**trace[-1])
            budget()
    writer=(param.detach()*scale)
    np.savez_compressed(w.run/'LAYER_FUNCTION.npz',reader=reader.cpu().numpy(),initial=initial.cpu().numpy(),writer=writer.cpu().numpy(),schedule=schedule,fit_indices=fit_ids)
    write(w.run/'LAYER_FIT.json',dict(layer=spec['layer'],gradient=selected,trace=trace,fit_seconds=time.perf_counter()-started,parameters=param.numel(),reader_strength=spec['ridge']))
    all_records=[]
    for method in ['raw_initial','raw_fitted','carry_oracle']:
        chosen=initial if method=='raw_initial' else writer
        for off in range(0,len(panel['pairs']),batch):
            pp=panel['pairs'][off:off+batch];ii=[p['recipient'] for p in pp];jj=[p['donor'] for p in pp]
            aa=torch.tensor([int('c' in rows[i]) for i in ii],device=device)
            roles=torch.tensor([0 if rows[i]['template']==0 else 1 for i in ii],device=device)
            truth=torch.tensor([float(rows[j]['carry'])-float(rows[i]['carry']) for i,j in zip(ii,jj)],device=device)
            def patch(current,step):
                query=((hidden[jj,:step+1]-current)*reader[aa,None,:]).sum(-1,keepdim=True)
                if method=='carry_oracle':query=truth[:,None,None].expand(-1,step+1,1)
                role_mask=(torch.arange(step+1,device=device)[None,:]==roles[:,None])
                return query*chosen[aa,None,:]*role_mask[...,None]
            answers,text,_,norms=generate(ii,patch=patch)
            for k,p in enumerate(pp):
                r,d=rows[p['recipient']],rows[p['donor']];answer=answers[k];numeric=answer is not None and 10<=answer<100
                record=dict(kind='rule_intervention',task=f"template{p['template']}",row_id=p['recipient'],donor_row=p['donor'],
                    component=p['component'],mode='tens_role',method=method,seed=0,target_seed=None,layer=spec['layer'],
                    operation=p['condition'],stratum=p['stratum'],split='development',
                    recipient_question=[r[v] for v in ['a','b','c'] if v in r],donor_question=[d[v] for v in ['a','b','c'] if v in d],
                    recipient_carry=r['carry'],donor_carry=d['carry'],original_answer=r['total'],expected=p['expected'],answer=answer,generated_text=text[k],
                    correct=answer==p['expected'],unit_preserved=bool(numeric and answer%10==r['unit']),retained_original=answer==r['total'],edit_norm=float(norms[k]))
                w.record(**record);all_records.append(record)
            budget()
        w.progress('LAYER_FUNCTION_TESTED',layer=spec['layer'],method=method,profile={c:np.mean([r['correct'] for r in all_records if r['method']==method and r['operation']==c]) for c in ['same_answer_opposite_carry','same_carry_different_answer']})
    eval_ids=sorted({p[k] for p in panel['pairs'] for k in ['recipient','donor']})
    write(w.run/'RULE_RESULTS.json',dict(scope=cfg['scope'],baseline_accuracy=float(np.mean([base[i]==rows[i]['total'] for i in eval_ids])),freeze=cfg['frozen_rule_evaluation']['freeze'],primary='Complete carry change and same-carry preservation by arity; sites selected only after exposed development.',seed_semantics='seed0 is a deterministic raw-operator sentinel, not an SAE seed.'))
    w.checks.update(source_fit_only=True,actual_model_gradient=True,frozen_inputs=True,real_generation_no_answer_prefix=True,all_failed_base_cases_retained=True,no_sae_or_target_used=True)
