"""Source-only arithmetic component discovery from activation patterns.

Feature groups are selected by a known addition-table rule, without model
gradients or output scores. Unseen operand pairs and prompt forms test each
group's selectivity and actual numeric-output effect before target fitting.
"""
import hashlib
from datetime import datetime,timezone


def run_arithmetic_discovery(cfg,w,model,tokenizer,module,saes,D,write,log,budget):
    import numpy as np
    import torch
    from ccad.artifacts import sha256
    rows=[]
    for tid,template in enumerate(cfg['arithmetic_templates']):
        for a in range(cfg['operand_limit']):
            for b in range(cfg['operand_limit']):
                if [a,b] in cfg.get('arithmetic_excluded_pairs',[]):continue
                held=int(hashlib.sha256(f'arithmetic-r32:{a}:{b}'.encode()).hexdigest()[:8],16)%3==0
                prompt=template.format(a=a,b=b)
                rows.append(dict(template=tid,a=a,b=b,total=a+b,prompt=prompt,split='fit' if tid==0 and not held else 'evaluation',held_operand_pair=held))
    write(w.run/'arithmetic_panel.json',dict(rows=rows,scope=cfg['scope']))
    tokenrows=[[tokenizer.eos_token_id]+tokenizer.encode(r['prompt']) for r in rows]
    assert max(map(len,tokenrows))<=cfg['max_length']
    maxsum=2*(cfg['operand_limit']-1)
    answers=[]
    for total in range(maxsum+1):
        tok={tokenizer.encode(prefix+str(total))[0] for prefix in ['', ' '] if len(tokenizer.encode(prefix+str(total)))==1}
        assert tok;answers.append(sorted(tok))
    assert len(set(t for aa in answers for t in aa))==sum(map(len,answers))
    write(w.run/'answer_tokens.json',dict(tokens=answers,scoring='For each possible sum, logsumexp of its single-token spellings with or without a preceding space; no fitted calibration. Full-vocabulary accuracy also retained.'))
    batch=cfg['arithmetic_batch'];all_h=[];all_scores=[];all_predictions=[]
    def forward(indices,delta=None):
        ids=torch.full((batch,cfg['max_length']),tokenizer.eos_token_id,device=w.device,dtype=torch.long)
        pos=[]
        padded=indices+[indices[0]]*(batch-len(indices))
        for j,i in enumerate(padded):
            tok=tokenrows[i];ids[j,:len(tok)]=torch.tensor(tok,device=w.device);pos.append(len(tok)-1)
        ix=torch.arange(batch,device=w.device);pos=torch.tensor(pos,device=w.device);cache={}
        def hook(m,args,out):
            h=out[0] if isinstance(out,tuple) else out;cache['h']=h[ix,pos].detach()
            if delta is None:return out
            hh=h.clone();hh[ix[:len(indices)],pos[:len(indices)]]-=delta
            return (hh,)+out[1:] if isinstance(out,tuple) else hh
        handle=module.register_forward_hook(hook)
        try:
            with torch.no_grad():logits=model(ids,use_cache=False).logits[ix,pos]
        finally:handle.remove()
        w.sequence_forwards+=batch;w.token_forwards+=ids.numel()
        scores=torch.stack([logits[:,aa].double().logsumexp(-1) for aa in answers],1)
        return scores[:len(indices)],cache['h'][:len(indices)],logits[:len(indices)].argmax(-1)
    for off in range(0,len(rows),batch):
        score,h,pred=forward(list(range(off,min(off+batch,len(rows)))));all_h.append(h);all_scores.append(score);all_predictions.append(pred);budget()
    h=torch.cat(all_h);scores=torch.cat(all_scores);vpred=torch.cat(all_predictions)
    quality=[]
    for tid in range(len(cfg['arithmetic_templates'])):
        ii=[i for i,r in enumerate(rows) if r['template']==tid];correct=torch.tensor([rows[i]['total'] for i in ii],device=w.device)
        quality.append(dict(template=tid,n=len(ii),numeric_accuracy=float((scores[ii].argmax(1)==correct).double().mean()),
                            vocabulary_accuracy=sum(int(vpred[i]) in answers[rows[i]['total']] for i in ii)/len(ii)))
    if cfg.get('arithmetic_model_only'):
        np.savez_compressed(w.run/'arithmetic_states.npz',hidden=h.cpu().numpy(),clean_scores=scores.cpu().numpy(),vocabulary_prediction=vpred.cpu().numpy())
        for i,r in enumerate(rows):w.record(kind='arithmetic_capability',method='base_model',task='arithmetic_template_'+str(r['template']),row_id=r['a']*cfg['operand_limit']+r['b'],component=f"{r['a']}+{r['b']}",numeric_prediction=int(scores[i].argmax()),correct=r['total'],full_vocabulary_correct=int(vpred[i]) in answers[r['total']])
        write(w.run/'arithmetic_results.json',dict(quality=quality,scope=cfg['scope']))
        w.checks['all_queries_exclude_in_prompt_example_operand_pairs']=all([r['a'],r['b']] not in cfg.get('arithmetic_excluded_pairs',[]) for r in rows)
        log('ARITHMETIC_CAPABILITY_EVALUATED',quality=quality)
        return
    fitix=[i for i,r in enumerate(rows) if r['split']=='fit'];evalix=[i for i,r in enumerate(rows) if r['split']=='evaluation']
    codes={key:torch.cat([ae.encode(h[i:i+256]) for i in range(0,len(h),256)]) for key,ae in saes.items()}
    meta=[];gates={}
    for key,z in codes.items():
        obj,s=key;gate=torch.zeros((z.shape[1],len(cfg['arithmetic_concepts'])),device=w.device);fit=z[fitix].double()
        rms=fit.square().mean(0).sqrt().clamp_min(1e-12)
        for k,total in enumerate(cfg['arithmetic_concepts']):
            positive=torch.tensor([rows[i]['total']==total for i in fitix],device=w.device)
            score=(fit[positive].mean(0)-fit[~positive].mean(0))/rms
            order=torch.argsort(score,descending=True,stable=True)[:cfg['arithmetic_component_members']]
            gate[order,k]=1
            meta.append(dict(objective=obj,source=s,total=total,fit_positive=int(positive.sum()),members=order.cpu().tolist(),activation_contrast=score[order].cpu().tolist()))
        gates[key]=gate
        np.savez_compressed(w.run/f'{obj}_s{s}_arithmetic_source.npz',gate=gate.cpu().numpy(),codes=z.cpu().numpy())
    np.savez_compressed(w.run/'arithmetic_states.npz',hidden=h.cpu().numpy(),clean_scores=scores.cpu().numpy(),vocabulary_prediction=vpred.cpu().numpy())
    write(w.run/'SOURCE_FREEZE.json',dict(written_at_utc=datetime.now(timezone.utc).isoformat(),components=meta,
        functional_outputs_used_for_member_selection=False,gradients=0,target_dictionaries_consulted=0,
        clean_outputs_already_computed=True,files=[dict(path=p.name,sha256=sha256(p)) for p in w.run.glob('*_source.npz')]))
    effects=[]
    for key,z in codes.items():
        obj,s=key
        for k,total in enumerate(cfg['arithmetic_concepts']):
            delta=(z*gates[key][:,k])@D[key];edited=[]
            for off in range(0,len(evalix),batch):
                ii=evalix[off:off+batch];out,hh,_=forward(ii,delta[ii]);assert float((hh-h[ii]).abs().max())<cfg['hidden_atol'];edited.append(out)
                for j,i in enumerate(ii):
                    r=rows[i];others=[c for c in range(maxsum+1) if c!=total]
                    cleanmargin=scores[i,total]-scores[i,others].logsumexp(-1)
                    editmargin=out[j,total]-out[j,others].logsumexp(-1)
                    w.record(kind='arithmetic_source',objective=obj,seed=s,operation=str(total),method='activation_component',
                             mode=obj+'|'+str(total),task='arithmetic_template_'+str(r['template']),row_id=r['a']*cfg['operand_limit']+r['b'],component=f"{r['a']}+{r['b']}",template=r['template'],a=r['a'],b=r['b'],total=r['total'],held_operand_pair=r['held_operand_pair'],
                             clean_numeric_prediction=int(scores[i].argmax()),edited_numeric_prediction=int(out[j].argmax()),
                             concept_margin=float(cleanmargin),edited_concept_margin=float(editmargin),
                             component_activation=float(z[i,gates[key][:,k]>0].sum()),edit_norm=float(delta[i].norm()))
                budget()
            out=torch.cat(edited)
            for tid in range(len(cfg['arithmetic_templates'])):
                for match in [True,False]:
                    jj=[j for j,i in enumerate(evalix) if rows[i]['template']==tid and (rows[i]['total']==total)==match]
                    ii=[evalix[j] for j in jj];other=[c for c in range(maxsum+1) if c!=total]
                    if not jj:continue
                    cm=scores[ii,total]-scores[ii][:,other].logsumexp(-1);em=out[jj,total]-out[jj][:,other].logsumexp(-1)
                    effects.append(dict(objective=obj,source=s,concept=total,template=tid,matches_concept=match,n=len(jj),
                        concept_margin_decrement=float((cm-em).mean()),mean_activation=float(z[ii][:,gates[key][:,k]>0].sum(1).mean()),
                        clean_accuracy=float((scores[ii].argmax(1)==torch.tensor([rows[i]['total'] for i in ii],device=w.device)).double().mean()),
                        edited_accuracy=float((out[jj].argmax(1)==torch.tensor([rows[i]['total'] for i in ii],device=w.device)).double().mean())))
            log('ARITHMETIC_COMPONENT_TESTED',objective=obj,source=s,total=total,evaluation_states=len(evalix))
    write(w.run/'arithmetic_results.json',dict(quality=quality,effects=effects,scope=cfg['scope'],components=meta))
    w.checks.update(source_selection_only_activation_and_known_sum_rule=True,no_target_dictionary_or_task_gradient=True,full_grid_and_failed_groups_retained=True)
