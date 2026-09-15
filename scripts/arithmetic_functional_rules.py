"""Test a source carry hypothesis using answer-controlled interventions."""
import hashlib
import json


def evaluate_rules(w, cfg, rows, hidden, codes, saes, generate, budget, base):
    import numpy as np
    import torch
    from run_arithmetic_digit_components import fisher
    from run_causalgym_multisite import write
    spec = cfg['functional_rule_test']
    fit = [i for i,r in enumerate(rows) if r['split']=='fit' and r['template']==0]
    dev = [i for i,r in enumerate(rows) if r['split']=='development' and r['template']==0]
    assert fit and not set(fit)&set(dev)
    hashkey = lambda value: hashlib.sha256(f"{spec['seed']}:{value}".encode()).hexdigest()
    lookup = {(r['a'],r['b'],r['template']):i for i,r in enumerate(rows)}
    used = set(); canonical = []
    for condition in ['same_answer_opposite_carry','same_carry_different_answer']:
        candidates = []
        for i in dev:
            for j in dev:
                if j<=i: continue
                r,d = rows[i],rows[j]
                if {r['a'],r['b']}&{d['a'],d['b']}: continue
                eligible = (r['total']==d['total'] and r['carry']!=d['carry'] if condition.startswith('same_answer')
                            else r['carry']==d['carry'] and r['unit']==d['unit'] and r['tens']!=d['tens'])
                if eligible and all(10<=a['total']+10*(int(b['carry'])-int(a['carry']))<100 for a,b in [(r,d),(d,r)]):
                    candidates.append((i,j))
        count = 0
        for i,j in sorted(candidates,key=hashkey):
            if i in used or j in used: continue
            used.update([i,j]);canonical.append(dict(component=f'pair{len(canonical):03}',condition=condition,recipient=i,donor=j))
            count += 1
            if count==spec['pairs_per_condition']: break
        assert count==spec['pairs_per_condition'], (condition,count)
    pairs=[]
    for p in canonical:
        for template in range(len(cfg['templates'])):
            indices=[lookup[rows[p[k]]['a'],rows[p[k]]['b'],template] for k in ['recipient','donor']]
            for i,j in [indices,indices[::-1]]:
                r,d=rows[i],rows[j]
                pairs.append(dict(component=p['component'],condition=p['condition'],template=template,
                                  recipient=i,donor=j,expected=r['total']+10*(int(d['carry'])-int(r['carry']))))
    assert all(rows[p['recipient']]['split']=='development' and rows[p['donor']]['split']=='development' for p in pairs)
    write(w.run/'RULE_PANEL.json',dict(pairs=pairs,canonical_pairs=canonical,fit_indices=fit,
          hypothesis='A carry component changes the answer by10*(donor carry-recipient carry), retaining the units digit. Same-carry controls retain the recipient answer.',
          selection='Deterministic metadata-only pair selection before any new intervention; no operand pair appears in two canonical pairs.',
          scope='Exposed original operand questions, new source counterfactual requests; not independent confirmation.'))
    groups_by_answer=[]
    for answer in sorted({rows[i]['total'] for i in fit}):
        groups=[[i for i in fit if rows[i]['total']==answer and rows[i]['carry']==flag] for flag in [False,True]]
        if all(groups):groups_by_answer.append(groups)
    assert groups_by_answer
    def contrast(x,conditioned=True):
        if conditioned:return torch.stack([x[b].mean(0)-x[a].mean(0) for a,b in groups_by_answer]).mean(0)
        return x[[i for i in fit if rows[i]['carry']]].mean(0)-x[[i for i in fit if not rows[i]['carry']]].mean(0)
    v=contrast(hidden[:,0]);v=v/v.norm().clamp_min(1e-12)
    batch=cfg['batch_size'];all_results=[]
    source_indices=None
    if spec.get('source_relation'):
        from pathlib import Path
        from run_causalgym_multisite import ROOT
        relation=spec['source_relation'];parent=ROOT/relation['from_run']
        assert json.loads(w.checked(parent/'status.json').read_text())['status']=='PASS'
        pcfg=json.loads(w.checked(parent/'config.resolved.json').read_text())
        assert all(pcfg[k]==cfg[k] for k in ['training_run','checkpoint_step','model_revision','source_cache_run'])
        with np.load(w.checked(parent/f"rule_members_seed{relation['seed']}.npz")) as data:
            source_indices=torch.tensor(data[relation['method']],device=w.device)
        write(w.run/'SOURCE_RULE_REFERENCE.json',dict(**relation,source_indices=source_indices.cpu().tolist(),
              meaning='Frozen source group validated by same-answer carry swaps and same-carry preservation; target relation fitting uses source fields only.'))
    for seed,ae in saes.items():
        if spec.get('evaluation_seeds') and seed not in spec['evaluation_seeds']:continue
        z=codes[seed];decoder=ae.decoder.weight.T;norm=decoder.norm(dim=1)
        scores=contrast(z[:,0]).abs()*norm;unconditional=contrast(z[:,0],False).abs()*norm
        answer_score=fisher(z[fit,0],[rows[i]['tens'] for i in fit])
        order=torch.argsort(scores,descending=True,stable=True)
        methods={f'conditional_carry_{k}':order[:k] for k in spec['member_counts']}
        methods['unconditional_carry_64']=torch.argsort(unconditional,descending=True,stable=True)[:64]
        methods['answer_tens_64']=torch.argsort(answer_score,descending=True,stable=True)[:64]
        active=torch.argsort(z[fit,0].square().mean(0).sqrt()*norm,descending=True,stable=True)[:512]
        perm=torch.randperm(len(active),generator=torch.Generator().manual_seed(spec['seed']))[:64].to(w.device)
        methods['random_active_64']=active[perm]
        member_weights={}
        if spec.get('fit_participation'):
            from ccad.semantic_participation import project_sparse_gates
            fspec=spec['fit_participation'];generator=torch.Generator().manual_seed(spec['seed'])
            fit_pairs=torch.cartesian_prod(torch.tensor(fit),torch.tensor(fit))
            fit_pairs=fit_pairs[fit_pairs[:,0]!=fit_pairs[:,1]]
            fit_pairs=fit_pairs[torch.randperm(len(fit_pairs),generator=generator)[:fspec['pairs']]].to(w.device)
            a,b=fit_pairs.T;full_x=z[b,0]-z[a,0]
            targets={'field':((hidden[b,0]-hidden[a,0])@v)[:,None]*v}
            if source_indices is not None:
                source_seed=spec['source_relation']['seed'];source_decoder=saes[source_seed].decoder.weight.T[source_indices]
                source_z=codes[source_seed]
                targets['transferred']=(source_z[b,0][:,source_indices]-source_z[a,0][:,source_indices])@source_decoder
                from scipy.optimize import linear_sum_assignment
                cosine=(source_decoder/source_decoder.norm(dim=1,keepdim=True))@(decoder/decoder.norm(dim=1,keepdim=True)).T
                src,assigned=linear_sum_assignment(-cosine.cpu().numpy())
                assert np.array_equal(src,np.arange(len(source_indices)))
                methods['assignment_64']=torch.tensor(assigned,device=w.device)
            for tag,target in targets.items():
                influence=(full_x*(target@decoder.T)).mean(0)
                bank=torch.argsort(influence,descending=True,stable=True)[:fspec['pool']]
                x=full_x[:,bank];d=decoder[bank];scale=target.square().sum().clamp_min(1e-10)
                methods[tag+'_ranked_binary_64']=bank[:64]
                with torch.enable_grad():
                    g=torch.zeros(len(bank),device=w.device);g[:64]=.5;g.requires_grad_()
                    optimizer=torch.optim.Adam([g],lr=fspec['lr']);trace=[]
                    for update in range(fspec['steps']):
                        optimizer.zero_grad();loss=((x*g)@d-target).square().sum()/scale
                        loss.backward();optimizer.step();project_sparse_gates(g,64)
                        if update%32==0 or update==fspec['steps']-1:trace.append([update+1,float(loss.detach())])
                selected=torch.where(g.detach()>0)[0];assert len(selected)<=64
                methods[tag+'_fitted_weighted_64']=bank[selected];member_weights[tag+'_fitted_weighted_64']=g.detach()[selected]
                methods[tag+'_fitted_binary_64']=bank[selected]
                suffix='' if tag=='field' else '_'+tag
                np.savez_compressed(w.run/f'rule_fit_seed{seed}{suffix}.npz',fit_pairs=fit_pairs.cpu().numpy(),bank=bank.cpu().numpy(),weights=g.detach().cpu().numpy(),influence=influence.cpu().numpy())
                write(w.run/f'RULE_FIT_s{seed}{suffix}.json',dict(fspec=fspec,trace=trace,active=int(len(selected)),
                    target=('Raw carry projection' if tag=='field' else 'Frozen source1 member field')+' on the same source-fit pairs; no new model outputs or gradients.',
                    fit_operand_rows=fit,relative_field_mse=float((((x*g.detach())@d-target).square().sum()/scale))))
                w.progress('SOURCE_RULE_FITTED',seed=seed,teacher=tag,active=int(len(selected)),final_loss=trace[-1][1]);budget()
        payload={name:ids.cpu().numpy() for name,ids in methods.items()}
        payload.update({name+'_weights':value.cpu().numpy() for name,value in member_weights.items()})
        np.savez_compressed(w.run/f'rule_members_seed{seed}.npz',conditional_score=scores.cpu().numpy(),
                            unconditional_score=unconditional.cpu().numpy(),answer_score=answer_score.cpu().numpy(),
                            raw_direction=v.cpu().numpy(),**payload)
        for name,indices in [('noop',None),('whole_state',None),('raw_carry_direction',None)]+list(methods.items()):
            if spec.get('evaluation_methods') and name not in spec['evaluation_methods']:continue
            records=[]
            for off in range(0,len(pairs),batch):
                pp=pairs[off:off+batch];ii=[p['recipient'] for p in pp];di=[p['donor'] for p in pp]
                tens_site=torch.tensor([p['template'] for p in pp],device=w.device)
                def patch(current,step):
                    at_tens=(torch.arange(step+1,device=w.device)[None,:]==tens_site[:,None])
                    if name=='whole_state':change=hidden[di,:step+1]-current
                    elif name=='raw_carry_direction':
                        difference=hidden[di,:step+1]-current;change=(difference@v)[...,None]*v
                    else:
                        current_codes=ae.encode(current.flatten(0,1)).reshape(len(ii),step+1,-1)
                        coeff=z[di,:step+1][...,indices]-current_codes[...,indices]
                        if name in member_weights:coeff=coeff*member_weights[name]
                        change=coeff@decoder[indices]
                    return change*at_tens[...,None]
                ans,text,replay,editnorm=generate(ii,patch=None if name=='noop' else patch)
                if name=='noop':assert ans==[base[i] for i in ii], 'Cached baseline differs from replay'
                for j,p in enumerate(pp):
                    r,d=rows[p['recipient']],rows[p['donor']];numeric=ans[j] is not None and 10<=ans[j]<100
                    record=dict(kind='rule_intervention',task=f"template{p['template']}",row_id=p['recipient'],
                        component=p['component'],mode='tens_role',method=name,seed=seed,target_seed=None,
                        operation=p['condition'],split='development',donor_row=p['donor'],
                        recipient_question=[r['a'],r['b']],donor_question=[d['a'],d['b']],
                        recipient_carry=r['carry'],donor_carry=d['carry'],original_answer=r['total'],
                        donor_original_answer=d['total'],expected=p['expected'],answer=ans[j],generated_text=text[j],
                        correct=ans[j]==p['expected'],retained_original=ans[j]==r['total'],
                        unit_preserved=bool(numeric and ans[j]%10==r['unit']),
                        signed_tens_change=((ans[j]//10-r['tens'])*(int(d['carry'])-int(r['carry'])) if numeric else None),
                        edit_norm=float(editnorm[j]))
                    w.record(**record);records.append(record)
                budget()
            profile={condition:dict(success=float(np.mean([r['correct'] for r in records if r['operation']==condition])),
                         unit_preserved=float(np.mean([r['unit_preserved'] for r in records if r['operation']==condition])))
                     for condition in ['same_answer_opposite_carry','same_carry_different_answer']}
            all_results.append(dict(seed=seed,method=name,profile=profile));w.progress('SOURCE_RULE_TESTED',seed=seed,method=name,profile=profile)
    write(w.run/'RULE_RESULTS.json',dict(results=all_results,fit_rows=len(fit),matched_fit_totals=len(groups_by_answer),
          source_selection='Equal-weight carry contrasts within answer totals; source-only cached states, no intervention-outcome selection.',
          operation='Donor source-code replacement at the tens prediction role, preserving reconstruction residual; later digits freely generated.',
          comparison='Raw rank1 uses the same answer-controlled contrast in hidden space. Full-state donor shares answer in the changing-carry test.',
          overall_mechanism_claim='Hypothesis test; membership correlation alone is not evidence of a carry mechanism.'))
    w.checks['source_only_selection']=True;w.checks['paired_question_disjointness']=True;w.checks['cached_baseline_replay']=True
