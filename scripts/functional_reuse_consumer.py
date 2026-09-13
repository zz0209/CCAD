"""Reuse source functional explanations to select target feature deletions.

All proposals select actual target members and delete their full codes.
The consumer chooses a support allowance using limited target validation,
then measures requested functional disruption and collateral disruption.
Cached source-model gradients are available to strong direct controls.
"""
import json
from collections import defaultdict
from datetime import datetime,timezone


def run_consumer(cfg,w,D,saes,capture,forward,checked,write,log,budget):
    import numpy as np,torch
    from ccad.artifacts import sha256
    parent=checked(cfg['reuse_consumer_parent']+'/config.resolved.json').parent
    rawparent=checked(cfg['reuse_raw_parent']+'/config.resolved.json').parent
    for p in [parent,rawparent]:assert json.loads(checked(p/'status.json').read_text())['status']=='PASS'
    pc=json.loads(checked(parent/'config.resolved.json').read_text())
    assert pc['source_tasks']==cfg['source_tasks']
    source_panel=[r for r in json.loads(checked(parent/'panel.json').read_text())['rows'] if r['split']=='source_selection']
    with np.load(checked(parent/'source_gradients.npz')) as a:
        h=torch.tensor(a['hidden'],device=w.device)
        grad=torch.tensor(a['gradient'],device=w.device)
    assert len(h)==len(source_panel)==192
    maps={};meta=[];ops=sorted(cfg['operations'],key=lambda op:cfg['operations'][op].index(1))
    for obj in cfg['objectives']:
        for s in cfg['seeds']:
            t=cfg['seeds'][(cfg['seeds'].index(s)+1)%len(cfg['seeds'])]
            with np.load(checked(parent/f'{obj}_s{s}_t{t}_map.npz')) as a:
                tp=torch.tensor(a['target_members'],device=w.device)
                sp=torch.tensor(a['source_members'],device=w.device)
                sg=torch.tensor(a['source_gate'],device=w.device)
                part=torch.tensor(a['partition64'],device=w.device)
                assignment=torch.tensor(a['assignment64'],device=w.device)
            with np.load(checked(rawparent/f'{obj}_s{s}_t{t}_raw_controls.npz')) as a:
                assert np.array_equal(a['target_members'],tp.cpu().numpy())
                raw=torch.tensor(a['full_ridge_0.1'],device=w.device)
            with torch.no_grad():z=saes[obj,t].encode(h)
            d=D[obj,t];energy=z.square().mean(0)*d.square().sum(1)
            credit=z*(grad@d.T)
            effect=torch.stack([credit[[i for i,r in enumerate(source_panel) if r['task']==task]].mean(0) for task in cfg['source_tasks']],1)
            role=(effect-(effect.clamp_min(0).sum(1,keepdim=True)-effect.clamp_min(0))/2).clamp_min(0)
            pullback=(raw*d[tp,None,:]).sum(-1)/d[tp].square().sum(1).clamp_min(1e-12)[:,None]
            pullback=pullback.clamp(0,1)
            rankings={
                'fcc_members':(tp,part.square()*energy[tp,None]),
                'assignment':(tp,assignment.square()*energy[tp,None]),
                'raw_pullback':(tp,pullback.square()*energy[tp,None]),
                'cached_gradient_pool':(tp,role[tp]),
                'cached_gradient_full':(torch.arange(len(d),device=w.device),role),
                'fcc_functional_priority':(tp,part*role[tp])}
            masks={};members={};details=[]
            for family,(pool,score) in rankings.items():
                for k,op in enumerate(ops):
                    eligible=torch.where(score[:,k]>0)[0]
                    order=eligible[torch.argsort(score[eligible,k],descending=True,stable=True)]
                    for cap in cfg['member_allowances']:
                        ix=pool[order[:cap]];key=(op,family,cap)
                        mask=torch.zeros(len(d),device=w.device);mask[ix]=1
                        masks[key]=mask;members[key]=ix.cpu().tolist()
                        details.append(dict(operation=op,family=family,allowance=cap,actual_members=len(ix),members=members[key]))
            payload={'source_members':sp.cpu().numpy(),'source_gate':sg.cpu().numpy(),'target_members':tp.cpu().numpy(),'source_gradient_credit':effect.cpu().numpy()}
            for (op,family,cap),ix in members.items():payload[f'{op}__{family}__{cap}']=np.asarray(ix,dtype=np.int64)
            path=w.run/f'{obj}_s{s}_t{t}_proposals.npz';np.savez_compressed(path,**payload)
            maps[obj,s]=dict(target=t,sp=sp,sg=sg,masks=masks)
            meta.append(dict(objective=obj,source=s,target=t,proposals=details,path=path.name,sha256=sha256(path)))
            log('REUSE_PROPOSALS_READY',objective=obj,source=s,target=t,proposals=len(details));budget()
    write(w.run/'PROPOSAL_FREEZE.json',dict(written_at_utc=datetime.now(timezone.utc).isoformat(),rows=meta,target_validation_outcomes_consumed=0,target_evaluation_outcomes_consumed=0,scope=cfg['scope']))
    del h,grad,z,credit
    outcomes=defaultdict(list);decisions=[]
    def utility(rows,op):
        own=cfg['source_tasks'][ops.index(op)]
        task={task:[r for r in rows if r['task']==task] for task in cfg['source_tasks']}
        rates={t:sum(r['introduced_error'] for r in rr)/len(rr) for t,rr in task.items()}
        margins={t:sum(r['decrement'] for r in rr)/len(rr) for t,rr in task.items()}
        other=[t for t in cfg['source_tasks'] if t!=own]
        return dict(selectivity=rates[own]-sum(rates[t] for t in other)/len(other),requested_error_rate=rates[own],collateral_error_rate=sum(rates[t] for t in other)/len(other),margin_selectivity=margins[own]-sum(max(margins[t],0) for t in other)/len(other),requested_margin_decrement=margins[own],task_rates=rates)
    for split in ['calibration','evaluation']:
        data=capture(split)
        for obj in cfg['objectives']:
            for s in cfg['seeds']:
                mp=maps[obj,s];t=mp['target'];z=data['codes'][obj,t];d=D[obj,t]
                for op in ops:
                    k=ops.index(op);source=(data['codes'][obj,s][:,mp['sp']]*mp['sg'][:,k])@D[obj,s][mp['sp']]
                    deltas={(family,cap):(z*mask)@d for (oo,family,cap),mask in mp['masks'].items() if oo==op}
                    deltas[('source',64)]=source
                    for off in range(0,len(data['rows']),cfg['batch_pairs']):
                        rr=data['rows'][off:off+cfg['batch_pairs']];clean,hh,_,_=forward(rr,phase='reuse_clean')
                        assert float((hh-data['hidden'][off:off+len(rr)]).abs().max())<cfg['hidden_atol']
                        for (family,cap),delta in deltas.items():
                            margin,_,_,_=forward(rr,delta[off:off+len(rr)],phase='reuse_'+family)
                            for j,r in enumerate(rr):
                                result=dict(task=r['task'],row_id=r['row_id'],introduced_error=bool(clean[j]>0 and margin[j]<=0),clean_correct=bool(clean[j]>0),edited_correct=bool(margin[j]>0),decrement=float(clean[j]-margin[j]))
                                outcomes[split,obj,s,op,family,cap].append(result)
                                w.record(kind='functional_reuse',task=r['task'],row_id=r['row_id'],component=r['task']+':'+str(r['row_id']),mode=obj+'|'+op+'|'+split+'|'+str(cap),operation=op,method=family,allowance=cap,seed=s,target_seed=t,objective=obj,split=split,margin=float(margin[j]),clean_margin=float(clean[j]),edit_norm=float(delta[off+j].norm()),introduced_error=result['introduced_error'])
                        budget()
                    log('REUSE_OPERATION_EVALUATED',split=split,objective=obj,source=s,operation=op)
        if split=='calibration':
            for obj in cfg['objectives']:
                for s in cfg['seeds']:
                    for op in ops:
                        for family in rankings:
                            for b in cfg['validation_pairs_per_task']:
                                if b==0:chosen=max(cfg['member_allowances']);scores=[]
                                else:
                                    scores=[]
                                    for cap in cfg['member_allowances']:
                                        rr=[r for r in outcomes['calibration',obj,s,op,family,cap] if r['row_id']<cfg['consumer_ranges']['calibration'][0]+b]
                                        scores.append(dict(allowance=cap,**utility(rr,op)))
                                    best=max(scores,key=lambda r:(r['selectivity'],r['margin_selectivity'],-r['allowance']));chosen=best['allowance']
                                decisions.append(dict(objective=obj,source=s,target=maps[obj,s]['target'],operation=op,family=family,validation_pairs_per_task=b,charged_candidate_pair_interventions=len(cfg['member_allowances'])*len(cfg['source_tasks'])*b,chosen_allowance=chosen,calibration_scores=scores))
            write(w.run/'SELECTION_FREEZE.json',dict(written_at_utc=datetime.now(timezone.utc).isoformat(),decisions=decisions,evaluation_outcomes_consumed=0,rule='Maximize requested clean-correct-to-wrong rate minus mean collateral rate; ties use continuous margin selectivity, then smaller member allowance. Zero validation uses64. Every family has the same four member allowances and consumed calibration pair IDs.'))
            log('ALL_REUSE_CHOICES_FROZEN',decisions=len(decisions))
        del data
    cells=[]
    for key,rr in outcomes.items():
        split,obj,s,op,family,cap=key
        cells.append(dict(split=split,objective=obj,source=s,operation=op,family=family,allowance=cap,pairs=len(rr),**utility(rr,op)))
    selected=[]
    for dec in decisions:
        rr=outcomes['evaluation',dec['objective'],dec['source'],dec['operation'],dec['family'],dec['chosen_allowance']]
        selected.append(dict(**{k:v for k,v in dec.items() if k!='calibration_scores'},**utility(rr,dec['operation'])))
    write(w.run/'consumer_results.json',dict(cells=cells,selected=selected,scope=cfg['scope'],primary='Requested introduced-error fraction minus the mean fraction for the other two functions, evaluated after member-allowance selection on separate validation. Higher is more selective. All component effects and absolute rates remain visible.'))
    w.checks['all_proposals_before_validation_and_selection_before_evaluation']=True
    w.checks['all_actual_deletions_have_binary_nonnegative_remaining_codes']=True
