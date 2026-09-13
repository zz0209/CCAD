"""Transfer source functional responses to an independent target SAE cohort.

Every family ranks the full target dictionary. Singleton member choices are
frozen using calibration, then reused for exact membership regions and unions.
"""
import itertools,json
from collections import defaultdict
from datetime import datetime,timezone


def run_consumer(cfg,w,D,saes,capture,forward,checked,write,log,budget):
    import numpy as np,torch
    from ccad.artifacts import sha256
    parent=checked(cfg['reuse_consumer_parent']+'/status.json').parent
    paths=checked(cfg['path_ensemble_parent']+'/status.json').parent
    for p in [parent,paths]:assert json.loads((p/'status.json').read_text())['status']=='PASS'
    source_panel=[r for r in json.loads(checked(parent/'panel.json').read_text())['rows'] if r['split']=='source_selection']
    with np.load(checked(parent/'source_gradients.npz')) as a:
        h=torch.tensor(a['hidden'],device=w.device);g0=torch.tensor(a['gradient'],device=w.device)
    ix=[i for i,r in enumerate(source_panel) if cfg['path_source_range'][0]<=r['row_id']<cfg['path_source_range'][1]]
    rows=[source_panel[i] for i in ix];hp=h[ix];gp=g0[ix]
    groups=[[i for i,r in enumerate(rows) if r['task']==task] for task in cfg['source_tasks']]
    groups64=[[i for i,r in enumerate(source_panel) if r['task']==task] for task in cfg['source_tasks']]
    ops=sorted(cfg['operations'],key=lambda op:cfg['operations'][op].index(1));maps={};meta=[]
    def role(C,groups=groups):
        means=torch.stack([C[:,ids,:].mean(1) for ids in groups],1)
        return torch.stack([(means[k,k]-means[k,[j for j in range(3) if j!=k]].clamp_min(0).mean(0)).clamp_min(0) for k in range(3)],1)
    for obj in cfg['objectives']:
        bank=[];sources={}
        for s in cfg['seeds']:
            oldt=cfg['seeds'][(cfg['seeds'].index(s)+1)%len(cfg['seeds'])]
            with np.load(checked(paths/f'{obj}_s{s}_t{oldt}_path_relation.npz')) as a:
                assert np.array_equal(a['row_ids'],[r['row_id'] for r in rows])
                assert np.array_equal(a['task_ids'],[cfg['source_tasks'].index(r['task']) for r in rows])
                bank.append(torch.tensor(a['source_path_gradient'],device=w.device))
                sources[s]=(torch.tensor(a['source_members'],device=w.device),torch.tensor(a['source_gate'],device=w.device))
        G=torch.stack(bank);factors=(G*gp[None,None]).sum(-1)/gp.square().sum(1).clamp_min(1e-16)[None,None]
        for s,t in zip(cfg['seeds'],cfg['target_seeds']):
            d=D[obj,t]
            with torch.no_grad():z64=saes[obj,t].encode(h)
            z=z64[ix];clean=z*(gp@d.T);credit64=z64*(g0@d.T)
            roles=[];scalars=[];wrongs=[];meanC=None
            for i,source in enumerate(cfg['seeds']):
                C=z[None]*torch.einsum('knd,pd->knp',G[i],d)
                meanC=C/len(cfg['seeds']) if meanC is None else meanC+C/len(cfg['seeds'])
                roles.append(role(C));scalars.append(role(factors[i,:,:,None]*clean[None]));wrongs.append(role(C[[1,2,0]]))
            roles=torch.stack(roles);scalars=torch.stack(scalars)
            rankings=dict(shared_path_full=roles.amin(0),mean_path_full=role(meanC),
                shared_scalar_full=scalars.amin(0),wrong_shared_path_full=torch.stack(wrongs).amin(0),
                single_path_full=roles[cfg['seeds'].index(s)],clean32_full=role(clean[None].expand(3,-1,-1)),
                cached64_full=role(credit64[None].expand(3,-1,-1),groups64))
            assert set(rankings)==set(cfg['methods'])
            masks={};details=[];payload={'source_seeds':np.asarray(cfg['seeds']),
                'source_role_scores':roles.cpu().numpy(),'source_scalar_role_scores':scalars.cpu().numpy()}
            for family,score in rankings.items():
                payload[family+'_score']=score.cpu().numpy()
                for k,op in enumerate(ops):
                    eligible=torch.where(score[:,k]>0)[0]
                    order=eligible[torch.argsort(score[eligible,k],descending=True,stable=True)]
                    for cap in cfg['member_allowances']:
                        ids=order[:cap];mask=torch.zeros(len(d),device=w.device);mask[ids]=1
                        masks[op,family,cap]=mask;payload[f'{op}__{family}__{cap}']=ids.cpu().numpy()
                        details.append(dict(operation=op,family=family,allowance=cap,actual_members=len(ids),members=ids.cpu().tolist()))
            path=w.run/f'{obj}_s{s}_t{t}_proposals.npz';np.savez_compressed(path,**payload)
            maps[obj,s]=dict(target=t,masks=masks,source=sources[s])
            meta.append(dict(objective=obj,source=s,target=t,source_bank=cfg['seeds'],proposals=details,path=path.name,sha256=sha256(path)))
            log('INDEPENDENT_PROPOSALS_READY',objective=obj,source_bank=cfg['seeds'],target=t,proposals=len(details));budget()
    write(w.run/'PROPOSAL_FREEZE.json',dict(written_at_utc=datetime.now(timezone.utc).isoformat(),rows=meta,target_validation_outcomes_consumed=0,target_evaluation_outcomes_consumed=0,scope=cfg['scope']))
    del h,g0,hp,gp,z,z64,clean,credit64,C,G,roles,scalars
    outcomes=defaultdict(list);decisions=[];structural=[]
    def utility(rr,requested):
        tasks={task:[r for r in rr if r['task']==task] for task in cfg['source_tasks']}
        rates={t:sum(r['introduced_error'] for r in rrs)/len(rrs) for t,rrs in tasks.items()}
        margins={t:sum(r['decrement'] for r in rrs)/len(rrs) for t,rrs in tasks.items()}
        own=[cfg['source_tasks'][k] for k in requested];other=[t for t in tasks if t not in own]
        req=sum(rates[t] for t in own)/len(own);col=sum(rates[t] for t in other)/len(other) if other else 0.
        rm=sum(margins[t] for t in own)/len(own);cm=sum(max(margins[t],0) for t in other)/len(other) if other else 0.
        return dict(selectivity=req-col,requested_error_rate=req,collateral_error_rate=col,margin_selectivity=rm-cm,requested_margin_decrement=rm,task_rates=rates,task_margin_decrements=margins)
    def evaluate(data,obj,s,op,family,cap,delta,kind='functional_reuse',region=None):
        t=maps[obj,s]['target'];results=[]
        for off in range(0,len(data['rows']),cfg['batch_pairs']):
            rr=data['rows'][off:off+cfg['batch_pairs']];clean=data['clean'][off:off+len(rr)]
            margin,hh,_,_=forward(rr,delta[off:off+len(rr)],phase=kind+'_'+family)
            assert float((hh-data['hidden'][off:off+len(rr)]).abs().max())<cfg['hidden_atol']
            for j,r in enumerate(rr):
                result=dict(task=r['task'],row_id=r['row_id'],introduced_error=bool(clean[j]>0 and margin[j]<=0),clean_correct=bool(clean[j]>0),edited_correct=bool(margin[j]>0),decrement=float(clean[j]-margin[j]))
                results.append(result)
                w.record(kind=kind,task=r['task'],row_id=r['row_id'],component=r['task']+':'+str(r['row_id']),mode=kind+'|'+obj+'|'+op+'|'+r['split']+'|'+str(cap),operation=op,method=family,allowance=cap,seed=s,target_seed=t,objective=obj,split=r['split'],margin=float(margin[j]),clean_margin=float(clean[j]),edit_norm=float(delta[off+j].norm()),introduced_error=result['introduced_error'],region=region)
            budget()
        return results
    for split in ['calibration','evaluation']:
        data=capture(split)
        for obj in cfg['objectives']:
            for s in cfg['seeds']:
                mp=maps[obj,s];t=mp['target'];z=data['codes'][obj,t];d=D[obj,t]
                for op in ops:
                    for family in cfg['methods']:
                        caps=cfg['member_allowances'] if split=='calibration' else sorted({x['chosen_allowance'] for x in decisions if x['objective']==obj and x['source']==s and x['operation']==op and x['family']==family})
                        for cap in caps:
                            delta=(z*mp['masks'][op,family,cap])@d
                            outcomes[split,obj,s,op,family,cap]=evaluate(data,obj,s,op,family,cap,delta)
                    if split=='evaluation':
                        sp,sg=mp['source'];delta=(data['codes'][obj,s][:,sp]*sg[:,ops.index(op)])@D[obj,s][sp]
                        outcomes[split,obj,s,op,'source',64]=evaluate(data,obj,s,op,'source',64,delta)
                    log('INDEPENDENT_OPERATION_EVALUATED',split=split,objective=obj,target=t,operation=op)
        if split=='calibration':
            for obj in cfg['objectives']:
                for s in cfg['seeds']:
                    for op in ops:
                        for family in cfg['methods']:
                            for b in cfg['validation_pairs_per_task']:
                                scores=[];chosen=max(cfg['member_allowances'])
                                if b:
                                    for cap in cfg['member_allowances']:
                                        rr=[r for r in outcomes['calibration',obj,s,op,family,cap] if r['row_id']<cfg['consumer_ranges']['calibration'][0]+b]
                                        scores.append(dict(allowance=cap,**utility(rr,[ops.index(op)])))
                                    chosen=max(scores,key=lambda r:(r['selectivity'],r['margin_selectivity'],-r['allowance']))['allowance']
                                decisions.append(dict(objective=obj,source=s,target=maps[obj,s]['target'],operation=op,family=family,validation_pairs_per_task=b,charged_candidate_pair_interventions=len(cfg['member_allowances'])*len(cfg['source_tasks'])*b,chosen_allowance=chosen,calibration_scores=scores))
            write(w.run/'SELECTION_FREEZE.json',dict(written_at_utc=datetime.now(timezone.utc).isoformat(),decisions=decisions,evaluation_outcomes_consumed=0,rule='Maximize requested introduced errors minus mean collateral; break ties by margin selectivity then smaller allowance. Zero validation uses64.'))
            for obj in cfg['objectives']:
                for s in cfg['seeds']:
                    mp=maps[obj,s];mp['structural_masks']={}
                    for family in cfg['structure_families']:
                        chosen=[]
                        for op in ops:
                            dec=next(x for x in decisions if x['objective']==obj and x['source']==s and x['operation']==op and x['family']==family and x['validation_pairs_per_task']==cfg['primary_validation_budget'])
                            chosen.append(mp['masks'][op,family,dec['chosen_allowance']].bool())
                        labels=sum((2**k)*m.long() for k,m in enumerate(chosen))
                        for label in range(1,8):
                            requested=[k for k in range(3) if label&(1<<k)];mask=labels==label
                            key=('region',family,str(label));mp['structural_masks'][key]=mask.float()
                            structural.append(dict(objective=obj,source=s,target=mp['target'],kind='region',family=family,operation=str(label),requested=requested,members=torch.where(mask)[0].cpu().tolist(),actual_members=int(mask.sum())))
                        for n in [2,3]:
                            for requested in itertools.combinations(range(3),n):
                                label=sum(1<<k for k in requested);mask=torch.stack([chosen[k] for k in requested]).any(0)
                                key=('union',family,str(label));mp['structural_masks'][key]=mask.float()
                                structural.append(dict(objective=obj,source=s,target=mp['target'],kind='union',family=family,operation=str(label),requested=list(requested),members=torch.where(mask)[0].cpu().tolist(),actual_members=int(mask.sum())))
            write(w.run/'STRUCTURE_FREEZE.json',dict(written_at_utc=datetime.now(timezone.utc).isoformat(),rows=structural,evaluation_outcomes_consumed=0,definition='Exact nonempty membership labels partition the union of three selected singleton sets. Every label including empty regions is retained. Union requests remove each member once. No joint or region outcome selects these sets.'))
            log('INDEPENDENT_SELECTION_AND_STRUCTURE_FROZEN',decisions=len(decisions),structural_requests=len(structural))
        else:
            structure_results=[]
            for obj in cfg['objectives']:
                for s in cfg['seeds']:
                    mp=maps[obj,s];z=data['codes'][obj,mp['target']];d=D[obj,mp['target']]
                    for (kind,family,label),mask in mp['structural_masks'].items():
                        detail=next(x for x in structural if x['objective']==obj and x['source']==s and x['kind']==kind and x['family']==family and x['operation']==label)
                        delta=(z*mask)@d
                        rr=evaluate(data,obj,s,label,family,detail['actual_members'],delta,kind='functional_'+kind,region=label)
                        structure_results.append(dict(**detail,**utility(rr,detail['requested'])))
                    log('INDEPENDENT_STRUCTURE_EVALUATED',objective=obj,target=mp['target'])
            write(w.run/'structure_results.json',dict(rows=structure_results,primary_validation_budget=cfg['primary_validation_budget'],scope='Member-role partitions and unions fixed before evaluation; all functions, all seeds, empty regions and collateral effects retained.'))
        del data
    cells=[]
    for (split,obj,s,op,family,cap),rr in outcomes.items():
        cells.append(dict(split=split,objective=obj,source=s,operation=op,family=family,allowance=cap,pairs=len(rr),**utility(rr,[ops.index(op)])))
    selected=[]
    for dec in decisions:
        rr=outcomes['evaluation',dec['objective'],dec['source'],dec['operation'],dec['family'],dec['chosen_allowance']]
        selected.append(dict(**{k:v for k,v in dec.items() if k!='calibration_scores'},**utility(rr,[ops.index(dec['operation'])])))
    write(w.run/'consumer_results.json',dict(cells=cells,selected=selected,scope=cfg['scope']))
    w.checks.update(all_proposals_before_validation_and_selection_before_evaluation=True,
        all_actual_deletions_have_binary_nonnegative_remaining_codes=True,
        disjoint_target_initializations=True,all_structural_requests_before_evaluation=True)
