"""Retain the task response tensor and condition each member query on its consumer."""
import json,itertools


def run_response_queries(cfg,w,D,saes,capture,forward,checked,write,log,budget):
    import numpy as np,torch
    from relational_exclusion_queries import evaluate_queries
    parent=checked(cfg['response_query_parent']+'/status.json').parent;pc=json.loads(checked(parent/'config.resolved.json').read_text())
    assert json.loads((parent/'status.json').read_text())['status']=='PASS'
    source=checked(pc['reuse_consumer_parent']+'/status.json').parent;paths=checked(pc['path_ensemble_parent']+'/status.json').parent
    rows=[x for x in json.loads(checked(source/'panel.json').read_text())['rows'] if x['split']=='source_selection']
    with np.load(checked(source/'source_gradients.npz')) as a:h=torch.tensor(a['hidden'],device=w.device);g=torch.tensor(a['gradient'],device=w.device)
    ix=[i for i,x in enumerate(rows) if pc['path_source_range'][0]<=x['row_id']<pc['path_source_range'][1]];gp=g[ix]
    groups=[[i for i,j in enumerate(ix) if rows[j]['task']==task] for task in cfg['source_tasks']];groups64=[[i for i,x in enumerate(rows) if x['task']==task] for task in cfg['source_tasks']]
    choices=json.loads(checked(parent/'SELECTION_FREEZE.json').read_text())['decisions'];ops=sorted(pc['operations'],key=lambda x:pc['operations'][x].index(1));proposals=json.loads(checked(parent/'PROPOSAL_FREEZE.json').read_text())['rows']
    queryrows=[];masks={}
    for obj in cfg['objectives']:
        bank=[]
        for i,s in enumerate(cfg['seeds']):
            oldt=cfg['seeds'][(i+1)%len(cfg['seeds'])]
            with np.load(checked(paths/f'{obj}_s{s}_t{oldt}_path_relation.npz')) as a:
                assert np.array_equal(a['row_ids'],[rows[j]['row_id'] for j in ix]);assert np.array_equal(a['task_ids'],[cfg['source_tasks'].index(rows[j]['task']) for j in ix]);bank.append(torch.tensor(a['source_path_gradient'],device=w.device))
        for s,t in zip(cfg['seeds'],cfg['target_seeds']):
            d=D[obj,t]
            with torch.no_grad():z64=saes[obj,t].encode(h)
            z=z64[ix];clean64=z64*(g@d.T);clean32=z*(gp@d.T);cleanmeans=torch.stack([clean64[j].mean(0) for j in groups64]);profiles=[];scalarprofiles=[];sourceids=[]
            for i,ss in enumerate(cfg['seeds']):
                if cfg.get('leave_target_out') and ss==t:continue
                G=bank[i];C=z[None]*torch.einsum('knd,pd->knp',G,d);profiles.append(torch.stack([C[:,j].mean(1) for j in groups],1))
                beta=(G*gp[None]).sum(-1)/gp.square().sum(-1).clamp_min(1e-16)[None];Cs=beta[:,:,None]*clean32[None];scalarprofiles.append(torch.stack([Cs[:,j].mean(1) for j in groups],1));sourceids.append(ss)
            P=torch.stack(profiles);S=torch.stack(scalarprofiles)
            item=next(x for x in proposals if x['objective']==obj and x['source']==s)
            with np.load(checked(parent/item['path'])) as a:own=torch.tensor(a['shared_path_full_score'],device=w.device)
            payload=dict(source_ids=np.asarray(sourceids),task_response_profiles=P.cpu().numpy(),scalar_response_profiles=S.cpu().numpy(),clean64_profiles=cleanmeans.cpu().numpy(),unconditional_scores=own.cpu().numpy())
            for k,l in itertools.permutations(range(3),2):
                choice=next(x for x in choices if x['objective']==obj and x['source']==s and x['operation']==ops[k] and x['family']=='shared_path_full' and x['validation_pairs_per_task']==16)
                with np.load(checked(parent/item['path'])) as a:count=len(a[f'{ops[k]}__shared_path_full__{choice["chosen_allowance"]}'])
                scores=dict(unconditional=own[:,k],conditional_path=(P[:,k,k]-P[:,k,l].clamp_min(0)).amin(0),conditional_scalar=(S[:,k,k]-S[:,k,l].clamp_min(0)).amin(0),conditional_clean=cleanmeans[k]-cleanmeans[l].clamp_min(0))
                for strategy,score in scores.items():
                    ids=torch.argsort(score,descending=True,stable=True)[:count];members=ids.cpu().tolist();operation=f'{k}>{l}';payload[f'{operation}__{strategy}__scores']=score.cpu().numpy()
                    detail=dict(objective=obj,source=s,target=t,family='response_query',strategy=strategy,operation=operation,delete_task=k,preserve_task=l,members=members,actual_members=count,full_members=count,shared_members=0,nonpositive_selected=int((score[ids]<=0).sum()),source_bank=sourceids)
                    queryrows.append(detail);mask=torch.zeros(len(d),device=w.device);mask[ids]=1;masks[obj,s,'response_query',operation,strategy]=mask
            np.savez_compressed(w.run/f'{obj}_s{s}_t{t}_response_tensor.npz',**payload)
            log('RESPONSE_QUERY_PROPOSALS_READY',objective=obj,target=t,source_bank=sourceids);budget()
    evaluate_queries(cfg,w,D,capture,forward,write,log,budget,queryrows,masks)
