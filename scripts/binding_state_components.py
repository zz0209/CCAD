"""Recover a source part as a bounded contribution of the current target code."""
import time


def state_components(w,cfg,rows,hidden,zs,zt,q,si,ti,ds,dt,forward_fields,seed,target_seed):
    import numpy as np
    import torch
    from run_causalgym_multisite import write
    from adaptive_native_execution import realize
    spec=cfg['absolute_relation']['state_components'];started=time.perf_counter()
    fitrows=[r['row_id'] for r in rows if r['split']=='fit']
    contexts=sorted({rows[i]['component'] for i in fitrows});cutoff=contexts[int(.75*len(contexts))]
    train=[i for i in fitrows if rows[i]['component']<cutoff]
    validation=[i for i in fitrows if rows[i]['component']>=cutoff]
    paired=[r['paired_row'] for r in rows];assert torch.equal(q,q[paired])
    y=zs[...,si];norm=ds.norm(dim=1);predictions={};readers={};payload={}
    def solve(x,y,fraction):
        x,y=x.double(),y.double();mx=x.mean(0);my=y.mean(0)
        xc=x-mx;scale=xc.square().mean(0).sqrt().clamp_min(1e-6);xx=xc/scale
        gram=xx@xx.T;ridge=fraction*gram.diag().mean().clamp_min(1e-8)
        dual=torch.linalg.solve(gram+ridge*torch.eye(len(x),device=x.device,dtype=x.dtype),y-my)
        b=xx.T@dual/scale[:,None];return b.float(),(my-mx@b).float()
    for name,x in [('code',zt[...,ti]),('raw',hidden)]:
        scores=[]
        for fraction in spec['ridge']:
            b,offset=solve(x[train].flatten(0,1),y[train].flatten(0,1),fraction)
            score=float((((x[validation]@b+offset).clamp_min(0)-y[validation])*norm).square().mean())
            scores.append((score,fraction))
        _,selected=min(scores)
        b,offset=solve(x[fitrows].flatten(0,1),y[fitrows].flatten(0,1),selected)
        pred=(x@b+offset).clamp_min(0);predictions[name]=pred
        readers[name]=dict(chosen_ridge=selected,validation=scores,
                           fit_rows=len(fitrows),training_contexts=len(contexts)*3//4,validation_contexts=len(contexts)//4)
        payload[name+'_reader']=b.cpu().numpy();payload[name+'_offset']=offset.cpu().numpy()
        part=(pred*q)@ds
        forward_fields['delete_readout_'+name]=-part
        forward_fields['replace_readout_'+name]=part[paired]-part
    diagnostics={}
    for name,pred in [('exact',y)]+list(predictions.items()):
        desired=(pred*q)@ds
        field,coeff,indices,info=realize(desired,zt[...,ti],dt,members=spec['members'],**spec['projection'])
        full=torch.zeros_like(zt[...,ti]).scatter(-1,indices,coeff)
        assert float(full.min())>=0 and bool((full<=zt[...,ti]+1e-6).all())
        info=dict(info,constraints='component coefficients between zero and current target code')
        diagnostics[name]=info
        payload[name+'_contribution_coefficients']=full.cpu().numpy()
        for op,c in [('delete',-full),('replace',full[paired]-full)]:
            keep=torch.argsort(c.abs()*dt.norm(dim=1),dim=-1,descending=True,stable=True)[...,:spec['members']]
            c=torch.zeros_like(c).scatter(-1,keep,c.gather(-1,keep))
            assert float((zt[...,ti]+c).min())>=-1e-6
            assert int((c!=0).sum(-1).max())<=spec['members']
            forward_fields[op+'_anchor_'+name]=(c@dt).detach()
        w.progress('STATE_COMPONENT_READY',origin=name,**info)
    np.savez_compressed(w.run/f'state_component_relations_s{seed}_t{target_seed}.npz',source_indices=si.cpu().numpy(),target_indices=ti.cpu().numpy(),**payload)
    write(w.run/f'STATE_COMPONENTS_s{seed}.json',dict(seconds=time.perf_counter()-started,readers=readers,projection=diagnostics,
          representation='0 <= target component coefficients <= current target code. Delete subtracts current component; replacement subtracts current and adds donor component, then trims64changes.',
          source_query='Same symmetric country-contrast query at recipient and donor',
          new_target_outputs_used=0,training_information='Original source-fit absolute code states; prediction fits use held-out source-fit contexts for ridge selection',
          exact_variant='Execution reference supplied with actual source codes; not a performance bound',
          raw_readout='Unrestricted source-basis contribution predicted from full hidden state'))
