"""Estimate component relations from target encoding responses to source deletion.

The reusable fits consume unlabelled natural states and frozen source identities.
An explicitly separate cached-gradient control consumes old source task gradients.
All coefficients freeze before this run's functional evaluation. Dynamic reencoding
and unrestricted raw regression diagnose representation/realization separately.
"""
import json
from datetime import datetime, timezone


def run_code_response(cfg,w,D,saes,capture,forward,checked,write,log,budget):
    import numpy as np
    import torch
    from ccad.artifacts import sha256
    from fit_component_correspondence import fit

    parent=checked(cfg['code_response_parent']+'/config.resolved.json').parent
    assert json.loads(checked(parent/'status.json').read_text())['status']=='PASS'
    pc=json.loads((parent/'config.resolved.json').read_text())
    assert pc['source_tasks']==cfg['source_tasks']
    with np.load(checked(cfg['natural_states'])) as a:
        available=np.flatnonzero(a['packed_positions']%cfg['natural_sequence_length']>=cfg['minimum_prefix'])
        ix=available[:cfg['natural_fit_rows']]
        h=torch.tensor(a['hidden'][ix],device=w.device)
        positions=a['packed_positions'][ix].copy()
    np.savez_compressed(w.run/'fit_rows.npz',row_indices=ix,packed_positions=positions)
    cap=cfg['target_members_per_component'];maps={};fitmeta=[]

    def encode(ae,states):
        return torch.cat([ae.encode(states[i:i+256]) for i in range(0,len(states),256)])

    def sparse_gate(coef,energy,binary=False):
        result=torch.zeros_like(coef)
        for k in range(coef.shape[1]):
            order=torch.argsort(coef[:,k].clamp_min(0).square()*energy,descending=True,stable=True)[:cap]
            result[order,k]=1 if binary else coef[order,k].clamp(0,1)
        return result

    # A separate information-rich baseline can read source task derivatives;
    # their values never enter the natural-data estimators below.
    panel=[r for r in json.loads(checked(parent/'panel.json').read_text())['rows'] if r['split']=='source_selection']
    with np.load(checked(parent/'source_gradients.npz')) as a:
        source_h=torch.tensor(a['hidden'],device=w.device)
        cached_grad=torch.tensor(a['gradient'],device=w.device)

    for obj in cfg['objectives']:
        for s,t in zip(cfg['seeds'],cfg['target_seeds']):
            with np.load(checked(parent/f'{obj}_s{s}_source.npz')) as a:
                sg=torch.tensor(a['gate'],device=w.device)
            ds,dt=D[obj,s],D[obj,t]
            with torch.no_grad():
                zs,zt=encode(saes[obj,s],h),encode(saes[obj,t],h)
                y=torch.stack([(zs*sg[:,k])@ds for k in range(3)],1)
                response=torch.stack([zt-encode(saes[obj,t],h-y[:,k]) for k in range(3)],2)
                z2=zt.double().square().mean(0)
                d2=dt.double().square().sum(1)
                energy=z2*d2
                signed=(zt.double()[:,:,None]*response.double()).mean(0)/z2.clamp_min(1e-12)[:,None]
                cf=signed.clamp(0,1)
                cross=torch.stack([zt.double().T@y[:,k].double()/len(h) for k in range(3)],1)
                B=(cross*dt.double()[:,None,:]).sum(-1)
                diagonal=B/energy.clamp_min(1e-12)[:,None]
                pool=torch.argsort((diagonal.clamp_min(0).square()*energy[:,None]).max(1).values,descending=True,stable=True)[:cfg['target_pool']]
                x=zt[:,pool].double();d=dt[pool].double()
                K=(x.T@x/len(h))*(d@d.T)
                coeff,info=fit(K,B[pool],steps=cfg['fit_steps'],ridge_fraction=cfg['ridge_fraction'],capacity=False)
                allowed=torch.zeros_like(coeff,dtype=torch.bool)
                for k in range(3):
                    allowed[torch.argsort(coeff[:,k].square()*energy[pool],descending=True,stable=True)[:cap],k]=True
                coeff,refit=fit(K,B[pool],steps=cfg['fit_steps'],ridge_fraction=cfg['ridge_fraction'],allowed=allowed,capacity=False)
                field=torch.zeros_like(cf);field[pool]=coeff
                A=x.T@x/len(h);A+=cfg['raw_ridge_fraction']*A.diag().mean()*torch.eye(len(pool),device=w.device,dtype=A.dtype)
                raw=torch.linalg.solve(A,x.T@y.double().reshape(len(h),-1)/len(h)).reshape(len(pool),3,-1).float()
                decoder=torch.zeros_like(cf)
                dn=dt/dt.norm(dim=1,keepdim=True).clamp_min(1e-12)
                for k in range(3):
                    sd=ds[sg[:,k]>0];score=(dn@(sd/sd.norm(dim=1,keepdim=True).clamp_min(1e-12)).T).max(1).values
                    decoder[torch.argsort(score,descending=True,stable=True)[:cap],k]=1
                cz=encode(saes[obj,t],source_h)
                credit=cz*(cached_grad@dt.T)
                effect=torch.stack([credit[[i for i,r in enumerate(panel) if r['task']==task]].mean(0) for task in cfg['source_tasks']],1)
                role=effect-(effect.clamp_min(0).sum(1,keepdim=True)-effect.clamp_min(0))/2
                cached=torch.zeros_like(cf)
                for k in range(3):cached[torch.argsort(role[:,k],descending=True,stable=True)[:cap],k]=1
                gates={'code_soft':sparse_gate(cf,energy),'code_binary':sparse_gate(cf,energy,True),
                       'field_soft':field,'field_binary':(field>0).double(),'decoder_binary':decoder,'cached_binary':cached}
                payload={name:g.float().cpu().numpy() for name,g in gates.items()}
                payload.update(signed_response_coefficient=signed.float().cpu().numpy(),raw=raw.cpu().numpy(),raw_pool=pool.cpu().numpy(),source_gate=sg.cpu().numpy())
                fp=w.run/f'{obj}_s{s}_t{t}_relation.npz';np.savez_compressed(fp,**payload)
                maps[obj,s]=dict(target=t,source_gate=sg,gates={name:g.float() for name,g in gates.items()},pool=pool,raw=raw)
                total=response.double().square().sum().clamp_min(1e-12)
                meta=dict(objective=obj,source=s,target=t,fit_rows=len(h),path=fp.name,sha256=sha256(fp),
                          increased_code_energy_fraction=float(response.double().clamp_max(0).square().sum()/total),
                          coefficient_negative=int((signed<0).sum()),coefficient_above_one=int((signed>1).sum()),
                          source_energy=y.double().square().sum(2).mean(0).cpu().tolist(),fit=info,refit=refit)
                fitmeta.append(meta);log('NATURAL_RELATION_FITTED',**meta);budget()
                del response,cross,zs,zt,x,y
    write(w.run/'RELATION_FREEZE.json',dict(written_at_utc=datetime.now(timezone.utc).isoformat(),fits=fitmeta,
        estimators_fit_on='Unlabelled natural hidden states and frozen source SAE feature identities',
        cached_baseline_information='192 previously labelled source task states and their cached derivatives, read separately',
        new_target_outputs_consumed=0,new_gradients=0,evaluation_is_development=True))
    del h,source_h,cached_grad
    data=capture('evaluation');cells=[]
    for obj in cfg['objectives']:
        for s in cfg['seeds']:
            mp=maps[obj,s];t=mp['target'];zs=data['codes'][obj,s];zt=data['codes'][obj,t];ds,dt=D[obj,s],D[obj,t]
            for k,op in enumerate(cfg['operations']):
                source=(zs*mp['source_gate'][:,k])@ds
                deltas={'source':source,**{name:(zt*g[:,k])@dt for name,g in mp['gates'].items()},
                        'raw_field':zt[:,mp['pool']]@mp['raw'][:,k],
                        'dynamic_reencoding':(zt-encode(saes[obj,t],data['hidden']-source))@dt}
                margins={name:[] for name in deltas};vector={}
                for name,delta in deltas.items():
                    vector[name]=dict(relative_field_mse=float((delta-source).double().square().sum()/source.double().square().sum().clamp_min(1e-12)),
                                      mean_edit_norm=float(delta.norm(dim=1).mean()))
                for off in range(0,len(data['rows']),cfg['batch_pairs']):
                    rr=data['rows'][off:off+cfg['batch_pairs']]
                    for name,delta in deltas.items():
                        margin,hh,_,_=forward(rr,delta[off:off+len(rr)],phase='code_response_'+name)
                        assert float((hh-data['hidden'][off:off+len(rr)]).abs().max())<cfg['hidden_atol']
                        margins[name].extend(margin.cpu().tolist())
                        for j,r in enumerate(rr):
                            w.record(kind='code_response_reuse',objective=obj,seed=s,target_seed=t,operation=op,
                                     method=name,task=r['task'],row_id=r['row_id'],component=r['task']+':'+str(r['row_id']),
                                     mode=obj+'|'+op,margin=float(margin[j]),clean_margin=float(data['clean'][off+j]),edit_norm=float(delta[off+j].norm()))
                    budget()
                src=np.asarray(margins['source']);clean=data['clean'].cpu().numpy()
                for name,values in margins.items():
                    values=np.asarray(values);effects=[]
                    for task in cfg['source_tasks']:
                        ix=np.asarray([i for i,r in enumerate(data['rows']) if r['task']==task])
                        effects.append(dict(task=task,source_decrement=float(np.mean(clean[ix]-src[ix])),decrement=float(np.mean(clean[ix]-values[ix])),
                                            source_margin_mae=float(np.mean(abs(values[ix]-src[ix]))),
                                            introduced_error=float(np.mean((clean[ix]>0)&(values[ix]<=0)))))
                    cells.append(dict(objective=obj,source=s,target=t,operation=op,method=name,
                        source_margin_mae=float(np.mean(abs(values-src))),effect_profile=effects,**vector[name]))
                log('SOURCE_PROFILE_EVALUATED',objective=obj,source=s,target=t,operation=op,methods=len(deltas))
    write(w.run/'code_response_results.json',dict(cells=cells,fitmeta=fitmeta,
        primary='Source intervention margin MAE across all three tasks, per source component and independent target pair',
        scope=cfg['scope'],raw_control='Unrestricted regression from the same natural target codes to source contribution field',
        dynamic_control='Recomputes source SAE and target encoder per input; a realization diagnostic, not a reusable target-only relation'))
    w.checks['all_relations_frozen_before_current_functional_evaluation']=True
    w.checks['native_gates_in_zero_one']=all(bool(((g>=0)&(g<=1)).all()) for mp in maps.values() for g in mp['gates'].values())
