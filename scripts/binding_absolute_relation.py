"""Fit the same fuzzy membership relation to states, differences, or both."""
import json
import time


def build_anchored_transfers(w,cfg,rows,hidden,zs,sae,conditional,target,zt,seed,target_seed):
    import numpy as np
    import torch
    from run_causalgym_multisite import ROOT,write
    from binding_correspondence import fit_union_relation
    spec=cfg['absolute_relation'];parent=ROOT/spec['parent'];started=time.perf_counter()
    assert json.loads(w.checked(parent/'status.json').read_text())['status']=='PASS'
    previous=json.loads(w.checked(parent/'config.resolved.json').read_text())
    for key in ['model_revision','training_runs','checkpoint_step','layers','semantic_queries']:
        assert cfg[key]==previous[key],key
    oldrows=json.loads(w.checked(parent/'panel.json').read_text())['rows']
    fit=lambda rs:[r for r in rs if r['split']=='fit']
    assert fit(rows)==fit(oldrows)
    with np.load(w.checked(parent/f'binding_relation_s{seed}_t{target_seed}.npz')) as z:
        si=torch.tensor(z['source_indices'],device=w.device);ti=torch.tensor(z['target_indices'],device=w.device)
        initial=torch.tensor(z['weights'][0],device=w.device)
        assignment=torch.tensor(z['assignment'],device=w.device)
        reader=torch.tensor(z['code_readout'],device=w.device)
        oldquery=torch.tensor(z['source_query'],device=w.device)
    ds=sae.decoder.weight.T[si];dt=target.decoder.weight.T[ti]
    q=conditional[...,si];paired=[r['paired_row'] for r in rows]
    fits=[r['row_id'] for r in rows if r['split']=='fit' and not r['donor']]
    assert torch.equal(q[fits],oldquery[fits])
    src={'replace':zs[paired][...,si]-zs[...,si],'delete':-zs[...,si]}
    tgt={'replace':zt[paired][...,ti]-zt[...,ti],'delete':-zt[...,ti]}
    outputs={f'source_{op}':(value*q)@ds for op,value in src.items()}
    fitq=q[fits].flatten(0,1)
    components={}
    for name,op in [('difference','replace'),('absolute','delete')]:
        x=tgt[op][fits].flatten(0,1);y=src[op][fits].flatten(0,1)
        norm=((y*fitq)@ds).square().sum().sqrt().clamp_min(1e-12)
        components[name]=(x,y,norm)
    relations={'initial':initial};info={};payload=dict(source_indices=si.cpu().numpy(),target_indices=ti.cpu().numpy())
    old_precision=torch.get_float32_matmul_precision();torch.set_float32_matmul_precision('highest')
    for family in ['difference','absolute','joint']:
        terms=['difference','absolute'] if family=='joint' else [family]
        # Equal normalized residual energies; fitting states and differences
        # consumes the same already-observed paired source-fit activations.
        x=torch.cat([components[k][0]/components[k][2] for k in terms])
        y=torch.cat([components[k][1]/components[k][2] for k in terms])
        qq=fitq.repeat(len(terms),1)
        rel,details=fit_union_relation(x,y,qq,ds,dt,initial,steps=spec['steps'],lr=spec['lr'],ridge_fraction=spec['ridge_fraction'])
        relations[family]=rel;info[family]=details
        # Check the combined loss against independently formed per-operation
        # normalized residuals, so stacking cannot silently change weighting.
        weights=(fitq@rel.T).clamp(0,1)
        direct=[]
        for term in terms:
            xx,yy,normalizer=components[term]
            direct.append((((xx*weights)@dt-(yy*fitq)@ds).square().sum()/normalizer.square()).item())
        assert abs(details['relative_query_field_mse']-sum(direct)/len(direct))<1e-5
        details['per_objective_relative_mse']=dict(zip(terms,direct))
        w.progress('ABSOLUTE_RELATION_FIT',source_seed=seed,family=family,**details)
    for family,rel in relations.items():
        gate=(q@rel.T).clamp(0,1);payload[family]=rel.cpu().numpy()
        for op,change in tgt.items():
            coeff=change*gate
            keep=torch.argsort(coeff.abs()*dt.norm(dim=1),dim=-1,descending=True,stable=True)[...,:spec['members']]
            coeff=torch.zeros_like(coeff).scatter(-1,keep,coeff.gather(-1,keep))
            assert int((coeff!=0).sum(-1).max())<=spec['members']
            assert float((zt[...,ti]+coeff).min())>=-1e-5
            if op=='delete':assert float(coeff.max())<=0
            outputs[f'{op}_{family}']=(coeff@dt).detach()
        payload[f'{family}_field_mse']=np.array([float((outputs[f'{op}_{family}'][fits]-outputs[f'source_{op}'][fits]).square().sum()/outputs[f'source_{op}'][fits].square().sum()) for op in ['replace','delete']])
    # Full-dictionary assignment and target-only re-identification receive the
    # same country metadata; neither sees target output labels.
    countries=sorted({c for r in rows for c in r['countries']});means=[]
    for country in countries:
        sites=[(r['row_id'],site) for r in rows if r['split']=='fit' for site in range(2) if r['countries'][site]==country]
        means.append(torch.stack([zt[i,site] for i,site in sites]).mean(0))
    means=torch.stack(means);direct=torch.zeros_like(zt);full_dt=target.decoder.weight.T
    for r in rows:
        other=rows[r['paired_row']]
        for site in range(2):
            u,v=[countries.index(rr['countries'][site]) for rr in [r,other]]
            score=(means[v]-means[u]).abs()*full_dt.norm(dim=1)
            selected=torch.argsort(score,descending=True,stable=True)[:spec['members']]
            direct[r['row_id'],site,selected]=1
    for op,change in [('replace',zt[paired]-zt),('delete',-zt)]:
        outputs[f'{op}_direct']=(change*direct)@full_dt
        outputs[f'{op}_assignment']=(change[...,assignment]*q)@full_dt[assignment]
    with np.load(w.checked(parent/f'selected_writer_s{seed}_t{target_seed}.npz')) as z:
        gain=torch.tensor(z['native_gain'],device=w.device)
    for op in ['replace','delete']:
        amplitude=(tgt['replace']@reader)*q if op=='replace' else src['delete']*q
        coeff=(amplitude@gain).maximum(-zt[...,ti])
        if op=='delete':coeff=coeff.clamp_max(0)
        keep=torch.argsort(coeff.abs()*dt.norm(dim=1),dim=-1,descending=True,stable=True)[...,:spec['members']]
        coeff=torch.zeros_like(coeff).scatter(-1,keep,coeff.gather(-1,keep))
        outputs[f'{op}_gain']=(coeff@dt).detach()
    if spec.get('state_components'):
        from binding_state_components import state_components
        state_components(w,cfg,rows,hidden,zs,zt,q,si,ti,ds,dt,outputs,seed,target_seed)
    torch.set_float32_matmul_precision(old_precision)
    np.savez_compressed(w.run/f'absolute_relation_s{seed}_t{target_seed}.npz',**payload)
    write(w.run/f'ABSOLUTE_RELATION_s{seed}.json',dict(fits=info,seconds=time.perf_counter()-started,
          fit_contexts=len({rows[i]['component'] for i in fits}),source_query='Frozen recipient-donor country contrast mask',
          source_amplitudes_at_execution='Native membership methods use only current target amplitudes and source query; gain-deletion comparator gets exact source amplitudes.',
          target_output_labels=0,target_output_gradients=0,initialization='Same existing member relation',
          operation_order=['replace','delete']))
    return {name:field.detach() for name,field in outputs.items()}
