"""Fit position-dependent weights on fixed cross-seed member candidates."""
import json
from datetime import datetime, timezone
from fit_component_correspondence import fit


def fit_positions(w,cfg,saes,codes,rows,budget):
    import numpy as np
    import torch
    previous_precision=torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    root=w.run.parent.parent;pc=cfg['position_relation'];prior=root/pc['response_run']
    assert json.loads(w.checked(prior/'status.json').read_text())['status']=='PASS'
    old=json.loads(w.checked(prior/'config.resolved.json').read_text())
    assert all(old[k]==cfg[k] for k in ['training_run','checkpoint_step','model_revision','source_cache_run'])
    assert json.loads(w.checked(prior/'panel.json').read_text())['rows']==rows
    source_run=root/old['response_relation']['relation_run']
    scalar_run=root/pc['scalar_run']
    assert json.loads(w.checked(scalar_run/'status.json').read_text())['status']=='PASS'
    role_schema=pc.get('role_schema',False)
    cap=cfg['members'][0];nsteps=cfg['max_new_tokens']+int(role_schema);width=saes[cfg['seeds'][0]].decoder.weight.shape[1]
    result,meta={},[]
    with np.load(w.checked(prior/'response_bank.npz')) as bank:
        pairs=bank['fit_pairs'].tolist();assert all(rows[i]['split']=='fit' for p in pairs for i in p)
        for s,t in pc['seed_pairs']:
            with np.load(w.checked(source_run/f'relation_s{s}_t{t}.npz')) as z:
                sg=torch.tensor(z['source_gate'],device=w.device);assert np.array_equal(z['fit_pairs'],pairs)
            with np.load(w.checked(prior/f'response_s{s}_t{t}.npz')) as z:
                old_field=torch.tensor(z['field_teacher'],device=w.device)
                old_direct=torch.tensor(z['direct_profile'],device=w.device)
            field=old_field[None].repeat(nsteps,1,1)
            direct=old_direct[None].repeat(nsteps,1,1)
            with np.load(w.checked(scalar_run/f'response_s{s}_t{t}_op0.npz')) as z:scalar0=z['direct_scalar']
            with np.load(w.checked(scalar_run/f'response_s{s}_t{t}_op1.npz')) as z:scalar1=z['direct_scalar']
            old_scalar=torch.tensor(np.stack([scalar0,scalar1],axis=1),device=w.device)
            scalar=old_scalar[None].repeat(nsteps,1,1)
            swapped=old_direct[None].repeat(nsteps,1,1)
            ds,dt=saes[s].decoder.weight.T,saes[t].decoder.weight.T
            for op in range(2):
                support=torch.where(old_field[:,op]>0)[0]
                with np.load(w.checked(prior/f'response_s{s}_t{t}_op{op}.npz')) as z:
                    a_total=torch.tensor(z['response_matrix'],device=w.device,dtype=torch.float64)
                    b=torch.tensor(z['source_response'],device=w.device,dtype=torch.float64)
                if pc.get('swapped_control'):
                    with np.load(w.checked(scalar_run/f'response_s{s}_t{t}_op{op}.npz')) as z:
                        assert np.array_equal(z['source_response'],b.float().cpu().numpy())
                        bswap=torch.tensor(z['swapped_source_response'],device=w.device,dtype=torch.float64)
                        swapped[:,:,op]=torch.tensor(z['direct_swapped'],device=w.device)[None]
                pool=torch.argsort(a_total.mean(0),descending=True,stable=True)[:cap]
                xp,yp,ap,valid=[],[],[],[]
                for off in range(0,len(pairs),cfg['batch_size']):
                    bi=off//cfg['batch_size'];h=torch.tensor(bank[f'states_{op}_{bi}'],device=w.device)
                    j=torch.tensor(bank[f'jacobians_{op}_{bi}'],device=w.device)
                    active=torch.tensor(bank[f'active_{op}_{bi}'],device=w.device)
                    ns=h.shape[1];donor=[p[1] for p in pairs[off:off+len(h)]]
                    zs=saes[s].encode(h.reshape(-1,h.shape[-1])).reshape(len(h),ns,-1)
                    zt=saes[t].encode(h.reshape(-1,h.shape[-1])).reshape(len(h),ns,-1)
                    x=(codes[t][donor,:ns]-zt)*active[:,:,None]
                    y=(((codes[s][donor,:ns]-zs)*sg[:,op])@ds)*active[:,:,None]
                    a=torch.einsum('brjd,fd->brjf',j,dt[pool])*x[:,None,:,pool]
                    if role_schema:
                        # Role0 predicts the English leading space, roles1/2 the tens/units,
                        # role3 the following token. Both templates share digit roles.
                        shift=torch.tensor([1-rows[p[0]]['template'] for p in pairs[off:off+len(h)]],device=w.device)
                        ix=torch.arange(len(h),device=w.device)[:,None]
                        role=torch.arange(ns,device=w.device)[None]+shift[:,None]
                        xx=x.new_zeros(len(h),nsteps,len(support));xx[ix,role]=x[:,:,support]
                        yy=y.new_zeros(len(h),nsteps,y.shape[-1]);yy[ix,role]=y
                        aa=a.new_zeros(len(h),nsteps,2,cap);aa[ix,role]=a.permute(0,2,1,3)
                        vv=active.new_zeros(len(h),nsteps);vv[ix,role]=active
                        xp.append(xx);yp.append(yy);ap.append(aa.permute(0,2,1,3));valid.append(vv)
                    else:
                        xp.append(torch.nn.functional.pad(x[:,:,support],(0,0,0,nsteps-ns)))
                        yp.append(torch.nn.functional.pad(y,(0,0,0,nsteps-ns)))
                        ap.append(torch.nn.functional.pad(a,(0,0,0,nsteps-ns)))
                        valid.append(torch.nn.functional.pad(active,(0,nsteps-ns)))
                x,y,a,v=torch.cat(xp).double(),torch.cat(yp).double(),torch.cat(ap).double(),torch.cat(valid)
                represented=v.any(0);n=int(represented.sum());assert represented[:n].all() and not represented[n:].any()
                # Reproduce the retained response matrix before changing its position factors.
                replay=float((a.sum(2).reshape(-1,cap)-a_total[:,pool]).abs().max())
                assert replay<pc['response_replay_atol'],replay
                d=dt[support].double();field_before=0.;field_after=0.;source_energy=float(y.square().sum())
                for pos in range(n):
                    xx,yy=x[v[:,pos],pos],y[v[:,pos],pos]
                    gram=(xx.T@xx/len(xx))*(d@d.T);rhs=(xx*(yy@d.T)).mean(0)[:,None]
                    g,info=fit(gram,rhs,steps=pc['steps'],ridge_fraction=pc['ridge_fraction'],capacity=False)
                    field[pos,support,op]=g[:,0].float()
                    field_before+=float((((xx*old_field[support,op])@d)-yy).square().sum())
                    field_after+=float((((xx*g[:,0])@d)-yy).square().sum())
                # Unobserved positions retain the original weights in every method.
                aa=a[:,:,:n].reshape(-1,n*cap)
                gram=aa.T@aa/len(aa);rhs=aa.T@b/len(aa)
                g,di=fit(gram,rhs,steps=pc['steps'],ridge_fraction=pc['ridge_fraction'],capacity=False)
                direct[:n,:,op]=0;direct[:n,pool,op]=g.reshape(n,cap).float()
                ar=a[:,:,:n].sum(3).reshape(-1,n)
                scalar_gram=ar.T@ar/len(ar)
                # Restrict the member quadratic to g[position,feature]=alpha[position].
                # Its ridge is cap times the member ridge, since each alpha is repeated.
                scalar_ridge=float(pc['ridge_fraction']*gram.diag().mean().clamp_min(1e-12)*cap/
                                   scalar_gram.diag().mean().clamp_min(1e-12))
                gs,si=fit(scalar_gram,ar.T@b/len(ar),steps=pc['steps'],ridge_fraction=scalar_ridge,capacity=False)
                scalar[:n,:,op]=0;scalar[:n,pool,op]=gs[:,0,None].float()
                if pc.get('swapped_control'):
                    gw,wi=fit(gram,aa.T@bswap/len(aa),steps=pc['steps'],ridge_fraction=pc['ridge_fraction'],capacity=False)
                    swapped[:n,:,op]=0;swapped[:n,pool,op]=gw.reshape(n,cap).float()
                meta.append(dict(source=s,target=t,operation=op,represented_positions=n,
                    field_candidates=len(support),direct_candidates=len(pool),response_replay_error=replay,
                    field_relative_mse_before=field_before/source_energy,field_relative_mse_after=field_after/source_energy,
                    response_relative_mse_direct=float((aa@g-b).square().sum()/b.square().sum()),
                    response_relative_mse_scalar=float((ar@gs-b).square().sum()/b.square().sum()),
                    position_scalars=gs[:,0].cpu().tolist(),direct_solver=di,scalar_solver=si))
                budget()
            methods=[('field',field),('direct',direct),('scalar',scalar)]
            if pc.get('swapped_control'):methods.append(('swapped',swapped))
            for name,g in methods:
                assert bool(((g>=0)&(g<=1)).all()) and bool(((g>0).any(0).sum(0)<=cap).all())
                result[t,f'position_s{s}_{name}',cap]=g
            result[t,f'position_s{s}_field_static',cap]=old_field
            np.savez_compressed(w.run/f'position_s{s}_t{t}.npz',field=field.cpu().numpy(),direct=direct.cpu().numpy(),
                scalar=scalar.cpu().numpy(),swapped=swapped.cpu().numpy() if pc.get('swapped_control') else np.empty(0),
                field_static=old_field.cpu().numpy(),fit_pairs=np.array(pairs))
            w.progress('POSITION_RELATION_FIT',source=s,target=t,source_fit_pairs=len(pairs),new_model_backwards=0)
    (w.run/'POSITION_FREEZE.json').write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        fits=meta,new_model_forwards=0,new_model_backwards=0,role_schema=role_schema,
        scope='Reuses R35 two-role bank and frozen source; fixed64member candidates, position-varying bounded weights. Unobserved positions retain original weights. Free generation supplies external evaluation.'),indent=2)+'\n')
    torch.set_float32_matmul_precision(previous_precision)
    return result,meta
