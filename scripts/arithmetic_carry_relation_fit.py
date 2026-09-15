"""Translate a frozen carry operation using only its designated source fit states."""
import json
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from arithmetic_counterfactual_fit import project_role_members
from run_causalgym_multisite import ROOT, write


def fit_relation(w, cfg, seed, ae, source_ae, budget):
    spec=cfg['carry_relation_fit']; parent=ROOT/spec['source_run']
    source_cfg=json.loads(w.checked(parent/'config.resolved.json').read_text())
    assert json.loads(w.checked(parent/'status.json').read_text())['status']=='PASS'
    assert all(source_cfg[k]==cfg[k] for k in ['training_run','checkpoint_step','model_revision'])
    fit_panel=json.loads(w.checked(parent/'SOURCE_FIT_PANEL.json').read_text())
    rows=fit_panel['rows']; assert len(rows)==405 and all(r['split']=='fit' for r in rows)
    old=ROOT/source_cfg['source_function_refit']['fit_cache_run']
    pieces=[]; source_pieces=[]
    for run,indices in [(old,fit_panel['original_indices']),(parent,fit_panel['new_indices'])]:
        with np.load(w.checked(run/'states.npz')) as d:pieces.append(d['hidden'][indices,0])
        code_file='source_seed1.npz' if run==old else 'evaluation_seed1.npz'
        with np.load(w.checked(run/code_file)) as d:source_pieces.append(d['codes'][indices,0])
    hidden=torch.tensor(np.concatenate(pieces),device=w.device)
    source_z=torch.tensor(np.concatenate(source_pieces),device=w.device)
    z=ae.encode(hidden); decoder=ae.decoder.weight.T
    with np.load(w.checked(parent/'rule_members_seed1.npz')) as d:stored={k:d[k] for k in d.files}
    source_ids=torch.tensor(stored[spec['source_method']],device=w.device)
    source_weights=torch.tensor(stored[spec['source_method']+'_weights'],device=w.device)
    source_decoder=source_ae.decoder.weight.T[source_ids]
    assert source_weights.shape==(2,len(source_ids))
    generator=torch.Generator().manual_seed(spec['seed']); chosen=[]
    for arity in [2,3]:
        ids=torch.tensor([i for i,r in enumerate(rows) if (3 if 'c' in r else 2)==arity])
        pp=torch.cartesian_prod(ids,ids); pp=pp[pp[:,0]!=pp[:,1]]
        chosen.append(pp[torch.randperm(len(pp),generator=generator)[:spec['pairs']//2]])
    pairs=torch.cat(chosen).to(w.device); a,b=pairs.T
    arities=torch.tensor([int('c' in rows[i]) for i in a.tolist()],device=w.device)
    source_current=source_z[a][:,source_ids]
    source_delta=(source_z[b][:,source_ids]-source_current)*source_weights[arities]
    teacher=((source_current+source_delta).clamp_min(0)-source_current)@source_decoder
    raw=torch.tensor(stored['raw_mixed_direction'],device=w.device)
    raw_teacher=((hidden[b]-hidden[a])@raw)[:,None]*raw
    full_x=z[b]-z[a]; payload={'raw_direction':stored['raw_direction'], 'raw_mixed_direction':stored['raw_mixed_direction']}
    cosine=(source_decoder/source_decoder.norm(dim=1,keepdim=True))@(decoder/decoder.norm(dim=1,keepdim=True)).T
    src,assigned=linear_sum_assignment(-cosine.cpu().numpy());assert np.array_equal(src,np.arange(len(source_ids)))
    payload.update(assignment_64=assigned,assignment_64_weights=source_weights.cpu().numpy(),
                   assignment_64_nonnegative_update=np.array(True),assignment_64_gain_bound=np.array(4.))
    for tag,target in [('translated',teacher),('direct',raw_teacher)]:
        influence=torch.stack([(full_x[arities==r]*(target[arities==r]@decoder.T)).mean(0) for r in range(2)])
        bank=torch.argsort(influence.clamp_min(0).square().sum(0),descending=True,stable=True)[:spec['pool']]
        x=full_x[:,bank];current=z[a][:,bank];d=decoder[bank]
        scale=target.square().sum().clamp_min(1e-10)
        with torch.enable_grad():
            g=torch.zeros((2,len(bank)),device=w.device);g[:,:64]=.5;g.requires_grad_()
            optimizer=torch.optim.Adam([g],lr=spec['lr']);trace=[]
            for update in range(spec['steps']):
                optimizer.zero_grad()
                delta=(current+x*g[arities]).clamp_min(0)-current
                loss=((delta@d-target).square().sum()/scale)
                loss.backward();optimizer.step()
                with torch.no_grad():
                    q=g.detach().clone()/4;project_role_members(q,64);g.copy_(q*4)
                if (update+1)%32==0:trace.append([update+1,float(loss.detach())])
        selected=torch.where((g.detach()>0).any(0))[0];assert len(selected)<=64
        name=tag+'_code_64';payload[name]=bank[selected].cpu().numpy()
        payload[name+'_weights']=g.detach()[:,selected].cpu().numpy()
        payload[name+'_nonnegative_update']=np.array(True);payload[name+'_gain_bound']=np.array(4.)
        np.savez_compressed(w.run/f'carry_relation_seed{seed}_{tag}.npz',pairs=pairs.cpu().numpy(),
            bank=bank.cpu().numpy(),weights=g.detach().cpu().numpy(),teacher=target.cpu().numpy(),influence=influence.cpu().numpy())
        write(w.run/f'CARRY_RELATION_seed{seed}_{tag}.json',dict(source_method=spec['source_method'],
            fit_rows=405,fit_pairs=len(pairs),members=len(selected),source_fit_only=True,fit=spec,trace=trace,
            target='Frozen source code update' if tag=='translated' else 'Same mixed-source raw carry projection',
            information='No target edited answers or output gradients. All fit states are source-training inputs.'))
        w.progress('CARRY_RELATION_FITTED',seed=seed,teacher=tag,loss=trace[-1][1],members=len(selected));budget()
    np.savez_compressed(w.run/f'rule_members_seed{seed}.npz',**payload)
    w.checks[f'carry_relation_source_fit_only_seed{seed}']=True
    return payload
