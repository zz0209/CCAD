"""Separate a source carry readout from its bounded native or raw write.

Uses established ridge readouts and rank-one interventions. The experiment
tests functional reuse, not a novel regression or projection algorithm.
"""
import json
import time
import numpy as np
import torch
from run_causalgym_multisite import ROOT, write


@torch.enable_grad()
def fit_readwrite(w, cfg, model, module, tok, ae, rows, tokenrows, codes, hidden, stored, budget):
    spec = cfg['source_function_refit']['read_write']
    parent = ROOT / spec['parent']
    with np.load(w.checked(ROOT / spec['readouts'])) as d:
        readouts = {k: d[k] for k in d.files}
    with np.load(w.checked(parent / 'counterfactual_seed1_k64_carry_function_arity_scalar.npz')) as d:
        schedule = d['fit_schedule']
    ids = torch.tensor(stored['code_scalar_64'], device=w.device)
    assert np.array_equal(ids.cpu(), readouts['members']) and len(ids) == 64
    assert schedule.shape == (256, cfg['batch_size'], 2)
    assert len(rows) == 405 and all(r['split'] == 'fit' and r['template'] == 0 for r in rows)
    arity = torch.tensor([int('c' in r) for r in rows], device=w.device)
    carry = torch.tensor([float(r['carry']) for r in rows], device=w.device)
    selected = codes[:, 0, ids]
    dec = ae.decoder.weight.T[ids]
    batch = cfg['batch_size']; ix = torch.arange(batch, device=w.device)
    physical_length = max(map(len, tokenrows)) + cfg['max_new_tokens']
    payload = dict(stored)
    fit_start = time.perf_counter()

    def slopes(x):
        result = []
        for r in range(2):
            xx = x[arity == r].double().clone(); yy = carry[arity == r].double().clone()
            rr = [row for row in rows if int('c' in row) == r]
            totals = torch.tensor([row['total'] for row in rr], device=w.device)
            for total in totals.unique():
                use = totals == total
                xx[use] -= xx[use].mean(0); yy[use] -= yy[use].mean()
            result.append((yy[:, None] * xx).sum(0) / yy.square().sum())
        return torch.stack(result).float()

    for tag in spec['variants']:
        raw = tag == 'readwrite_raw'
        diagonal = tag == 'diagonal_refit_64'
        cone = tag in ['cone_fixed_64','cone_sparse_64']
        sparse = tag in ['readwrite_sparse_64','cone_sparse_64']
        rms_mode = sparse or cone or tag == 'readwrite_rescaled_64'
        x = hidden[:, 0] if raw else selected
        reader = None if diagonal else torch.tensor(readouts[('raw' if raw else 'members') + '_0.1_weights'],device=w.device,dtype=torch.float32)
        initial = torch.tensor(stored['code_scalar_64_weights'],device=w.device) if diagonal else slopes(codes[:,0] if sparse else x)
        scale = torch.ones((2,1),device=w.device) if diagonal else initial.norm(dim=1,keepdim=True).clamp_min(1e-6) / np.sqrt(initial.shape[1])
        if rms_mode:
            scale_source=codes[:,0] if sparse else selected
            scale=torch.stack([scale_source[arity==r].square().mean(0).sqrt() for r in range(2)]).clamp_min(.01)
        if cone:
            initial=torch.stack([initial.clamp_min(0),(-initial).clamp_min(0)],dim=1).flatten(0,1)
            scale=scale.repeat_interleave(2,dim=0)
        param = torch.nn.Parameter(initial / scale)
        optimizer = torch.optim.Adam([param], lr=spec['lr'])
        trace = []

        def objective(pairs, value):
            ii,jj = torch.as_tensor(pairs,device=w.device).T
            assert bool((arity[ii] == arity[jj]).all())
            targets=[];positions=[]
            tokens=torch.full((batch,physical_length),tok.eos_token_id,device=w.device,dtype=torch.long)
            attention=torch.zeros_like(tokens)
            for b,(i,j) in enumerate(pairs):
                answer=rows[i]['total'] + 10*(int(rows[j]['carry'])-int(rows[i]['carry']))
                out=tok.encode(str(answer),add_special_tokens=False)
                assert 10 <= answer < 100 and len(out)==2
                seq=tokenrows[i]+out
                tokens[b,:len(seq)]=torch.tensor(seq,device=w.device)
                attention[b,:len(seq)]=1
                positions.append(len(tokenrows[i])-1)
                targets.append(out+tok.encode('\n',add_special_tokens=False))
            pos=torch.tensor(positions,device=w.device)
            target=torch.tensor(targets,device=w.device)
            sites=pos[:,None]+torch.arange(3,device=w.device)

            def hook(_m,_a,output):
                h=output[0] if isinstance(output,tuple) else output
                current=h[ix,pos]
                z=ae.encode(current) if not raw else None
                xx=current if raw else z[:,ids]
                difference=x[jj]-xx
                weight=(value*scale)[arity[ii]]
                coeff=difference*weight if diagonal else (difference*reader[arity[ii]]).sum(-1,keepdim=True)*weight
                if cone:
                    query=(difference*reader[arity[ii]]).sum(-1,keepdim=True)
                    branch=2*arity[ii]+(query[:,0]<0).long()
                    coeff=query.abs()*(value*scale)[branch]
                selected_z=z if sparse else (z[:,ids] if not raw else None)
                decoder=ae.decoder.weight.T if sparse else dec
                delta=coeff if raw else (coeff@decoder if cone else ((selected_z+coeff).clamp_min(0)-selected_z)@decoder)
                edited=h.clone();edited[ix,pos]+=delta
                return (edited,)+output[1:] if isinstance(output,tuple) else edited

            handle=module.register_forward_hook(hook)
            try:
                output=model(tokens,attention_mask=attention,use_cache=False)
                logits=output.logits[ix[:,None],sites]
                loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.flatten())
                w.sequence_forwards+=batch;w.token_forwards+=batch*physical_length
                return loss
            finally:
                handle.remove()

        # Check a real model derivative of this new execution operator.
        loss=objective(schedule[0],param);gradient=torch.autograd.grad(loss,param)[0]
        coord=int(gradient.flatten().abs().argmax());analytic=float(gradient.flatten()[coord]);eps=.01
        with torch.no_grad():
            plus=param.detach().clone();minus=param.detach().clone()
            plus.flatten()[coord]+=eps;minus.flatten()[coord]-=eps
            finite=float((objective(schedule[0],plus)-objective(schedule[0],minus))/(2*eps))
        assert abs(finite-analytic)<.015+.05*abs(analytic),(tag,finite,analytic)
        selection_score=torch.zeros_like(param)
        for step,pairs in enumerate(schedule):
            if rms_mode and step < 16:
                probe=torch.zeros_like(param,requires_grad=True)
                loss=objective(pairs,probe)
                derivative=torch.autograd.grad(loss,probe)[0]
                selection_score-=derivative.detach()
                if step==15:
                    score=selection_score.clamp_min(0) if cone else selection_score
                    chosen=torch.argsort(score.square().sum(0),descending=True,stable=True)[:64]
                    with torch.no_grad():
                        param.zero_();param[:,chosen]=.25*score[:,chosen].sign()
                budget();continue
            optimizer.zero_grad(set_to_none=True)
            loss=objective(pairs,param);assert bool(torch.isfinite(loss))
            loss.backward();optimizer.step()
            with torch.no_grad():
                if rms_mode:
                    feasible=param.clamp(0 if cone else -4,4)
                    gain=(2*param*feasible-feasible.square()).sum(0)
                    chosen=torch.argsort(gain,descending=True,stable=True)[:64]
                    param.zero_();param[:,chosen]=feasible[:,chosen]
                elif diagonal:
                    param.clamp_(0,4)
                else:
                    bound=4*np.sqrt(param.shape[1])
                    param.mul_((bound/param.norm(dim=1,keepdim=True).clamp_min(1e-9)).clamp_max(1))
            if (step+1)%32 in [0,31]:
                trace.append({'step':step+1,'loss':float(loss.detach())})
                w.progress('CARRY_READ_WRITE_FIT',method=tag,**trace[-1])
            budget()
        weight=(param.detach()*scale).cpu().numpy()
        active=torch.where((param.detach()!=0).any(0))[0] if sparse else ids
        payload[tag]=active.cpu().numpy() if not raw else np.array([],dtype=np.int64)
        if diagonal:
            payload[tag+'_weights']=weight;payload[tag+'_gain_bound']=np.array(4.)
        else:
            payload[tag+'_readout']=reader.cpu().numpy();payload[tag+'_writer']=weight[:,active.cpu().numpy()] if sparse else weight
            payload[tag+'_operator']=np.array('raw' if raw else ('cone' if cone else 'code'))
            if sparse:payload[tag+'_read_indices']=ids.cpu().numpy()
        payload[tag+'_nonnegative_update']=np.array(not raw)
        np.savez_compressed(w.run/f'{tag}_fit.npz',initial=initial.detach().cpu().numpy(),final=weight,
                            schedule=schedule,member_ids=ids.cpu().numpy(),**({} if diagonal else {'reader':reader.cpu().numpy()}))
        write(w.run/f'{tag}_fit.json',dict(steps=256,learning_rate=spec['lr'],trace=trace,
            reader='Fixed ridge on the405sourcefit carrylabels; shrinkage selected withinfit' if not diagonal else None,
            gradient=dict(analytic=analytic,finite=finite,error=abs(finite-analytic)),
            scope='Fixed64read members for native methods;64write allowance, selected from full dictionary for sparse variant. Raw uses1536hidden dimensions.256backwards; initialization and projection recorded separately.',
            normalization=('Positive addition per sign/arity, code-RMS normalization, weights[0,4], shared64write union;16selection+240optimization.' if cone else ('Signed writer normalized by source-code RMS; weights in[-4,4], shared64write support, fixed64read support.16selection+240optimization; source support relearned only for sparse variant.' if rms_mode else 'Code/raw writer row norm capped at4times initial slope norm; diagonal gains in[0,4].')),
            fit_only=True,elapsed_since_all_fits_start=time.perf_counter()-fit_start))
        w.checks[f'carry_readwrite_gradient_{tag}']=True
    if spec.get('unclipped_control'):
        name='unclipped_code_64'
        for suffix in ['', '_readout','_writer']:
            payload[name+suffix]=stored['readwrite_code_64'+suffix]
        payload[name+'_operator']=np.array('linear_code')
        payload[name+'_nonnegative_update']=np.array(False)
    return payload
