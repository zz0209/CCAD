"""Realize a predicted source request with bounded target donor changes.

This is batched bounded least squares, not a new optimization algorithm.
The target dictionary and the paired recipient/donor codes define the feasible
set; no model outputs or desired answer labels enter this projection.
"""
import time


def realize(field, change, decoder, members=64, steps=256, refine_steps=128,
            batch_size=128, candidate_limit=128, current_codes=None, support_mask=None):
    import torch
    original_shape=field.shape[:-1]
    field=field.flatten(0,-2).float(); change=change.flatten(0,-2).float()
    if current_codes is not None:current_codes=current_codes.flatten(0,-2).float()
    if support_mask is not None:support_mask=support_mask.flatten(0,-2).bool()
    decoder=decoder.float(); norm=decoder.norm(dim=1)
    count=min(candidate_limit,change.shape[-1]); outputs=[]; coefficients=[]; indices=[]
    dense_errors=[]; sparse_errors=[]; active_counts=[]; seconds=time.perf_counter()
    for start in range(0,len(field),batch_size):
        x=change[start:start+batch_size]; y=field[start:start+batch_size]
        # Each TopK code has at most k nonzeros. The donor difference has at
        # most 2k; candidate_limit=2k retains every active coefficient.
        active=(x!=0).sum(1)
        if current_codes is None:
            assert int(active.max())<=candidate_limit
            ix=torch.argsort(x.abs()*norm,dim=1,descending=True,stable=True)[:,:count]
        else:
            cc=current_codes[start:start+batch_size]
            cross=y@decoder.T
            trial=(cross/norm.square().clamp_min(1e-12)).maximum(-cc)
            improvement=cross*trial-.5*norm.square()*trial.square()
            if support_mask is not None:improvement=improvement.masked_fill(~support_mask[start:start+batch_size],-torch.inf)
            ix=torch.argsort(improvement,dim=1,descending=True,stable=True)[:,:count]
        xx=x.gather(1,ix); d=decoder[ix]
        gram=d@d.transpose(1,2); rhs=torch.einsum('nkd,nd->nk',d,y)
        low=xx.clamp_max(0) if current_codes is None else -cc.gather(1,ix)
        high=xx.clamp_min(0) if current_codes is None else torch.full_like(low,torch.inf)
        if support_mask is not None:
            allowed=support_mask[start:start+batch_size].gather(1,ix)
            low=torch.where(allowed,low,0);high=torch.where(allowed,high,0)
        lip=gram.abs().sum(-1).amax(-1,keepdim=True).clamp_min(1e-10)
        def solve(initial,mask,nsteps):
            lo=low*mask;hi=torch.where(mask>0,high,0);c=initial.clone();v=c.clone();t=1.
            for _ in range(nsteps):
                grad=torch.bmm(gram,v.unsqueeze(-1)).squeeze(-1)-rhs
                new=(v-grad/lip).maximum(lo).minimum(hi)
                tn=(1+(1+4*t*t)**.5)/2
                v=new+(t-1)/tn*(new-c);c=new;t=tn
            return c
        dense=solve(torch.zeros_like(xx),torch.ones_like(xx),steps)
        mask=torch.zeros_like(xx)
        keep=torch.argsort(dense.abs()*norm[ix],dim=1,descending=True,stable=True)[:,:min(members,count)]
        mask.scatter_(1,keep,1)
        sparse=solve(dense*mask,mask,refine_steps)
        pred=torch.einsum('nk,nkd->nd',sparse,d)
        dense_pred=torch.einsum('nk,nkd->nd',dense,d)
        assert bool((sparse>=low-1e-6).all() and (sparse<=high+1e-6).all())
        assert int((sparse!=0).sum(1).max())<=members
        outputs.append(pred);coefficients.append(sparse);indices.append(ix)
        dense_errors.append((dense_pred-y).square().sum(1));sparse_errors.append((pred-y).square().sum(1))
        active_counts.append(active)
    prediction=torch.cat(outputs).reshape(*original_shape,decoder.shape[1])
    coeff=torch.cat(coefficients).reshape(*original_shape,count)
    ix=torch.cat(indices).reshape(*original_shape,count)
    diagnostics=dict(solver='projected FISTA; bounded coefficients then top-k and fixed-support refit',
        constraints='target codes remain nonnegative' if current_codes is not None else 'target donor interpolation',
        steps=steps,refine_steps=refine_steps,members=members,candidate_limit=candidate_limit,
        wall_seconds=time.perf_counter()-seconds,
        dense_relative_mse=float(torch.cat(dense_errors).sum()/field.square().sum().clamp_min(1e-12)),
        sparse_relative_mse=float(torch.cat(sparse_errors).sum()/field.square().sum().clamp_min(1e-12)),
        active_candidates_mean=float(torch.cat(active_counts).float().mean()),
        active_candidates_max=int(torch.cat(active_counts).max()))
    return prediction,coeff,ix,diagnostics
