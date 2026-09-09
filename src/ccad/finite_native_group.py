"""Fit actual finite LM responses while keeping a fixed native donor-mixing group."""
from __future__ import annotations
import time

def project_box_mass(x,budget):
    """Euclidean projection onto [0,1]^m with sum at most budget."""
    import torch
    y=x.clamp(0,1)
    if float(y.sum())<=budget:return y
    lo=float(x.min())-1;hi=float(x.max())
    for _ in range(40):
        mid=(lo+hi)/2
        if float((x-mid).clamp(0,1).sum())>budget:lo=mid
        else:hi=mid
    return (x-hi).clamp(0,1)

def fit_native(api,rows,region,dz,decoder,initial,*,budget,pool_size=256,steps=64,batch_size=16,lr=.1,reference=None):
    """Source: donor full-vocabulary CE. Target: source full-vocabulary KL.

    A relaxed box/mass phase finds a support; the final quarter refits that fixed
    support. Support selection is heuristic and training intermediates can be
    dense. The returned deployment weights have at most budget positive entries.
    """
    import torch
    timer=time.perf_counter();device=dz.device;N=len(rows);assert len(dz)==N
    # Actual loss gradients at no edit expose features omitted by a local
    # source-margin contribution pool. This uses training data only.
    zero=torch.nn.Parameter(torch.zeros(dz.shape[1],device=device));gains=torch.zeros_like(zero)
    def loss(local,w,dec,x):
        batch=api.batch([rows[i] for i in local]);bp,_=api.positions(batch,region)
        v=api.forward(batch,positions=bp,delta=(x[local]*w)@dec,differentiable=True)
        if reference is None:out=-v['log_probs'][torch.arange(len(local),device=device),batch.src_labels].mean()
        else:
            ref=reference[local];out=(ref.exp()*(ref-v['log_probs'])).sum(1).mean()
        return out
    for off in range(0,N,batch_size):
        local=list(range(off,min(off+batch_size,N)));v=loss(local,zero,decoder,dz)
        gains-=torch.autograd.grad(v,zero)[0]*len(local)/N;api.backward_sequences+=len(local)
    score=gains.clamp_min(0);pool=torch.argsort(score,descending=True,stable=True)[:pool_size]
    param=torch.nn.Parameter(initial[pool].clone());optimizer=torch.optim.Adam([param],lr=lr)
    x=dz[:,pool];dec=decoder[pool];trace=[];support_mask=None;cut=steps*3//4
    generator=torch.Generator(device='cpu').manual_seed(420);order=torch.randperm(N,generator=generator).tolist()
    for step in range(steps):
        if step==cut:
            with torch.no_grad():
                keep=torch.argsort(param,descending=True,stable=True)[:budget];support_mask=torch.zeros_like(param);support_mask[keep]=1;param.mul_(support_mask)
            optimizer=torch.optim.Adam([param],lr=lr*.5)
        if (step*batch_size)%N==0:order=torch.randperm(N,generator=generator).tolist()
        local=[order[(step*batch_size+j)%N] for j in range(batch_size)];optimizer.zero_grad(set_to_none=True)
        effective=param if support_mask is None else param*support_mask
        value=loss(local,effective,dec,x);value.backward();api.backward_sequences+=len(local)
        optimizer.step()
        with torch.no_grad():
            param.copy_(project_box_mass(param,budget))
            if support_mask is not None:param.mul_(support_mask)
        if step%8==0 or step==steps-1:trace.append(dict(step=step+1,loss=float(value.detach()),positive=int((param>1e-7).sum()),mass=float(param.detach().sum()),phase='relaxed' if step<cut else 'fixed_support'))
    result=torch.zeros(dz.shape[1],device=device);result[pool]=param.detach();assert int((result>1e-7).sum())<=budget
    return result,dict(loss='original donor-label full-vocabulary CE' if reference is None else 'full-vocabulary KL from actual source edit',steps=steps,batch_size=batch_size,unique_train_rows=N,pool=pool.cpu().tolist(),initial_loss_gradient=gains.detach().cpu().tolist(),trace=trace,wall_seconds=time.perf_counter()-timer,positive=int((result>1e-7).sum()),mass=float(result.sum()),scope='Finite response fit. Training relaxation has a mass cap; final deployment uses a fixed cardinality cap. No global optimality claim.')

def fit_das(api,rows,region,delta,*,rank=1,steps=100,batch_size=4,lr=.005):
    """Standard low-rank orthogonal interchange with the original CE endpoint.

    This adapter uses QR reparameterization rather than pyvene's parametrization,
    so it is a DAS-style comparator, not an unmodified original solver replay.
    """
    import torch
    timer=time.perf_counter();device=delta.device;generator=torch.Generator(device=device).manual_seed(42)
    weight=torch.nn.Parameter(torch.randn(delta.shape[1],rank,device=device,generator=generator)/delta.shape[1]**.5);opt=torch.optim.Adam([weight],lr=lr);N=len(rows);trace=[]
    gen=torch.Generator().manual_seed(42)
    for step in range(steps):
        local=torch.randint(N,(batch_size,),generator=gen).tolist();batch=api.batch([rows[i] for i in local]);bp,_=api.positions(batch,region)
        basis=torch.linalg.qr(weight,mode='reduced').Q;qt=(delta[local]@basis)@basis.T
        out=api.forward(batch,positions=bp,delta=qt,differentiable=True);loss=-out['log_probs'][torch.arange(batch_size,device=device),batch.src_labels].mean()
        opt.zero_grad(set_to_none=True);loss.backward();api.backward_sequences+=batch_size
        scale=(step+1)/max(1,steps*.1) if step<steps*.1 else (steps-step)/(steps*.9)
        for group in opt.param_groups:group['lr']=lr*scale
        opt.step()
        if step%20==0 or step==steps-1:trace.append(dict(step=step+1,loss=float(loss.detach())))
    basis=torch.linalg.qr(weight.detach(),mode='reduced').Q
    return basis,dict(rank=rank,steps=steps,batch_size=batch_size,lr=lr,train_rows=N,trace=trace,wall_seconds=time.perf_counter()-timer,objective='original full-vocabulary donor CE',scope='DAS-style QR basis adapter, original data/labels/IIA; not pyvene solver reproduction.')


def fit_compiled(api,rows,region,alpha,initial,unit,reference,*,decoder=None,steps=64,batch_size=16,lr=.005,penalty=.001):
    """Fit two fixed native directions to actual source output distributions.

    With decoder[sign,member,hook], parameters are nonnegative code weights
    on the two previously fixed supports. With no decoder, parameters are
    unrestricted hidden directions: a matched-information two-direction
    control. Both use the same target-only scalar, source KL, direction-space
    penalty, update count and sampled training rows. No held rows enter.
    """
    import torch
    timer=time.perf_counter();device=alpha.device;N=len(rows)
    assert len(alpha)==len(reference)==N
    param=torch.nn.Parameter(initial.detach().clone());opt=torch.optim.Adam([param],lr=lr)
    gen=torch.Generator().manual_seed(420);trace=[]
    def directions():return param if decoder is None else (param.unsqueeze(1)@decoder).squeeze(1)
    for step in range(steps):
        local=torch.randint(N,(batch_size,),generator=gen).tolist()
        batch=api.batch([rows[i] for i in local]);bp,_=api.positions(batch,region)
        v=directions();a=alpha[local];qt=a.abs()[:,None]*v[(a<0).long()]
        out=api.forward(batch,positions=bp,delta=qt,differentiable=True);ref=reference[local]
        kl=(ref.exp()*(ref-out['log_probs'])).sum(1).mean()
        direction_error=(v-unit).square().sum(1).mean();loss=kl+penalty*direction_error
        opt.zero_grad(set_to_none=True);loss.backward();api.backward_sequences+=batch_size
        torch.nn.utils.clip_grad_norm_([param],1.);opt.step()
        if decoder is not None:
            with torch.no_grad():param.clamp_(min=0)
        if step%8==0 or step==steps-1:trace.append(dict(step=step+1,source_kl=float(kl.detach()),direction_squared_error=float(direction_error.detach()),loss=float(loss.detach())))
    assert bool(torch.isfinite(param).all())
    return param.detach(),dict(steps=steps,batch_size=batch_size,lr=lr,penalty=penalty,gradient_norm_cap=1.,unique_train_rows=N,parameters=param.numel(),native=decoder is not None,trace=trace,wall_seconds=time.perf_counter()-timer,objective='Source full-vocabulary KL plus the same mean squared unit-direction penalty',scope='Fixed sign-specific code supports with nonnegative weights, or unrestricted two-direction control. Same target scalar, original source training responses and sampled update budget. Readout is unchanged; no target test labels or held source effects enter fitting. No global optimum claim.')
