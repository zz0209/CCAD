"""Finite-distribution fitting in a fixed source decoder subspace.

Behavioral distillation and L-BFGS are standard methods. Production fitting
uses DirectLayerNormHead. The retained algebraic SubspaceLayerNormHead has
double-precision identity tests but failed production floating-point checks;
it is not used by the experimental optimizer.
"""
from __future__ import annotations
import time


class SubspaceLayerNormHead:
    """Evaluate the original head at h + c D without a d-by-vocab multiply each step."""
    def __init__(self, residuals, decoder, norm, head, dtype=None):
        import torch
        self.torch=torch;dtype=dtype or residuals.dtype
        # Accumulate cached projections accurately, then fit with chosen dtype.
        with torch.no_grad():
            h=residuals.to(torch.float64);d=decoder.to(torch.float64)
            hc=h-h.mean(-1,keepdim=True);dc=d-d.mean(-1,keepdim=True)
            gamma=norm.weight.to(torch.float64) if norm.weight is not None else torch.ones(h.shape[-1],device=h.device,dtype=h.dtype)
            beta=norm.bias.to(torch.float64) if norm.bias is not None else torch.zeros_like(gamma)
            w=head.weight.detach().to(torch.float64)
            self.base_projection=((hc*gamma)@w.T).to(dtype)
            self.component_projection=((dc*gamma)@w.T).to(dtype)
            self.bias=(beta@w.T).to(dtype)
            if head.bias is not None:self.bias=self.bias+head.bias.to(dtype)
            self.base_variance=hc.square().mean(-1).to(dtype)
            self.cross=(hc@dc.T/h.shape[-1]).to(dtype)
            self.gram=(dc@dc.T/h.shape[-1]).to(dtype)
            self.eps=norm.eps;self.hidden_size=h.shape[-1]

    def logits(self, codes, ids):
        variance=self.base_variance[ids]+2*(codes*self.cross[ids]).sum(-1)+((codes@self.gram)*codes).sum(-1)
        numerator=self.base_projection[ids]+codes@self.component_projection
        return numerator/(variance+self.eps).sqrt().unsqueeze(-1)+self.bias

    def logprobs(self,codes,ids):return self.torch.log_softmax(self.logits(codes,ids),dim=-1)


class DirectLayerNormHead:
    """Original float32 production head with the actual rounded residual update.

This deliberately keeps the original matrix-multiplication kernel path when
the algebraic low-rank rearrangement differs under finite precision.
"""
    def __init__(self,residuals,decoder,norm,head):
        import torch
        self.torch=torch;self.base=residuals.to(norm.weight.dtype)
        self.decoder=decoder;self.norm=norm;self.head=head

    def logits(self,codes,ids):
        update=(codes@self.decoder).to(self.base.dtype)
        return self.head(self.norm(self.base[ids]+update))

    def logprobs(self,codes,ids):
        return self.torch.log_softmax(self.logits(codes,ids).double(),dim=-1)


def fit_finite_map(head,x,source_codes,initial,donors,masks,fit_ids,cal_ids,config,mode='matrix',progress=None,budget_check=None):
    """Fit one fixed support, with identical teacher operations across controls.

Matrix, scalar, and diagonal modes respectively change all signed allocation
entries, one global gain, or only each one-to-one atom's scalar coefficient.
The old map initializes every candidate and checkpoint zero remains eligible.
Calibration chooses an iterate, never an evaluation context or source query.
"""
    import torch
    started=time.perf_counter();device=x.device;dtype=x.dtype
    rms=x[fit_ids].square().mean(0).sqrt().clamp_min(1e-8)
    xs=x/rms;theta0=(initial*rms[:,None]).contiguous()
    source_scale=source_codes[fit_ids].square().mean(0).sqrt().clamp_min(1e-6)
    if mode=='matrix':parameter=torch.nn.Parameter(theta0.clone())
    elif mode=='scalar':parameter=torch.nn.Parameter(torch.ones((),device=device,dtype=dtype))
    elif mode=='diagonal':
        assert theta0.shape[0]==theta0.shape[1]
        assert torch.count_nonzero(theta0-torch.diag(torch.diag(theta0)))==0
        parameter=torch.nn.Parameter(torch.diag(theta0).clone())
    else:raise ValueError('Unknown finite fit parameterization')
    def theta():
        if mode=='matrix':return parameter
        if mode=='scalar':return theta0*parameter
        return torch.diag(parameter)
    rows=torch.arange(len(x),device=device);operations=[]
    with torch.no_grad():
        for name,mask in masks:
            for consumer in ['contrast','complete_removal']:
                z=source_codes[donors]-source_codes if consumer=='contrast' else -source_codes
                teacher=head.logprobs(z*mask,rows)
                operations.append(dict(name=name,consumer=consumer,mask=mask,teacher=teacher,probability=teacher.exp()))
    history=[];calls=0;queries=0
    batch=config['fit_batch_size'];total=len(fit_ids)*len(operations)
    def losses(ids,backward=False):
        nonlocal queries
        accumulated=torch.zeros((),device=device,dtype=dtype)
        for op in operations:
            for off in range(0,len(ids),batch):
                if budget_check:budget_check()
                j=ids[off:off+batch]
                design=xs[donors[j]]-xs[j] if op['consumer']=='contrast' else -xs[j]
                lp=head.logprobs((design@theta())*op['mask'],j)
                loss=(op['probability'][j]*(op['teacher'][j]-lp)).sum()/ (total if backward else len(ids)*len(operations))
                if backward:loss.backward()
                accumulated+=loss.detach();queries+=len(j)
        return accumulated
    with torch.no_grad():
        initial_train=float(losses(fit_ids));initial_cal=float(losses(cal_ids))
    history.append(dict(checkpoint=0,train_kl=initial_train,calibration_kl=initial_cal))
    best=theta0.clone();best_loss=initial_cal;best_step=0
    # Same prediction-change anchor for every information/parameter budget.
    anchor_weight=config['anchor_fraction']*max(initial_train,1e-6)
    optimizer=torch.optim.LBFGS([parameter],lr=1.,max_iter=config['lbfgs_inner_steps'],max_eval=config['lbfgs_inner_steps']*2,history_size=config['history_size'],tolerance_grad=1e-7,tolerance_change=1e-9,line_search_fn='strong_wolfe')
    for checkpoint in range(1,config['fit_checkpoints']+1):
        def closure():
            nonlocal calls
            optimizer.zero_grad(set_to_none=True)
            data=losses(fit_ids,backward=True)
            change=xs[fit_ids]@(theta()-theta0)
            penalty=anchor_weight*(change/source_scale).square().mean()
            penalty.backward();calls+=1
            value=data+penalty.detach()
            if not torch.isfinite(value) or not torch.isfinite(parameter.grad).all():raise FloatingPointError('Nonfinite finite-output fit')
            return value
        optimizer.step(closure)
        with torch.no_grad():
            train=float(losses(fit_ids));cal=float(losses(cal_ids))
            history.append(dict(checkpoint=checkpoint,train_kl=train,calibration_kl=cal,closure_evaluations=calls))
            if cal<best_loss:best_loss=cal;best=theta().detach().clone();best_step=checkpoint
        if progress:progress(checkpoint=checkpoint,training_kl=train,calibration_kl=cal,closures=calls)
    weights=(best/rms[:,None]).detach().cpu().numpy()
    return weights,dict(mode=mode,parameter_count=parameter.numel(),fit_row_count=len(fit_ids),calibration_row_count=len(cal_ids),fit_row_ids=fit_ids.cpu().tolist(),calibration_row_ids=cal_ids.cpu().tolist(),training_masks=[(n,a.detach().cpu().tolist()) for n,a in masks],consumers=['contrast','complete_removal'],history=history,selected_checkpoint=best_step,selected_calibration_kl=best_loss,anchor_weight=anchor_weight,anchor_fraction=config['anchor_fraction'],closure_evaluations=calls,fitting_head_queries=queries,teacher_head_queries=len(x)*len(operations),head_backend=type(head).__name__,wall_seconds=time.perf_counter()-started,scope='Full-vocabulary finite KL on original temporal development; signed map and support fixed across operations. Calibration chooses an iterate, not new input selection. Initial map already used all development codes, so calibration is not independent scientific evidence.')
