"""Match a predicted source request's response using a fixed target code bank.

The model and dictionaries stay frozen. Only the current intervention's code
changes are optimized, with a nonnegative edited code at every position.
This is projected output distillation, not a new optimization algorithm.
"""
import time


def refine(model, module, ids, attention, sites, logit_sites, teacher_logits,
           decoder, initial, lower, teacher_field, work, steps=12, lr=.25,
           field_anchor=0., parameterization="members"):
    import torch
    start=time.perf_counter();batch=initial.shape[0]
    ix=torch.arange(batch,device=ids.device)
    teacher_logp=teacher_logits.detach().float().log_softmax(-1)
    teacher_p=teacher_logp.exp()
    assert parameterization in ["members", "clipped_scalar"]
    c0=initial.detach().clone()
    variable=(torch.ones_like(c0[..., :1]) if parameterization=="clipped_scalar" else c0.clone()).requires_grad_(True)
    opt=torch.optim.Adam([variable],lr=lr)
    def coefficients():
        return torch.maximum(c0*variable,lower) if parameterization=="clipped_scalar" else variable
    best=c0.clone();best_loss=torch.full((batch,),torch.inf,device=c0.device)
    first=None;history=[];best_kl=torch.zeros_like(best_loss)
    energy=teacher_field.square().sum((1,2)).clamp_min(1e-8)

    def forward(change):
        def hook(_m,_a,out):
            h=out[0] if isinstance(out,tuple) else out
            hh=h.clone();hh[ix[:,None],sites]+=change
            return (hh,)+out[1:] if isinstance(out,tuple) else hh
        handle=module.register_forward_hook(hook)
        try:
            output=model(ids,attention_mask=attention,use_cache=False)
            work.sequence_forwards+=len(ids);work.token_forwards+=ids.numel()
            return output.logits[ix,logit_sites].float()
        finally:handle.remove()

    with torch.enable_grad():
        for it in range(steps+1):
            c=coefficients()
            field=c@decoder
            logits=forward(field)
            kl=(teacher_p*(teacher_logp-logits.log_softmax(-1))).sum(-1)
            mse=(field-teacher_field).square().sum((1,2))/energy
            loss=kl+field_anchor*mse
            with torch.no_grad():
                improve=loss<best_loss
                best[improve]=c[improve];best_loss[improve]=loss[improve];best_kl[improve]=kl[improve]
                if first is None:first=dict(kl=kl.detach().cpu().tolist(),relative_mse=mse.detach().cpu().tolist())
                history.append(dict(iteration=it,mean_kl=float(kl.mean()),mean_relative_mse=float(mse.mean())))
            if it==steps:break
            opt.zero_grad(set_to_none=True);loss.sum().backward();opt.step()
            with torch.no_grad():
                if parameterization=="clipped_scalar":variable.clamp_min_(0.)
                else:variable.copy_(torch.maximum(variable,lower))
    with torch.no_grad():
        field=best@decoder;logits=forward(field)
        final_kl=(teacher_p*(teacher_logp-logits.log_softmax(-1))).sum(-1)
        code=best-lower
        assert float(code.min())>=-1e-6
    diagnostics=dict(steps=steps,lr=lr,field_anchor=field_anchor,parameterization=parameterization,optimized_parameters=variable.numel(),
        additional_forward_batches=steps+2,backward_batches=steps,
        solver_seconds=time.perf_counter()-start,initial=first,
        final_kl=final_kl.cpu().tolist(),min_edited_code=float(code.min()),
        max_changed_members=int((best!=0).sum(-1).max()),history=history)
    return logits,field,diagnostics
