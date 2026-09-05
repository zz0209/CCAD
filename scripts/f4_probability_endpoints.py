"""Frozen descriptive probability endpoints; never used to fit/select mappings."""
import numpy as np


def log_prob(logits):
    values=np.asarray(logits,dtype=np.float64)
    if values.ndim!=2 or not np.isfinite(values).all():raise ValueError('Expected finite position/vocabulary logits')
    shifted=values-values.max(axis=-1,keepdims=True)
    return shifted-np.log(np.exp(shifted).sum(axis=-1,keepdims=True))


def endpoint_positions(tokens, interventions, boundary=0):
    """No EOS prediction, final unavailable label, or subsequent packed document."""
    tokens=np.asarray(tokens);interventions=set(interventions)
    primary=[];downstream=[];affected=False
    for p in range(len(tokens)-1):
        if tokens[p]==boundary:affected=False
        if p in interventions:affected=True
        valid=tokens[p]!=boundary and tokens[p+1]!=boundary
        if valid and p in interventions:primary.append(p)
        if valid and affected:downstream.append(p)
    return dict(intervention_positions=primary,same_document_downstream=downstream)


def probability_metrics(baseline, source, candidate, tokens, positions):
    """KL(Psource||Pcandidate)/KL(Psource||Pbaseline), pooled over positions.

    NLL deltas are intervention minus baseline; positive means observed-token
    loss increases. Relative squared error uses the source delta vector.
    """
    pos=np.asarray(positions,dtype=int);tokens=np.asarray(tokens)
    if not len(pos):return dict(positions=[],position_count=0,status='NO_VALID_NEXT_TOKEN')
    lb,ls,lc=(log_prob(np.asarray(v)[pos]) for v in (baseline,source,candidate))
    ps=np.exp(ls)
    def kl(other):
        value=np.sum(ps*(ls-other),axis=-1)
        if np.min(value)<-1e-10:raise ValueError('Negative KL exceeds numerical tolerance')
        return np.maximum(value,0.)
    kb,kc=kl(lb),kl(lc);gold=tokens[pos+1].astype(int);ix=np.arange(len(pos))
    nb,ns,nc=(-v[ix,gold] for v in (lb,ls,lc));ds=ns-nb;dc=nc-nb
    denom=float(kb.sum());se=float(ds@ds);error=float(np.sum((ds-dc)**2))
    return dict(status='MEASURED',positions=pos.tolist(),position_count=len(pos),observed_next_token_ids=gold.tolist(),
                source_to_baseline_kl_sum=denom,source_to_candidate_kl_sum=float(kc.sum()),
                source_kl_mean=float(kb.mean()),candidate_error_kl_mean=float(kc.mean()),
                normalized_kl_error=float(kc.sum()/denom) if denom>1e-12*len(pos) else None,
                baseline_nll_mean=float(nb.mean()),source_nll_mean=float(ns.mean()),candidate_nll_mean=float(nc.mean()),
                source_nll_delta_mean=float(ds.mean()),candidate_nll_delta_mean=float(dc.mean()),
                nll_delta_rmse=float(np.sqrt(error/len(pos))),source_nll_delta_rms=float(np.sqrt(se/len(pos))),
                normalized_nll_delta_squared_error=error/se if se>1e-12*len(pos) else None,
                source_nll_deltas=ds.tolist(),candidate_nll_deltas=dc.tolist(),
                source_to_baseline_kl=kb.tolist(),source_to_candidate_kl=kc.tolist())
