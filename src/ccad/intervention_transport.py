import torch


def allocate_columns(z, columns, decoder, allowance, active_only=True, capacity_first=False):
    """Select a common support and allocate feasible negative capacity."""
    if active_only:
        columns = columns * (z > 0).unsqueeze(-1)
    if capacity_first:
        positive, negative = columns.clamp_min(0), (-columns).clamp_min(0)
        scale = (z / negative.sum(-1).clamp_min(1e-20)).clamp_max(1)
        columns = positive - negative * scale.unsqueeze(-1)
    score = columns.abs().sum(-1) * decoder.norm(dim=0)
    selected = torch.zeros_like(score).scatter(-1, score.topk(allowance, dim=-1).indices, 1)
    columns = columns * selected.unsqueeze(-1)
    positive, negative = columns.clamp_min(0), (-columns).clamp_min(0)
    scale = (z / negative.sum(-1).clamp_min(1e-20)).clamp_max(1)
    return positive - negative * scale.unsqueeze(-1)


def transport_delta(h, target, source, q, write_matrix, allowance, active_only=True, capacity_first=False):
    # 对应列先共同分配成员和可删除容量，再接受部分请求。
    z = target.encode(h)
    zs = torch.relu((h - source['center']) @ source['encoder'].T + source['encoder_bias'])
    basis = -(write_matrix @ source['decoder'].T)
    columns = basis * zs.unsqueeze(-2)
    allocated = allocate_columns(z, columns, target.decoder.weight, allowance, active_only, capacity_first)
    dz = allocated @ q
    return dz @ target.decoder.weight.T, dz, allocated


def refine_columns(h, target, source, columns, allowance, steps=64, write_matrix=None, metric_root=None, accelerate=False):
    """Refine a fixed common support by projected least squares, without labels.

    Projection bounds the sum of negative parts in each row by its current code.
    The positive part is unchanged by this projection. All q in [0,1]^p remain
    feasible. This is a diagnostic convex subproblem on an already chosen support.
    """
    z=target.encode(h)
    shape=h.shape[:-1];n=h.numel()//h.shape[-1];p=columns.shape[-1]
    flat=columns.reshape(n,columns.shape[-2],p)
    zs=torch.relu((h-source['center'])@source['encoder'].T+source['encoder_bias']).reshape(n,p)
    matrix=target.encoder.weight if write_matrix is None else write_matrix
    candidates=-(matrix@source['decoder'].T)[None]*zs[:,None,:]
    score=candidates.abs().sum(-1)*target.decoder.weight.norm(dim=0)
    ids=score.topk(allowance,dim=-1).indices
    a=flat.gather(1,ids[...,None].expand(-1,-1,p))
    d=target.decoder.weight.T[ids]
    cap=z.reshape(n,-1).gather(1,ids)
    y=-source['decoder'].T[None]*zs[:,None,:]
    if metric_root is not None:
        d=d@metric_root
        y=metric_root.T@y
    gram=d@d.transpose(-1,-2);rhs=d@y
    lipschitz=(gram.abs().sum(-1).amax(-1) if accelerate else torch.linalg.eigvalsh(gram)[...,-1]).clamp_min(1e-8)
    divisor=torch.arange(1,p+1,device=h.device,dtype=h.dtype)
    current=a; momentum=1.
    for _ in range(steps):
        proposal=current-(gram@current-rhs)/lipschitz[:,None,None]
        negative=(-proposal).clamp_min(0)
        ordered=negative.sort(dim=-1,descending=True).values
        threshold=(ordered.cumsum(-1)-cap[...,None])/divisor
        rank=(ordered>threshold).sum(-1).clamp_min(1)-1
        tau=threshold.gather(-1,rank[...,None]).clamp_min(0)
        updated=proposal.clamp_min(0)-(negative-tau).clamp_min(0)
        next_momentum=(1+(1+4*momentum**2)**.5)/2 if accelerate else 1.
        current=updated+(momentum-1)/next_momentum*(updated-a)
        a=updated; momentum=next_momentum
    result=torch.zeros_like(flat).scatter(1,ids[...,None].expand(-1,-1,p),a)
    return result.reshape(*shape,columns.shape[-2],p)
