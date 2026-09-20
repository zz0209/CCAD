import torch


def project_capacity(values, capacity):
    negative = (-values).clamp_min(0)
    ordered = negative.sort(dim=-1, descending=True).values
    divisor = torch.arange(1, values.shape[-1] + 1, device=values.device, dtype=values.dtype)
    threshold = (ordered.cumsum(-1) - capacity[..., None]) / divisor
    rank = (ordered > threshold).sum(-1).clamp_min(1) - 1
    tau = threshold.gather(-1, rank[..., None]).clamp_min(0)
    return values.clamp_min(0) - (negative - tau).clamp_min(0)


def pursuit_columns(h, target, source, allowance, metric_root=None, steps=128,
                    candidate_limit=128, batch_size=128):
    # 同一功能目标决定共同成员及其系数，所有部分请求共享返回的列。
    shape = h.shape[:-1]
    x = h.reshape(-1, h.shape[-1])
    z = target.encode(x)
    zs = torch.relu((x - source['center']) @ source['encoder'].T + source['encoder_bias'])
    decoder = target.decoder.weight.T
    source_decoder = source['decoder']
    if metric_root is not None:
        decoder = decoder @ metric_root
        source_decoder = source_decoder @ metric_root
    norm = decoder.square().sum(-1).clamp_min(1e-12)
    cross = -decoder @ source_decoder.T
    count = min(candidate_limit, len(decoder))
    if not 0 < allowance <= count:
        raise ValueError('Member allowance must fit the candidate set')
    result = torch.zeros((len(x), len(decoder), zs.shape[-1]), device=h.device, dtype=h.dtype)
    active = (zs.abs().sum(-1) > 0).nonzero().flatten()
    for start in range(0, len(active), batch_size):
        rows = active[start:start + batch_size]
        rhs_all = cross[None] * zs[rows, None, :]
        trial = project_capacity(rhs_all / norm[None, :, None], z[rows])
        gain = (rhs_all * trial).sum(-1) - .5 * norm[None] * trial.square().sum(-1)
        candidates = gain.topk(count, dim=-1).indices
        del trial, rhs_all, gain
        d = decoder[candidates]
        gram = d @ d.transpose(-1, -2)
        rhs = cross[candidates] * zs[rows, None, :]
        cap = z[rows].gather(1, candidates)
        residual_rhs = rhs.clone()
        used = torch.zeros_like(cap, dtype=torch.bool)
        selected = []
        coefficients = []
        batch = torch.arange(len(rows), device=h.device)
        for _ in range(allowance):
            trial = project_capacity(residual_rhs / norm[candidates, None], cap)
            gain = (residual_rhs * trial).sum(-1) - .5 * norm[candidates] * trial.square().sum(-1)
            choice = gain.masked_fill(used, -torch.inf).argmax(-1)
            coefficient = trial[batch, choice]
            selected.append(choice)
            coefficients.append(coefficient)
            residual_rhs -= gram[batch, :, choice][..., None] * coefficient[:, None, :]
            used[batch, choice] = True
        local_ids = torch.stack(selected, dim=1)
        ids = candidates.gather(1, local_ids)
        a = torch.stack(coefficients, dim=1)
        d = decoder[ids]
        gram = d @ d.transpose(-1, -2)
        rhs = cross[ids] * zs[rows, None, :]
        cap = z[rows].gather(1, ids)
        lipschitz = gram.abs().sum(-1).amax(-1).clamp_min(1e-8)
        current = a
        momentum = 1.
        for _ in range(steps):
            proposal = current - (gram @ current - rhs) / lipschitz[:, None, None]
            updated = project_capacity(proposal, cap)
            next_momentum = (1 + (1 + 4 * momentum ** 2) ** .5) / 2
            current = updated + (momentum - 1) / next_momentum * (updated - a)
            a = updated
            momentum = next_momentum
        result[rows[:, None], ids] = a
    return result.reshape(*shape, len(decoder), zs.shape[-1])


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
