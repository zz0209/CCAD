import torch


def refine_request(h, target, source, query, columns, allowance, steps=128, bounded=True):
    shape = h.shape[:-1]
    x = h.reshape(-1, h.shape[-1])
    z = target.encode(x)
    zs = torch.relu((x-source['center'])@source['encoder'].T+source['encoder_bias'])
    candidate = -(target.encoder.weight@source['decoder'].T)[None]*zs[:, None, :]
    support = (candidate.abs().sum(-1)*target.decoder.weight.norm(dim=0)).topk(allowance, dim=-1).indices
    d = target.decoder.weight.T[support]
    cap = z.gather(1, support)
    a = columns.reshape(len(x), len(target.decoder.weight.T), len(query)).gather(
        1, support[..., None].expand(-1, -1, len(query)))@query
    desired = -(zs*query)@source['decoder']
    initial = a.clone()
    gram = d@d.transpose(-1, -2)
    rhs = torch.einsum('nkd,nd->nk', d, desired)
    lip = gram.abs().sum(-1).amax(-1).clamp_min(1e-10)
    current, momentum = a, 1.
    for _ in range(steps):
        proposal = current-(torch.einsum('nkj,nj->nk', gram, current)-rhs)/lip[:, None]
        updated = proposal.maximum(-cap) if bounded else proposal
        next_momentum = (1+(1+4*momentum**2)**.5)/2
        current = updated+(momentum-1)/next_momentum*(updated-a)
        a, momentum = updated, next_momentum
    before = torch.einsum('nk,nkd->nd', initial, d)
    after = torch.einsum('nk,nkd->nd', a, d)
    before_error = (before-desired).square().sum(-1)
    after_error = (after-desired).square().sum(-1)
    # 每个状态保留同一目标下误差较小的可行解。
    improve = after_error <= before_error
    a = torch.where(improve[:, None], a, initial)
    after = torch.einsum('nk,nkd->nd', a, d)
    gradient = torch.einsum('nkj,nj->nk', gram, a)-rhs
    mapping = a-(a-gradient/lip[:, None]).maximum(-cap) if bounded else gradient/lip[:, None]
    details = dict(source_energy=desired.square().sum(-1),
        common_error=before_error, request_error=(after-desired).square().sum(-1),
        kkt_mapping=mapping.square().sum(-1), minimum_code=(cap+a).amin(-1),
        active_constraints=((cap+a).abs()<1e-6).sum(-1))
    return after.reshape(*shape, h.shape[-1]), details
