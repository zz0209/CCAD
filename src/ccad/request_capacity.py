import torch


def part_request_columns(source_codes, basis, part_ids, target_codes, norms, allowance, member_support=False):
    parts = torch.unique(part_ids, sorted=True)
    columns = torch.stack([source_codes[:, part_ids == part] @ basis[:, part_ids == part].T
                           for part in parts], dim=-1)
    active = target_codes > 0
    score = (source_codes@basis.abs().T if member_support else columns.abs().sum(-1))*active*norms
    ids = score.topk(allowance, dim=-1).indices
    selected = active*torch.zeros_like(score).scatter(-1, ids, 1.)
    columns = columns*selected.unsqueeze(-1)
    positive, negative = columns.clamp_min(0), (-columns).clamp_min(0)
    negative_sum = negative.sum(-1)
    scale = (target_codes/negative_sum.clamp_min(1e-20)).clamp_max(1)
    return positive-negative*scale.unsqueeze(-1), parts, scale, negative_sum
