import torch


def paired_coefficients(encoder_rows, decoder_columns, source_fields, method='oblique'):
    if method not in ('oblique', 'orthogonal'):
        raise ValueError(f'Unknown paired projection method {method}')
    if encoder_rows.shape[-2] != decoder_columns.shape[-1]:
        raise ValueError('Encoder and decoder member dimensions differ')
    if encoder_rows.shape[-1] != decoder_columns.shape[-2] or source_fields.shape[-2] != decoder_columns.shape[-2]:
        raise ValueError('Encoder, decoder and source hidden dimensions differ')
    if encoder_rows.shape[:-2] != decoder_columns.shape[:-2]:
        raise ValueError('Encoder and decoder batch dimensions differ')
    w = encoder_rows.to(torch.float64)
    d = decoder_columns.to(torch.float64)
    # padding 不参与成员数与数值秩阈值，保持同一支持的读取规则。
    present = (w.abs().sum(-1) > 0) | (d.abs().sum(-2) > 0)
    member_count = present.sum(-1)
    relative_cutoff = member_count.clamp_min(1).to(w.dtype) * torch.finfo(torch.float32).eps
    matrix = w @ d if method == 'oblique' else d
    inverse = torch.linalg.pinv(matrix, atol=torch.zeros_like(relative_cutoff), rtol=relative_cutoff)
    field = source_fields.to(torch.float64)
    coefficients = inverse @ (w @ field) if method == 'oblique' else inverse @ field
    with torch.no_grad():
        singular = torch.linalg.svdvals(matrix.detach())
        threshold = singular[..., :1] * relative_cutoff[..., None]
        retained = singular > threshold
        rank = retained.sum(-1)
        minimum = singular.masked_fill(~retained, torch.inf).amin(-1)
        condition = torch.where(rank > 0, singular[..., 0] / minimum, torch.inf)
        denominator = torch.linalg.matrix_norm(inverse).clamp_min(torch.finfo(torch.float64).tiny)
        inverse_identity_error = torch.linalg.matrix_norm(inverse @ matrix @ inverse - inverse) / denominator
        diagnostics = dict(rank=rank, member_count=member_count, relative_cutoff=relative_cutoff,
                           condition_retained=condition, inverse_identity_relative_error=inverse_identity_error)
    return coefficients.to(source_fields.dtype), diagnostics
