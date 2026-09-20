import torch

from ccad.intervention_transport import allocate_columns
from ccad.request_inference import solve_request_columns


def infer_source_fields(h, target, source_decoder, amplitudes, allowance, metric_root=None, steps=128):
    if amplitudes.shape[:-1] != h.shape[:-1] or source_decoder.shape != (amplitudes.shape[-1], h.shape[-1]):
        raise ValueError('Source amplitudes, decoder and hidden states have incompatible shapes')
    shape = h.shape[:-1]
    x = h.reshape(-1, h.shape[-1])
    amplitude = amplitudes.reshape(len(x), -1)
    p = amplitude.shape[-1]
    z = target.encode(x)
    if not 0 < allowance <= z.shape[-1]:
        raise ValueError('Member allowance exceeds the target dictionary')
    candidates = (target.encoder.weight @ source_decoder.T)[None] * amplitude[:, None, :]
    initial = allocate_columns(z, candidates, target.decoder.weight, allowance, active_only=False)
    score = candidates.abs().sum(-1) * target.decoder.weight.norm(dim=0)
    ids = score.topk(allowance, dim=-1).indices
    a = initial.gather(1, ids[..., None].expand(-1, -1, p))
    d = target.decoder.weight.T[ids]
    y = source_decoder.T[None] * amplitude[:, None, :]
    if metric_root is not None:
        d = d @ metric_root
        y = metric_root.T @ y
    gram = d @ d.transpose(-1, -2)
    rhs = d @ y
    capacity = z.gather(1, ids)
    a = solve_request_columns(gram, rhs, capacity, a, torch.eye(p, device=h.device, dtype=h.dtype), steps)
    result = torch.zeros_like(candidates).scatter(1, ids[..., None].expand(-1, -1, p), a)
    return result.reshape(*shape, z.shape[-1], p)
