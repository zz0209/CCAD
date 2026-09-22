import torch
from torch import nn

from ccad.intervention_transport import project_capacity


def compute_columns(fields, codes, decoder, initial_columns, weight, steps=4):
    if fields.ndim != 3 or codes.ndim != 2 or decoder.ndim != 3:
        raise ValueError('Expected fields [b,d,p], codes [b,m], decoder [b,d,m]')
    batch, hidden_dim, parts = fields.shape
    members = codes.shape[1]
    if codes.shape[0] != batch or decoder.shape != (batch, hidden_dim, members):
        raise ValueError('Local target support dimensions do not match')
    if initial_columns.shape != (batch, members, parts):
        raise ValueError('Initial columns must use the supplied local support')
    if weight.shape != (hidden_dim, hidden_dim) or steps < 1:
        raise ValueError('Weight dimensions or recurrence depth are invalid')

    # 支持由调用者提供；零填充成员的 decoder 和初值也应为零。
    transpose = decoder.transpose(-1, -2)
    gram = transpose @ decoder
    lipschitz = gram.abs().sum(-1).amax(-1).detach().clamp_min(1e-8)
    columns = initial_columns
    for _ in range(steps):
        residual = fields - decoder @ columns
        update = transpose @ (weight @ residual)
        columns = project_capacity(columns + update / lipschitz[:, None, None], codes)
    return columns, {'lipschitz': lipschitz}


class SharedActionEncoder(nn.Module):
    def __init__(self, hidden_dim, steps=4):
        super().__init__()
        if hidden_dim < 1 or steps < 1:
            raise ValueError('Hidden dimension and recurrence depth must be positive')
        self.weight = nn.Parameter(torch.eye(hidden_dim))
        self.steps = steps

    def compute_columns(self, fields, codes, decoder, initial_columns):
        return compute_columns(fields, codes, decoder, initial_columns, self.weight, self.steps)

    def forward(self, fields, codes, decoder, initial_columns, query):
        columns, diagnostics = self.compute_columns(fields, codes, decoder, initial_columns)
        if query.ndim == 1 and query.shape == (fields.shape[-1],):
            change = columns @ query
        elif query.shape == (fields.shape[0], fields.shape[-1]):
            change = (columns @ query.unsqueeze(-1)).squeeze(-1)
        else:
            raise ValueError('Query must have shape [p] or [b,p]')
        delta = (decoder @ change.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            changed = (change != 0).sum(-1)
            counts = {
                'states': codes.shape[0],
                'changed': int(changed.sum()),
                'increased': int((change > 0).sum()),
                'max_changed_per_state': int(changed.max()),
                'minimum_final_code': float((codes + change).min()),
                'minimum_capacity_margin': float((codes - (-columns).clamp_min(0).sum(-1)).min()),
                'recurrence_steps': self.steps,
                'lipschitz_min': float(diagnostics['lipschitz'].min()),
                'lipschitz_max': float(diagnostics['lipschitz'].max()),
            }
        return delta, counts
