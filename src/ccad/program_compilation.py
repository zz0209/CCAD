import torch


def project_program_rows(values):
    negative = (-values).clamp_min(0)
    ordered = negative.sort(dim=-1, descending=True).values
    divisor = torch.arange(1, values.shape[-1] + 1, device=values.device, dtype=values.dtype)
    thresholds = (ordered.cumsum(-1) - 1) / divisor
    ranks = (ordered > thresholds).sum(-1).clamp_min(1) - 1
    threshold = thresholds.gather(-1, ranks.unsqueeze(-1)).clamp_min(0)
    return values.clamp_min(0) - (negative - threshold).clamp_min(0)


@torch.no_grad()
def compile_program(z, decoder, fields, steps=512):
    if z.ndim != 2 or decoder.ndim != 2 or fields.ndim != 3:
        raise ValueError('Expected z[N,F], decoder[d,F], fields[N,G,d]')
    n, width = z.shape
    if n == 0 or width == 0 or fields.shape[1] == 0:
        raise ValueError('Compilation requires states, members, and parts')
    if decoder.shape[1] != width or fields.shape[0] != n or fields.shape[2] != decoder.shape[0]:
        raise ValueError('Incompatible compilation dimensions')
    if z.device != decoder.device or z.device != fields.device:
        raise ValueError('Compilation inputs must share a device')
    if not isinstance(steps, int) or steps < 1:
        raise ValueError('Compilation requires a positive integer step count')
    if not all(bool(torch.isfinite(value).all()) for value in (z, decoder, fields)):
        raise ValueError('Compilation inputs must be finite')
    if bool((z < 0).any()):
        raise ValueError('Target codes must be nonnegative')
    codes, atoms, targets = z.double(), decoder.double(), fields.double()
    gram = (codes.T @ codes) * (atoms.T @ atoms) / n
    projected_targets = torch.einsum('ngd,df->ngf', targets, atoms)
    rhs = torch.einsum('nf,ngf->fg', codes, projected_targets) / n
    lipschitz = torch.linalg.eigvalsh(gram)[-1]
    if not bool(torch.isfinite(lipschitz)) or float(lipschitz) <= 0:
        raise ValueError('Compilation Gram must have positive finite maximum eigenvalue')
    constant = targets.square().sum() / n
    coefficients = torch.zeros_like(rhs)
    extrapolated = coefficients.clone()
    momentum = 1.
    history = [dict(iteration=0, objective=float(constant))]
    for iteration in range(1, steps + 1):
        candidate = extrapolated - (gram @ extrapolated - rhs) / lipschitz
        updated = project_program_rows(candidate)
        next_momentum = (1 + (1 + 4 * momentum ** 2) ** .5) / 2
        extrapolated = updated + (momentum - 1) / next_momentum * (updated - coefficients)
        coefficients, momentum = updated, next_momentum
        if iteration % 64 == 0 or iteration == steps:
            objective = constant + (coefficients * (gram @ coefficients - 2 * rhs)).sum()
            history.append(dict(iteration=iteration, objective=float(objective)))
    result = coefficients.to(z.dtype)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Compilation produced nonfinite coefficients')
    # 返回精度中的系数决定最终诊断，保存的程序与记录保持一致。
    saved = result.double()
    objective = constant + (saved * (gram @ saved - 2 * rhs)).sum()
    diagnostics = dict(states=n, members=width, parts=fields.shape[1], iterations=steps,
                       objective_initial=float(constant), objective_final=float(objective),
                       objective_definition='Mean over states of sum over parts and hidden dimensions of squared field error',
                       largest_gram_eigenvalue=float(lipschitz),
                       maximum_negative_row_sum=float((-saved).clamp_min(0).sum(-1).max()),
                       positive_entries=int((saved > 0).sum()), negative_entries=int((saved < 0).sum()),
                       objective_history=history)
    return result, diagnostics
