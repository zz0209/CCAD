import torch

from ccad.intervention_transport import project_capacity


def request_second_moment(groups, member_weight=0.1):
    if groups.ndim != 2 or not 0 <= member_weight <= 1:
        raise ValueError('Expected member-by-group indicators and a mixture weight in [0, 1]')
    if not torch.all((groups == 0) | (groups == 1)) or not torch.all(groups.sum(-1) == 1):
        raise ValueError('Every member must belong to exactly one group')
    members, count = groups.shape
    group_moment = (torch.ones((count, count), device=groups.device, dtype=groups.dtype) * 0.25
                    + torch.eye(count, device=groups.device, dtype=groups.dtype) / 12)
    member_moment = (torch.ones((members, members), device=groups.device, dtype=groups.dtype) * 0.25
                     + torch.eye(members, device=groups.device, dtype=groups.dtype) / 12)
    return (1 - member_weight) * (groups @ group_moment @ groups.T) + member_weight * member_moment


def solve_request_columns(gram, rhs, capacity, initial, request_moment, steps):
    if gram.shape[-2:] != (initial.shape[-2], initial.shape[-2]):
        raise ValueError('Gram and member dimensions disagree')
    if rhs.shape != initial.shape or capacity.shape != initial.shape[:-1]:
        raise ValueError('Coefficient, linear term and capacity dimensions disagree')
    if request_moment.shape != (initial.shape[-1], initial.shape[-1]):
        raise ValueError('Request dimension differs from source columns')
    if steps < 1 or not torch.allclose(request_moment, request_moment.T):
        raise ValueError('Positive steps and a symmetric request second moment are required')
    eigenvalues = torch.linalg.eigvalsh(request_moment.double()).to(request_moment.dtype)
    tolerance = torch.finfo(request_moment.dtype).eps * request_moment.shape[0] * eigenvalues[-1].abs()
    if eigenvalues[0] < -tolerance or eigenvalues[-1] <= 0:
        raise ValueError('Request second moment must be nonzero positive semidefinite')
    if (capacity < 0).any():
        raise ValueError('Negative native capacity')
    lipschitz = (gram.abs().sum(-1).amax(-1) * eigenvalues[-1]).clamp_min(1e-8)
    if ((-initial).clamp_min(0).sum(-1) > capacity + 2e-5).any():
        raise ValueError('Initial coefficients exceed native capacity')
    coefficients = initial
    current = coefficients
    momentum = 1.0
    for _ in range(steps):
        gradient = (gram @ current - rhs) @ request_moment
        updated = project_capacity(current - gradient / lipschitz[..., None, None], capacity)
        next_momentum = (1 + (1 + 4 * momentum ** 2) ** 0.5) / 2
        current = updated + (momentum - 1) / next_momentum * (updated - coefficients)
        coefficients = updated
        momentum = next_momentum
    return coefficients
