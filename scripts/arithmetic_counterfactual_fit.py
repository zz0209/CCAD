"""Fit bounded SAE masks to complete counterfactual answers.

Reuses CCAD's sparse bounded-gate projection and RAVEL's Cause/Iso objective.
Training uses labelled hybrid sequences; evaluation remains free generation.
Gates can share weights across positions or condition them on answer roles.
"""
from __future__ import annotations

import json
import random
import time

from ccad.semantic_participation import project_sparse_gates


def project_role_members(value, members):
    """Project role-by-feature weights onto [0,1] with a feature-union budget.

    Retaining feature f reduces squared projection error by the sum, over roles,
    of 2*v*clip(v)-clip(v)^2. Selecting the largest gains is the exact projection.
    """
    import torch
    with torch.no_grad():
        feasible = value.clamp(0, 1)
        gain = (2 * value * feasible - feasible.square()).sum(0)
        keep = torch.argsort(gain, descending=True, stable=True)[:members]
        result = torch.zeros_like(value)
        result[:, keep] = feasible[:, keep]
        value.copy_(result)


def member_counts(gate):
    active = gate > 0
    return (active.any(0) if gate.ndim == 3 else active).sum(0)


def natural_latent_reference(w, rows, hidden, base, rank=16):
    """Fit answer-contrast coordinates from correct natural fit trajectories.

    Conditional means average over operand identity. The projection is a
    descriptive answer subspace, not an independently identified causal space.
    Its sole purpose is an auxiliary training objective; free generation tests
    whether the learned masks acquire more reusable functions.
    """
    import numpy as np
    import torch

    indices = [i for i, r in enumerate(rows)
               if r['split'] == 'fit' and base[i] == r['total']]
    assert indices and all(rows[i]['split'] == 'fit' for i in indices)
    means = torch.zeros((100, 3, hidden.shape[-1]), device=hidden.device)
    counts = torch.zeros(100, dtype=torch.long, device=hidden.device)
    for answer in range(100):
        group = [i for i in indices if rows[i]['total'] == answer]
        counts[answer] = len(group)
        if group:
            means[answer] = hidden[group, :3].mean(0)
    present = counts > 0
    assert int(present.sum()) > rank
    bases, scales = [], []
    for site in range(3):
        centered = means[present, site] - means[present, site].mean(0)
        _, _, vh = torch.linalg.svd(centered.double(), full_matrices=False)
        basis = vh[:rank].T.float()
        bases.append(basis)
        scales.append(((centered @ basis).square().sum(1)).mean().clamp_min(1e-8))
    basis, scale = torch.stack(bases), torch.stack(scales)
    projected = torch.einsum('asd,sdr->asr', means, basis)
    # Check geometry independently of the model's TF32 matmul setting.
    orthogonality_error = float((basis.double().transpose(1, 2) @ basis.double()
                                - torch.eye(rank, device=hidden.device, dtype=torch.float64)).abs().max())
    assert orthogonality_error < 1e-5, orthogonality_error
    np.savez_compressed(w.run / 'natural_latent_reference.npz',
                        basis=basis.cpu().numpy(), scale=scale.cpu().numpy(),
                        projected=projected.cpu().numpy(), counts=counts.cpu().numpy(),
                        fit_indices=np.array(indices))
    w.checks['natural_latent_reference_fit_only'] = True
    w.progress('NATURAL_LATENT_REFERENCE', fit_rows=len(indices),
               answer_classes=int(present.sum()), rank=rank,
               orthogonality_error=orthogonality_error,
               normalization=scale.cpu().tolist())
    return dict(basis=basis, scale=scale, projected=projected, present=present)


def fit_gates(w, cfg, model, module, tokenizer, ae, rows, tokenrows, codes,
              initial_gate, seed, members, physical_length, budget,
              tag=None, initial_scale=.5, checkpoints=None,
              selection_batches=0, integrated_steps=1, fixed_support=False,
              source_context=None, latent_context=None, latent_weight=0., view_map=None):
    import numpy as np
    import torch

    fit = [i for i, row in enumerate(rows) if row['split'] == 'fit' and not row.get('source_view')]
    rng = random.Random(cfg['counterfactual_fit']['seed'])
    carry_rule = cfg['counterfactual_fit'].get('carry_rule', False)
    carry_pairs = [[], []]
    training_pairs = []
    for i in fit:
        if carry_rule:
            for condition in range(2):
                candidates = [j for j in fit
                    if not {rows[i]['a'], rows[i]['b']} & {rows[j]['a'], rows[j]['b']}
                    and ((rows[i]['total'] == rows[j]['total'] and rows[i]['carry'] != rows[j]['carry'])
                         if condition == 0 else
                         (rows[i]['carry'] == rows[j]['carry'] and rows[i]['unit'] == rows[j]['unit']
                          and rows[i]['total'] != rows[j]['total']))]
                if candidates:
                    carry_pairs[condition].append((i, rng.choice(candidates)))
            continue
        candidates = [j for j in fit if rows[i]['unit'] != rows[j]['unit']
                      and rows[i]['tens'] != rows[j]['tens']
                      and not {rows[i]['a'], rows[i]['b']} & {rows[j]['a'], rows[j]['b']}]
        training_pairs.append((i, rng.choice(candidates)))
    if carry_rule:
        assert all(len(group) >= cfg['batch_size'] for group in carry_pairs)
        training_pairs = carry_pairs[0] + carry_pairs[1]
    assert all(rows[i]['template'] == 0 for pair in training_pairs for i in pair)
    role_schema = cfg['counterfactual_fit'].get('role_schema', False)
    if carry_rule:
        assert not role_schema and source_context is None and latent_context is None and view_map is None
        assert integrated_steps == 1
    start = initial_gate.detach().clone()
    if role_schema and start.ndim == 2:
        start = start[None].repeat(cfg['max_new_tokens'] + 1, 1, 1)
        # No training target follows the newline; those later roles remain zero.
        start[4:] = 0
    assert start.ndim == (3 if role_schema else 2)
    param = torch.nn.Parameter(start * initial_scale)
    support = start > 0 if fixed_support else None
    saved = ({0: param.detach().clone()} if 0 in checkpoints else {}) if checkpoints is not None else None
    stem = f'counterfactual_seed{seed}_k{members}' + (f'_{tag}' if tag else '')
    optimizer = torch.optim.Adam([param], lr=cfg['counterfactual_fit']['lr'])
    batch = cfg['batch_size']
    ix = torch.arange(batch, device=w.device)
    newline = tokenizer.encode('\n', add_special_tokens=False)
    assert len(newline) == 1
    history = []
    objective_components = {}
    train_start = time.perf_counter()

    def objective(pairs, operation, gate, source_alpha=0.):
        ids = torch.full((batch, physical_length), tokenizer.eos_token_id,
                         device=w.device, dtype=torch.long)
        attention = torch.zeros_like(ids)
        targets, positions, answers = [], [], []
        for b, (i, j) in enumerate(pairs):
            answer = (10 * rows[i]['tens'] + rows[j]['unit'] if operation == 0
                      else 10 * rows[j]['tens'] + rows[i]['unit'])
            if carry_rule:
                assert operation == 0
                answer = rows[i]['total'] + 10 * (int(rows[j]['carry']) - int(rows[i]['carry']))
                assert 10 <= answer < 100
            prefix = cfg['counterfactual_fit'].get('answer_prefixes', {}).get(str(rows[i]['template']), '')
            assert rows[i]['template'] == rows[j]['template']
            output_tokens = tokenizer.encode(prefix + str(answer), add_special_tokens=False)
            assert len(output_tokens) in [2, 3], (answer, output_tokens)
            targets.append(output_tokens + newline)
            answers.append(answer)
            prompt = tokenrows[i]
            seq = prompt + output_tokens
            assert len(seq) <= physical_length
            ids[b, :len(seq)] = torch.tensor(seq, device=w.device)
            attention[b, :len(seq)] = 1
            positions.append(len(prompt) - 1)
        nsites = max(map(len, targets))
        target_tensor = torch.full((batch, nsites), -100, device=w.device, dtype=torch.long)
        for b, target in enumerate(targets):
            target_tensor[b, :len(target)] = torch.tensor(target, device=w.device)
        active = target_tensor != -100
        sites = torch.tensor(positions, device=w.device)[:, None] + torch.arange(nsites, device=w.device)
        donor = codes[[j for i, j in pairs], :nsites]
        latent = {}

        def hook(_module, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            current = h[ix[:, None], sites]
            z = ae.encode(current.reshape(-1, current.shape[-1])).reshape(batch, nsites, -1)
            applied_gate = gate
            if role_schema:
                shift = torch.tensor([1 - rows[i]['template'] for i, j in pairs], device=w.device)
                roles = torch.arange(nsites, device=w.device)[None] + shift[:, None]
                applied_gate = gate[roles]
            delta = ((donor - z) * applied_gate) @ ae.decoder.weight.T
            if source_alpha:
                source_ae, source_codes, source_gate = source_context
                source_z = source_ae.encode(current.reshape(-1,current.shape[-1])).reshape(batch,nsites,-1)
                source_donor = source_codes[[j for i,j in pairs], :nsites]
                delta = delta + source_alpha * ((source_donor-source_z)*source_gate[:,operation]) @ source_ae.decoder.weight.T
            delta = delta * active[:, :, None]
            if carry_rule:
                delta = delta * (torch.arange(nsites, device=w.device) == 0)[None, :, None]
            edited = h.clone()
            edited[ix[:, None], sites] += delta
            if latent_context is not None:
                assert nsites == 3
                latent['projected'] = torch.einsum(
                    'bsd,sdr->bsr', current + delta, latent_context['basis'])
            return (edited,) + output[1:] if isinstance(output, tuple) else edited

        handle = module.register_forward_hook(hook)
        try:
            result = model(ids, attention_mask=attention, use_cache=False)
            logits = result.logits[ix[:, None], sites]
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_tensor.flatten())
            loss = ce
            objective_components.clear()
            objective_components['ce'] = float(ce.detach())
            if latent_context is not None:
                answer_ix = torch.tensor(answers, device=w.device)
                valid = latent_context['present'][answer_ix]
                # Unsupported hybrid totals retain CE and contribute zero CL.
                distance = (latent['projected'] - latent_context['projected'][answer_ix]).square().sum(-1)
                cl = ((distance / latent_context['scale']) * valid[:, None]).sum() / (3 * valid.sum().clamp_min(1))
                loss = loss + latent_weight * cl
                objective_components.update(cl=float(cl.detach()), natural_coverage=float(valid.float().mean()))
            w.sequence_forwards += batch
            w.token_forwards += batch * physical_length
            return loss
        finally:
            handle.remove()

    # One finite difference at an interior gate verifies the actual model path.
    check_pairs = training_pairs[:batch]
    check_alpha = .5 if source_context is not None else 0.
    loss = objective(check_pairs, 0, param[..., 0], source_alpha=check_alpha)
    grad = torch.autograd.grad(loss, param)[0]
    interior = torch.where(start[..., 0].flatten() > 0)[0]
    coord = int(interior[grad[..., 0].flatten()[interior].abs().argmax()])
    analytic = float(grad[..., 0].flatten()[coord])
    eps = .01
    with torch.no_grad():
        plus, minus = param[..., 0].clone().contiguous(), param[..., 0].clone().contiguous()
        plus.view(-1)[coord] += eps
        minus.view(-1)[coord] -= eps
        finite = float((objective(check_pairs, 0, plus, source_alpha=check_alpha)
                        - objective(check_pairs, 0, minus, source_alpha=check_alpha)) / (2 * eps))
    error = abs(finite - analytic)
    assert error < .015 + .05 * abs(analytic), (finite, analytic, error)
    w.checks[f'counterfactual_gradient_seed{seed}_k{members}' + (f'_{tag}' if tag else '')] = True
    del grad, loss
    view_gradient = None
    if view_map is not None and cfg['counterfactual_fit'].get('answer_prefixes'):
        view_pairs = [(view_map[i], view_map[j]) for i, j in training_pairs
                      if view_map[i] != i and view_map[j] != j][:batch]
        assert len(view_pairs) == batch
        loss = objective(view_pairs, 0, param[..., 0])
        derivative = torch.autograd.grad(loss, param)[0]
        coord_v = int(interior[derivative[..., 0].flatten()[interior].abs().argmax()])
        analytic_v = float(derivative[..., 0].flatten()[coord_v])
        with torch.no_grad():
            plus, minus = param[..., 0].clone().contiguous(), param[..., 0].clone().contiguous()
            plus.view(-1)[coord_v] += eps
            minus.view(-1)[coord_v] -= eps
            finite_v = float((objective(view_pairs, 0, plus) - objective(view_pairs, 0, minus)) / (2 * eps))
        error_v = abs(finite_v - analytic_v)
        assert error_v < .015 + .05 * abs(analytic_v), (finite_v, analytic_v, error_v)
        view_gradient = dict(analytic=analytic_v, finite=finite_v, error=error_v)
        w.checks[f'formatted_view_gradient_seed{seed}_{tag}'] = True
        del derivative, loss

    steps = cfg['counterfactual_fit']['steps']
    assert 0 <= selection_batches < steps
    assert selection_batches % (2 * integrated_steps) == 0
    schedule = [rng.sample(training_pairs, batch) for _ in range(steps)]
    if carry_rule:
        schedule = [rng.sample(carry_pairs[step % 2], batch) for step in range(steps)]
    original_schedule = [list(pairs) for pairs in schedule]
    if view_map is not None:
        # Each request sees equal numbers of original and answer-equivalent
        # operand forms. Pair labels and total backward budget are unchanged.
        for step in range(steps):
            if (step // 2) % 2:
                schedule[step] = [(view_map[i], view_map[j]) if view_map[i] != i and view_map[j] != j
                                  else (i, j) for i, j in schedule[step]]
        assert all(rows[i]['total'] == rows[v]['total'] for i, v in view_map.items())
    selection_scores = torch.zeros_like(param)
    selection_trace = []
    for step in range(steps):
        operation = 0 if carry_rule else step % 2
        pairs = schedule[step]
        if step < selection_batches:
            # Gate derivatives include (donor code - current code) times the
            # downstream gradient. Four-point IG repeats each batch at all
            # midpoints; every backward call is charged to the same total budget.
            if integrated_steps > 1:
                pairs = schedule[2 * (step // (2 * integrated_steps)) + operation]
                alpha = ((step // 2) % integrated_steps + .5) / integrated_steps
            else:
                alpha = 0.
            # A source-functional path changes the source component; the target
            # gate is a zero-valued probe whose derivative scores all members.
            probe = torch.full_like(param[..., operation], 0. if source_context is not None else alpha,
                                    requires_grad=True)
            selection_loss = objective(pairs, operation, probe,
                                       source_alpha=alpha if source_context is not None else 0.)
            derivative = torch.autograd.grad(selection_loss, probe)[0]
            assert torch.isfinite(derivative).all()
            selection_scores[..., operation] -= derivative.detach()
            selection_trace.append(dict(step=step + 1, operation=operation,
                                        alpha=alpha, loss=float(selection_loss.detach()),
                                        pairs=[list(p) for p in pairs]))
            if step + 1 == selection_batches:
                with torch.no_grad():
                    param.zero_()
                    for c in range(2):
                        score = selection_scores[..., c]
                        ranking = score.clamp_min(0).square().sum(0) if role_schema else score
                        chosen = torch.argsort(ranking, descending=True, stable=True)[:members]
                        chosen = chosen[ranking[chosen] > 0]
                        if role_schema:
                            param[:, chosen, c] = .5 * (score[:, chosen] > 0)
                        else:
                            param[chosen, c] = .5
                if saved is not None and step + 1 in checkpoints:
                    saved[step + 1] = param.detach().clone()
                w.progress('TARGET_GRADIENT_SELECTION', seed=seed, initialization=tag,
                           backward_batches=selection_batches, integrated_steps=integrated_steps,
                           active=member_counts(param.detach()).cpu().tolist())
            budget()
            continue
        optimizer.zero_grad(set_to_none=True)
        loss = objective(pairs, operation, param[..., operation])
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        for c in range(2):
            # This projects a possibly noncontiguous column; the helper preserves
            # storage by copying its result back, with a separate per-request budget.
            column = param[..., c].detach().clone()
            if support is not None:
                column *= support[..., c]
            if role_schema:
                project_role_members(column, members)
            else:
                project_sparse_gates(column, members)
            with torch.no_grad():
                param[..., c].copy_(column)
        if saved is not None and step + 1 in checkpoints:
            saved[step + 1] = param.detach().clone()
        if step == 0 or (step + 1) % 32 == 0 or step + 1 == steps:
            item = dict(step=step + 1, operation=operation, loss=float(loss.detach()),
                        objective_components=dict(objective_components),
                        fit_elapsed_seconds=time.perf_counter() - train_start,
                        active=member_counts(param.detach()).cpu().tolist())
            history.append(item)
            w.progress('COUNTERFACTUAL_SOURCE_FIT', seed=seed, members=members, initialization=tag, **item)
        budget()
    fitted = param.detach()
    assert bool(((fitted >= 0) & (fitted <= 1)).all())
    assert bool((member_counts(fitted) <= members).all())
    if support is not None:
        assert not bool((fitted[~support] > 0).any())
    np.savez_compressed(w.run / f'{stem}.npz',
                        gates=fitted.cpu().numpy(), fit_pairs=np.array(training_pairs),
                        fit_schedule=np.array(schedule), selection_scores=selection_scores.cpu().numpy(),
                        **(dict(original_schedule=np.array(original_schedule),
                                view_pairs=np.array([(view_map[i], view_map[j]) for i, j in training_pairs])) if view_map is not None else {}),
                        **({f'updates_{n}': g.cpu().numpy() for n, g in saved.items()} if saved is not None else {}))
    (w.run / f'{stem}.json').write_text(json.dumps(dict(
        seed=seed, members=members, gradient=dict(analytic=analytic, finite=finite, error=error,source_alpha=check_alpha),
        formatted_view_gradient=view_gradient,
        trace=history, training_pairs=len(training_pairs), updates=steps, initialization=tag,
        initial_scale=initial_scale, saved_updates=sorted(saved) if saved is not None else [],
        backward_budget=dict(selection=selection_batches, optimization=steps-selection_batches,
                             total=steps, numerical_check=1 + int(view_gradient is not None)),
        forward_budget=dict(selection=selection_batches, optimization=steps-selection_batches,
                            numerical_check=3 * (1 + int(view_gradient is not None))),
        integrated_steps=integrated_steps, fixed_support=fixed_support,
        source_functional_path=source_context is not None,
        latent_weight=latent_weight, natural_latent_objective=latent_context is not None,
        answer_equivalent_views=view_map is not None,
        carry_rule=carry_rule,
        carry_pair_counts=[len(group) for group in carry_pairs] if carry_rule else None,
        role_schema=role_schema, role_order=['leading_space','tens','units','following_token','later_1','later_2'] if role_schema else None,
        selection_trace=selection_trace,
        supervision=('Full-answer CE for recipient total +10*(donor carry - recipient carry); balanced change/preserve batches, one shared gate, original fit split only.' if carry_rule else 'Hybrid full-answer CE: donor target digit, recipient preserved digit, newline. Fit split only.'),
        evaluation=('One shared bounded gate at the tens role only; no answer prefix at evaluation.' if carry_rule else ('Bounded gates at each generated prefix; answer-role weights with a shared member union.' if role_schema else 'A single bounded gate is applied at every generated position; no answer prefix at evaluation.')),
        final_active=member_counts(fitted).cpu().tolist()), indent=2) + '\n')
    return (fitted, saved) if checkpoints is not None else fitted
