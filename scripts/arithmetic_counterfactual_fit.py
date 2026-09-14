"""Fit bounded SAE masks to complete counterfactual answers.

Reuses CCAD's sparse bounded-gate projection and RAVEL's Cause/Iso objective.
Training uses labelled hybrid sequences; evaluation remains free generation.
The same learned gate acts at every generated position, including preserved digits.
"""
from __future__ import annotations

import json
import random
import time

from ccad.semantic_participation import project_sparse_gates


def fit_gates(w, cfg, model, module, tokenizer, ae, rows, tokenrows, codes,
              initial_gate, seed, members, physical_length, budget,
              tag=None, initial_scale=.5, checkpoints=None,
              selection_batches=0, integrated_steps=1, fixed_support=False,
              source_context=None):
    import numpy as np
    import torch

    fit = [i for i, row in enumerate(rows) if row['split'] == 'fit']
    rng = random.Random(cfg['counterfactual_fit']['seed'])
    training_pairs = []
    for i in fit:
        candidates = [j for j in fit if rows[i]['unit'] != rows[j]['unit']
                      and rows[i]['tens'] != rows[j]['tens']
                      and not {rows[i]['a'], rows[i]['b']} & {rows[j]['a'], rows[j]['b']}]
        training_pairs.append((i, rng.choice(candidates)))
    assert all(rows[i]['template'] == 0 for pair in training_pairs for i in pair)
    param = torch.nn.Parameter(initial_gate.detach().clone() * initial_scale)
    support = initial_gate > 0 if fixed_support else None
    saved = ({0: param.detach().clone()} if 0 in checkpoints else {}) if checkpoints is not None else None
    stem = f'counterfactual_seed{seed}_k{members}' + (f'_{tag}' if tag else '')
    optimizer = torch.optim.Adam([param], lr=cfg['counterfactual_fit']['lr'])
    batch = cfg['batch_size']
    ix = torch.arange(batch, device=w.device)
    newline = tokenizer.encode('\n', add_special_tokens=False)
    assert len(newline) == 1
    history = []
    train_start = time.perf_counter()

    def objective(pairs, operation, gate, source_alpha=0.):
        ids = torch.full((batch, physical_length), tokenizer.eos_token_id,
                         device=w.device, dtype=torch.long)
        attention = torch.zeros_like(ids)
        targets, positions = [], []
        for b, (i, j) in enumerate(pairs):
            answer = (10 * rows[i]['tens'] + rows[j]['unit'] if operation == 0
                      else 10 * rows[j]['tens'] + rows[i]['unit'])
            output_tokens = tokenizer.encode(str(answer), add_special_tokens=False)
            assert len(output_tokens) == 2, (answer, output_tokens)
            targets.append(output_tokens + newline)
            prompt = tokenrows[i]
            seq = prompt + output_tokens
            assert len(seq) <= physical_length
            ids[b, :len(seq)] = torch.tensor(seq, device=w.device)
            attention[b, :len(seq)] = 1
            positions.append(len(prompt) - 1)
        sites = torch.tensor(positions, device=w.device)[:, None] + torch.arange(3, device=w.device)
        donor = codes[[j for i, j in pairs], :3]

        def hook(_module, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            current = h[ix[:, None], sites]
            z = ae.encode(current.reshape(-1, current.shape[-1])).reshape(batch, 3, -1)
            delta = ((donor - z) * gate) @ ae.decoder.weight.T
            if source_alpha:
                source_ae, source_codes, source_gate = source_context
                source_z = source_ae.encode(current.reshape(-1,current.shape[-1])).reshape(batch,3,-1)
                source_donor = source_codes[[j for i,j in pairs], :3]
                delta = delta + source_alpha * ((source_donor-source_z)*source_gate[:,operation]) @ source_ae.decoder.weight.T
            edited = h.clone()
            edited[ix[:, None], sites] += delta
            return (edited,) + output[1:] if isinstance(output, tuple) else edited

        handle = module.register_forward_hook(hook)
        try:
            result = model(ids, attention_mask=attention, use_cache=False)
            logits = result.logits[ix[:, None], sites]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                torch.tensor(targets, device=w.device).flatten())
            w.sequence_forwards += batch
            w.token_forwards += batch * physical_length
            return loss
        finally:
            handle.remove()

    # One finite difference at an interior gate verifies the actual model path.
    check_pairs = training_pairs[:batch]
    check_alpha = .5 if source_context is not None else 0.
    loss = objective(check_pairs, 0, param[:, 0], source_alpha=check_alpha)
    grad = torch.autograd.grad(loss, param)[0]
    interior = torch.where(initial_gate[:, 0] > 0)[0]
    coord = int(interior[grad[interior, 0].abs().argmax()])
    analytic = float(grad[coord, 0])
    eps = .01
    with torch.no_grad():
        plus, minus = param[:, 0].clone(), param[:, 0].clone()
        plus[coord] += eps
        minus[coord] -= eps
        finite = float((objective(check_pairs, 0, plus, source_alpha=check_alpha)
                        - objective(check_pairs, 0, minus, source_alpha=check_alpha)) / (2 * eps))
    error = abs(finite - analytic)
    assert error < .015 + .05 * abs(analytic), (finite, analytic, error)
    w.checks[f'counterfactual_gradient_seed{seed}_k{members}' + (f'_{tag}' if tag else '')] = True
    del grad, loss

    steps = cfg['counterfactual_fit']['steps']
    assert 0 <= selection_batches < steps
    assert selection_batches % (2 * integrated_steps) == 0
    schedule = [rng.sample(training_pairs, batch) for _ in range(steps)]
    selection_scores = torch.zeros_like(param)
    selection_trace = []
    for step in range(steps):
        operation = step % 2
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
            probe = torch.full_like(param[:, operation], 0. if source_context is not None else alpha,
                                    requires_grad=True)
            selection_loss = objective(pairs, operation, probe,
                                       source_alpha=alpha if source_context is not None else 0.)
            derivative = torch.autograd.grad(selection_loss, probe)[0]
            assert torch.isfinite(derivative).all()
            selection_scores[:, operation] -= derivative.detach()
            selection_trace.append(dict(step=step + 1, operation=operation,
                                        alpha=alpha, loss=float(selection_loss.detach()),
                                        pairs=[list(p) for p in pairs]))
            if step + 1 == selection_batches:
                with torch.no_grad():
                    param.zero_()
                    for c in range(2):
                        chosen = torch.argsort(selection_scores[:, c], descending=True, stable=True)[:members]
                        chosen = chosen[selection_scores[chosen, c] > 0]
                        param[chosen, c] = .5
                if saved is not None and step + 1 in checkpoints:
                    saved[step + 1] = param.detach().clone()
                w.progress('TARGET_GRADIENT_SELECTION', seed=seed, initialization=tag,
                           backward_batches=selection_batches, integrated_steps=integrated_steps,
                           active=(param.detach() > 0).sum(0).cpu().tolist())
            budget()
            continue
        optimizer.zero_grad(set_to_none=True)
        loss = objective(pairs, operation, param[:, operation])
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        for c in range(2):
            # This projects a possibly noncontiguous column; the helper preserves
            # storage by copying its result back, with a separate per-request budget.
            column = param[:, c].detach().clone()
            if support is not None:
                column *= support[:, c]
            project_sparse_gates(column, members)
            with torch.no_grad():
                param[:, c].copy_(column)
        if saved is not None and step + 1 in checkpoints:
            saved[step + 1] = param.detach().clone()
        if step == 0 or (step + 1) % 32 == 0 or step + 1 == steps:
            item = dict(step=step + 1, operation=operation, loss=float(loss.detach()),
                        fit_elapsed_seconds=time.perf_counter() - train_start,
                        active=(param.detach() > 0).sum(0).cpu().tolist())
            history.append(item)
            w.progress('COUNTERFACTUAL_SOURCE_FIT', seed=seed, members=members, initialization=tag, **item)
        budget()
    fitted = param.detach()
    assert bool(((fitted >= 0) & (fitted <= 1)).all())
    assert bool(((fitted > 0).sum(0) <= members).all())
    if support is not None:
        assert not bool((fitted[~support] > 0).any())
    np.savez_compressed(w.run / f'{stem}.npz',
                        gates=fitted.cpu().numpy(), fit_pairs=np.array(training_pairs),
                        fit_schedule=np.array(schedule), selection_scores=selection_scores.cpu().numpy(),
                        **({f'updates_{n}': g.cpu().numpy() for n, g in saved.items()} if saved is not None else {}))
    (w.run / f'{stem}.json').write_text(json.dumps(dict(
        seed=seed, members=members, gradient=dict(analytic=analytic, finite=finite, error=error,source_alpha=check_alpha),
        trace=history, training_pairs=len(training_pairs), updates=steps, initialization=tag,
        initial_scale=initial_scale, saved_updates=sorted(saved) if saved is not None else [],
        backward_budget=dict(selection=selection_batches, optimization=steps-selection_batches,
                             total=steps, numerical_check=1),
        forward_budget=dict(selection=selection_batches, optimization=steps-selection_batches,
                            numerical_check=3),
        integrated_steps=integrated_steps, fixed_support=fixed_support,
        source_functional_path=source_context is not None,
        selection_trace=selection_trace,
        supervision='Hybrid full-answer CE: donor target digit, recipient preserved digit, newline. Fit split only.',
        evaluation='A single bounded gate is applied at every generated position; no answer prefix at evaluation.',
        final_active=(fitted > 0).sum(0).cpu().tolist()), indent=2) + '\n')
    return (fitted, saved) if checkpoints is not None else fitted
