"""Fit a reusable, role-conditioned source-member to target-code relation.

Only the fixed source-fit pairs and source interventions supply supervision.
The source query is applied before the learned map. Evaluation uses frozen
matrices and one model continuation, with no output optimization at query time.
"""
import json
import time
from pathlib import Path


def fit_writers(w, cfg, model, module, tok, saes):
    import numpy as np
    import torch
    from run_causalgym_multisite import ROOT, write

    spec = cfg['request_writer_fit']
    parent = ROOT / cfg['member_queries']['relation_run']
    cache = ROOT / cfg['member_queries']['source_cache_identity']
    reader = ROOT / cfg['query_readouts']['fit_run']
    panel = json.loads(w.checked(parent / 'panel.json').read_text())
    rows = panel['rows']
    with np.load(w.checked(cache / 'states.npz')) as z:
        hidden = z['hidden']
    with np.load(w.checked(parent / 'source_view_states.npz')) as z:
        hidden = np.concatenate([hidden, z['hidden']])
    assert len(hidden) == len(rows)
    hidden = torch.tensor(hidden, device=w.device)
    length = cfg['max_new_tokens']
    metadata = []
    started = time.perf_counter()
    for s, t in spec['seed_pairs']:
        with np.load(w.checked(reader / f'readout_s{s}_t{t}.npz')) as z:
            payload = {k: z[k] for k in z.files}
        pairs = payload['fit_pairs']
        assert all(rows[int(i)]['split'] == 'fit' for i in pairs.ravel())
        gates = torch.tensor(payload['source_gate'], device=w.device)
        codes = {}
        with torch.no_grad():
            for seed in [s, t]:
                codes[seed] = saes[seed].encode(hidden.reshape(-1, hidden.shape[-1])).reshape(*hidden.shape[:-1], -1)
        tokenrows = [tok.encode(rows[int(i)]['prompt'], add_special_tokens=False) for i in pairs[:, 0]]
        physical = max(map(len, tokenrows)) + length
        assert physical <= cfg['max_length']
        output = dict(fit_pairs=pairs)
        for c, op in enumerate(['unit', 'tens']):
            si = torch.tensor(payload[f'{op}_source_indices'], device=w.device)
            ti = torch.tensor(payload[f'{op}_target_indices'], device=w.device)
            ds = saes[s].decoder.weight.T[si].detach()
            dt = saes[t].decoder.weight.T[ti].detach()
            assert torch.allclose(ds, torch.tensor(payload[f'{op}_decoder'], device=w.device))
            read_native = torch.tensor(payload[f'{op}_full_activation'][:, ti.cpu().numpy()], device=w.device)
            # The selected input bank is an exact restriction, not a new fit.
            other = np.delete(payload[f'{op}_full_activation'], ti.cpu().numpy(), axis=1)
            assert np.count_nonzero(other) == 0
            read_raw = torch.tensor(payload[f'{op}_raw'], device=w.device)
            gram = dt @ dt.T
            init = torch.linalg.solve(gram + spec['decoder_ridge'] * torch.eye(len(ti), device=w.device), dt @ ds.T).T
            matrices = {'native': torch.nn.Parameter(init[None].repeat(len(gates), 1, 1)),
                        'raw': torch.nn.Parameter(torch.eye(len(si), device=w.device)[None].repeat(len(gates), 1, 1))}
            opts = {k: torch.optim.Adam([v], lr=spec['lr']) for k, v in matrices.items()}
            rng = np.random.default_rng(spec['seed'] + s * 10 + c)
            queries = [np.ones(len(si), dtype=np.float32)]
            for _ in range(spec['random_partitions']):
                q = np.zeros(len(si), dtype=np.float32)
                q[rng.permutation(len(si))[:len(si)//2]] = 1
                queries.extend([q, 1-q])
            queries = torch.tensor(np.stack(queries), device=w.device)
            output[f'{op}_training_queries'] = queries.cpu().numpy()
            output[f'{op}_source_indices'] = si.cpu().numpy()
            output[f'{op}_target_indices'] = ti.cpu().numpy()
            output[f'{op}_target_decoder'] = dt.cpu().numpy()
            trajectories = {}
            calls = dict(teacher_forward=0, native_forward=0, raw_forward=0,
                         native_backward=0, raw_backward=0)

            def make_batch(selection, qid, generated=True):
                rec = pairs[selection, 0].tolist()
                donors = pairs[selection, 1].tolist()
                ids = torch.full((len(selection), physical), tok.eos_token_id, dtype=torch.long, device=w.device)
                attention = torch.zeros_like(ids)
                pos = []
                for b, p in enumerate(selection):
                    tokens = tokenrows[p]
                    ids[b, :len(tokens)] = torch.tensor(tokens, device=w.device)
                    attention[b, :len(tokens)] = 1
                    pos.append(len(tokens)-1)
                    if generated:
                        ids[b, len(tokens):len(tokens)+length] = trajectories[qid, int(p)]
                        attention[b, len(tokens):len(tokens)+length] = 1
                pos = torch.tensor(pos, device=w.device)
                roles = torch.arange(length, device=w.device)[None] + torch.tensor([1-rows[i]['template'] for i in rec], device=w.device)[:, None]
                return ids, attention, pos, roles, donors

            def forward(kind, ids, attention, pos, roles, donors, qid, steps):
                sites = pos[:, None] + torch.arange(steps, device=w.device)[None]
                ix = torch.arange(len(pos), device=w.device)[:, None]
                rr = roles[:, :steps]
                q = queries[qid]
                def hook(_m, _a, out):
                    h = out[0] if isinstance(out, tuple) else out
                    current = h[ix, sites]
                    if kind == 'teacher':
                        code = saes[s].encode(current.reshape(-1, current.shape[-1])).reshape(*current.shape[:-1], -1)
                        amp = (codes[s][donors, :steps, :][..., si] - code[..., si]) * gates[rr][:, :, si, c]
                        field = (amp * q) @ ds
                    elif kind == 'native':
                        code = saes[t].encode(current.reshape(-1, current.shape[-1])).reshape(*current.shape[:-1], -1)[..., ti]
                        difference = codes[t][donors, :steps, :][..., ti] - code
                        amp = torch.einsum('blj,blji->bli', difference, read_native[rr])
                        coeff = torch.einsum('bli,blij->blj', amp * q, matrices[kind][rr])
                        field = torch.maximum(coeff, -code) @ dt
                    else:
                        amp = torch.einsum('blj,blji->bli', hidden[donors, :steps] - current, read_raw[rr])
                        field = torch.einsum('bli,blij->blj', amp * q, matrices[kind][rr]) @ ds
                    edited = h.clone()
                    edited[ix, sites] += field
                    return (edited,) + out[1:] if isinstance(out, tuple) else edited
                handle = module.register_forward_hook(hook)
                try:
                    result = model(ids, attention_mask=attention, use_cache=False).logits[ix, sites]
                finally:
                    handle.remove()
                calls[kind + '_forward'] += 1
                w.sequence_forwards += len(pos)
                w.token_forwards += ids.numel()
                return result

            # Cache only the source-generated tokens; target logits remain unseen.
            with torch.no_grad():
                for qid in range(len(queries)):
                    for off in range(0, len(pairs), cfg['batch_size']):
                        sel = list(range(off, min(off+cfg['batch_size'], len(pairs))))
                        ids, attn, pos, roles, donors = make_batch(sel, qid, False)
                        finished = torch.zeros(len(sel), dtype=torch.bool, device=w.device)
                        generated = []
                        for step in range(length):
                            logits = forward('teacher', ids, attn, pos, roles, donors, qid, step+1)[:, -1]
                            nxt = logits.argmax(-1)
                            nxt = torch.where(finished, tok.eos_token_id, nxt)
                            finished |= nxt == tok.eos_token_id
                            ids[torch.arange(len(sel), device=w.device), pos+step+1] = nxt
                            attn[torch.arange(len(sel), device=w.device), pos+step+1] = 1
                            generated.append(nxt)
                        tokens = torch.stack(generated, dim=1)
                        for b, p in enumerate(sel):
                            trajectories[qid, p] = tokens[b].clone()
                    w.progress('REQUEST_TEACHER', operation=op, query=qid, total_queries=len(queries), **calls)
            output[f'{op}_teacher_tokens'] = torch.stack([trajectories[q, p] for q in range(len(queries)) for p in range(len(pairs))]).cpu().numpy()
            for kind, matrix in matrices.items():
                output[f'{op}_{kind}_0'] = matrix.detach().cpu().numpy().copy()
            for step in range(spec['updates']):
                qid = 0 if step % 2 == 0 else 1 + int(rng.integers(len(queries)-1))
                sel = rng.choice(len(pairs), cfg['batch_size'], replace=False).tolist()
                ids, attn, pos, roles, donors = make_batch(sel, qid)
                with torch.no_grad():
                    teacher_log = forward('teacher', ids, attn, pos, roles, donors, qid, length).log_softmax(-1)
                    teacher_prob = teacher_log.exp()
                losses = {}
                for kind, matrix in matrices.items():
                    opts[kind].zero_grad(set_to_none=True)
                    student_log = forward(kind, ids, attn, pos, roles, donors, qid, length).log_softmax(-1)
                    loss = (teacher_prob * (teacher_log-student_log)).sum(-1).mean()
                    assert torch.isfinite(loss)
                    loss.backward()
                    calls[kind + '_backward'] += 1
                    if step == 0:
                        coordinate = np.unravel_index(int(matrix.grad.abs().argmax()), matrix.shape)
                        analytic = float(matrix.grad[coordinate])
                        original = float(matrix[coordinate].detach())
                        finite = []
                        with torch.no_grad():
                            for sign in [-1, 1]:
                                matrix[coordinate] = original + sign * spec['gradient_epsilon']
                                ll = forward(kind, ids, attn, pos, roles, donors, qid, length).log_softmax(-1)
                                finite.append(float((teacher_prob*(teacher_log-ll)).sum(-1).mean()))
                            matrix[coordinate] = original
                        fd = (finite[1]-finite[0])/(2*spec['gradient_epsilon'])
                        assert abs(fd-analytic) < .015+.08*abs(analytic), (kind, fd, analytic)
                        w.record(kind='gradient_check', task='arithmetic', row_id=c, component=op,
                                 method=kind, seed=s, analytic=analytic, finite_difference=fd)
                    torch.nn.utils.clip_grad_norm_([matrix], spec['gradient_clip'])
                    opts[kind].step()
                    losses[kind] = float(loss.detach())
                    del student_log, loss
                w.record(kind='request_fit', task='arithmetic', row_id=step, component=op,
                         method='matched_native_raw', seed=s, operation=op, query=qid, pair_indices=sel, losses=losses)
                if (step+1) in spec['save_updates']:
                    for kind, matrix in matrices.items():
                        output[f'{op}_{kind}_{step+1}'] = matrix.detach().cpu().numpy().copy()
                    np.savez_compressed(w.run/f'request_writer_s{s}_t{t}.npz', **output)
                if (step+1) % 16 == 0:
                    w.progress('REQUEST_FIT', operation=op, updates=step+1, **losses, **calls)
                if time.perf_counter()-w.wall_start > cfg['budget_seconds']:
                    raise TimeoutError('Request writer pilot budget')
            metadata.append(dict(source=s, target=t, operation=op, calls=calls,
                                 fit_pairs=len(pairs), training_queries=len(queries)))
        np.savez_compressed(w.run/f'request_writer_s{s}_t{t}.npz', **output)
    write(w.run/'REQUEST_WRITER_FIT.json', dict(seconds=time.perf_counter()-started, records=metadata,
        supervision='Actual source-intervention distributions on source-generated prefixes; source-fit inputs only',
        held_out_query='Role participation split is never used in fitting',
        runtime='Frozen role matrices; target-native nonnegative-code projection; no output gradients'))
    w.checks.update(request_fit_only_original_source_pairs=True, request_fit_matched_raw_output_budget=True)


def attach_writers(w, cfg, readouts):
    import numpy as np
    import torch
    from run_causalgym_multisite import ROOT
    spec = cfg['request_writers']
    parent = ROOT/spec['run'] if spec.get('run') else w.run
    for s, t in spec['seed_pairs']:
        with np.load(w.checked(parent/f'request_writer_s{s}_t{t}.npz')) as z:
            for kind, base in [('native', 'full_activation'), ('raw', 'raw')]:
                for (seed, name, cap), operations in list(readouts.items()):
                    if seed != t or not name.startswith(base+'_readout'):
                        continue
                    for update in spec['updates']:
                        mapped = {}
                        for op, data in operations.items():
                            mapped[op] = dict(data, writer=dict(kind=kind,
                                matrix=torch.tensor(z[f'{op}_{kind}_{update}'], device=w.device),
                                decoder=torch.tensor(z[f'{op}_target_decoder'], device=w.device) if kind=='native' else data['decoder']))
                        readouts[seed, f'learned_{kind}_u{update}'+name.split('_readout', 1)[1], cap] = mapped
