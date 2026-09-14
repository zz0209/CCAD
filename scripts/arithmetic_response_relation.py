"""Fit target memberships to finite source responses in two answer roles.

Each source supplies the finite change of requested/preserved digit margins.
The shared model supplies per-example Jacobians. The same bank also supports
direct gradient member selection and field fitting on identical prefix states.
Free generation remains the external consumer.
"""
import json
from datetime import datetime, timezone

from fit_component_correspondence import fit


def fit_response_relations(w, cfg, model, module, tok, rows, tokenrows, saes,
                           codes, physical_length, budget):
    import numpy as np
    import torch

    # Tiny finite-difference checks need full-precision products; cached donor
    # codes and the external generation consumer retain their existing precision.
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    rc = cfg['response_relation']
    root = w.run.parent.parent
    prior = root / rc['relation_run']
    assert json.loads(w.checked(prior/'status.json').read_text())['status'] == 'PASS'
    prior_cfg = json.loads(w.checked(prior/'config.resolved.json').read_text())
    assert all(prior_cfg[k] == cfg[k] for k in ['training_run','checkpoint_step','model_revision','source_cache_run'])
    assert json.loads(w.checked(prior/'panel.json').read_text())['rows'] == rows
    assert len(set(s for s,t in rc['seed_pairs'])) == len(rc['seed_pairs'])
    assert len(set(t for s,t in rc['seed_pairs'])) == len(rc['seed_pairs'])
    cap, width = cfg['members'][0], saes[cfg['seeds'][0]].decoder.weight.shape[1]
    pairs = None
    source_gates, original_fields = {}, {}
    for s, t in rc['seed_pairs']:
        old_t = s % 5 + 1
        with np.load(w.checked(prior / f'relation_s{s}_t{old_t}.npz')) as z:
            pp = z['fit_pairs'].tolist()
            if pairs is None:
                pairs = pp
            assert pairs == pp
            source_gates[s] = torch.tensor(z['source_gate'], device=w.device)
            if old_t == t:
                original_fields[s, t] = torch.tensor(z['clean'], device=w.device)
    assert len(pairs) == rc['fit_pairs']
    assert all(rows[i]['split'] == 'fit' for pair in pairs for i in pair)
    batch = cfg['batch_size']
    digit_tokens = [tok.encode(str(i), add_special_tokens=False) for i in range(10)]
    assert all(len(x) == 1 for x in digit_tokens)
    digit_tokens = torch.tensor([x[0] for x in digit_tokens], device=w.device)
    result, meta = {}, []
    method_specs = rc.get('methods', [dict(name='response',anchor=0.),
        dict(name='response_anchor',anchor=rc['field_anchor']),
        dict(name='direct_profile',anchor=0.),dict(name='field_teacher',anchor=None)])
    banks = []
    forward_count, backward_count = 0, 0

    for operation in range(2):
        target_matrices = {t: [] for _, t in rc['seed_pairs']}
        current_codes = {t: [] for _, t in rc['seed_pairs']}
        source_fields = {s: [] for s, _ in rc['seed_pairs']}
        source_effects = {s: [] for s, _ in rc['seed_pairs']}
        source_linear = {s: [] for s, _ in rc['seed_pairs']}
        source_swapped = {s: [] for s, _ in rc['seed_pairs']}
        bank_h, bank_j, bank_active, bank_base = [], [], [], []
        for off in range(0, len(pairs), batch):
            pp = pairs[off:off + batch]
            assert len(pp) == batch
            ids = torch.full((batch, physical_length), tok.eos_token_id, device=w.device, dtype=torch.long)
            attention = torch.zeros_like(ids)
            positions, lengths, role_sites, desired, alternative = [], [], [], [], []
            for b, (i, j) in enumerate(pp):
                answer = 10*rows[i]['tens']+rows[j]['unit'] if operation == 0 else 10*rows[j]['tens']+rows[i]['unit']
                prefix = ' ' if rows[i]['template'] == 1 else ''
                assert rows[i]['template'] == rows[j]['template']
                output = tok.encode(prefix+str(answer), add_special_tokens=False)
                assert len(output) == 2+len(prefix)
                seq = tokenrows[i]+output
                ids[b, :len(seq)] = torch.tensor(seq, device=w.device)
                attention[b, :len(seq)] = 1
                positions.append(len(tokenrows[i])-1)
                lengths.append(len(output)+1)
                # The two rows are always requested then preserved, independent of digit order.
                ds = [rows[j]['unit'], rows[i]['tens']] if operation == 0 else [rows[j]['tens'], rows[i]['unit']]
                alt = [rows[i]['unit'], rows[j]['tens']] if operation == 0 else [rows[i]['tens'], rows[j]['unit']]
                role_sites.append([len(prefix)+1, len(prefix)] if operation == 0 else [len(prefix), len(prefix)+1])
                desired.append(ds); alternative.append(alt)
            nsites = max(lengths)
            ix = torch.arange(batch, device=w.device)
            sites = torch.tensor(positions, device=w.device)[:, None]+torch.arange(nsites, device=w.device)
            active = torch.arange(nsites, device=w.device)[None, :] < torch.tensor(lengths, device=w.device)[:, None]
            logit_sites = torch.tensor(positions, device=w.device)[:, None]+torch.tensor(role_sites, device=w.device)
            desired = digit_tokens[torch.tensor(desired, device=w.device)]
            alternative = digit_tokens[torch.tensor(alternative, device=w.device)]
            cache = {}

            def forward(source=None, test_delta=None, differentiable=False, source_operation=None):
                nonlocal forward_count
                def hook(_m, _a, output):
                    h = output[0] if isinstance(output, tuple) else output
                    current = h[ix[:, None], sites]
                    if differentiable:
                        leaf = current.detach().clone().requires_grad_(True)
                        cache['leaf'] = leaf
                        change = leaf-current
                    elif source is not None:
                        ae = saes[source]
                        z = ae.encode(current.reshape(-1, current.shape[-1])).reshape(batch, nsites, -1)
                        co = operation if source_operation is None else source_operation
                        change = ((codes[source][[j for i, j in pp], :nsites]-z)*source_gates[source][:, co]) @ ae.decoder.weight.T
                    else:
                        change = test_delta
                    out = h.clone()
                    out[ix[:, None], sites] += change*active[:, :, None]
                    return (out,)+output[1:] if isinstance(output, tuple) else out
                handle = module.register_forward_hook(hook)
                try:
                    outputs = model(ids, attention_mask=attention, use_cache=False)
                    logits = outputs.logits[ix[:, None], logit_sites]
                    wanted = logits.gather(2, desired[:, :, None])
                    if rc.get('digit_profile'):
                        margin = (wanted-logits[:, :, digit_tokens]).reshape(batch,-1)
                    else:
                        margin = wanted.squeeze(-1)-logits.gather(2, alternative[:, :, None]).squeeze(-1)
                    w.sequence_forwards += batch; w.token_forwards += batch*physical_length
                    forward_count += 1
                    return margin
                finally:
                    handle.remove()

            margins = forward(differentiable=True)
            nresponses = margins.shape[1]
            gradients = [torch.autograd.grad(margins[:, role].sum(), cache['leaf'], retain_graph=role+1<nresponses)[0].detach() for role in range(nresponses)]
            backward_count += nresponses
            jac = torch.stack(gradients, 1)
            current = cache['leaf'].detach()
            base_margin = margins.detach()
            del margins
            bank_h.append(current.cpu()); bank_j.append(jac.cpu()); bank_active.append(active.cpu()); bank_base.append(base_margin.cpu())
            for s, t in rc['seed_pairs']:
                with torch.no_grad():
                    zs = saes[s].encode(current.reshape(-1, current.shape[-1])).reshape(batch, nsites, -1)
                    y = ((codes[s][[j for i, j in pp], :nsites]-zs)*source_gates[s][:, operation]) @ saes[s].decoder.weight.T
                    y = y*active[:, :, None]
                    effect = forward(source=s)-base_margin
                    zt = saes[t].encode(current.reshape(-1, current.shape[-1])).reshape(batch, nsites, -1)
                    x = (codes[t][[j for i, j in pp], :nsites]-zt)*active[:, :, None]
                    a = (torch.einsum('brjd,fd->brjf', jac, saes[t].decoder.weight.T)*x[:, None]).sum(2)
                    target_matrices[t].append(a); current_codes[t].append(x)
                    source_fields[s].append(y); source_effects[s].append(effect)
                    source_linear[s].append(torch.einsum('brjd,bjd->br',jac,y))
                    if any(sp.get('source_response') == 'swapped' for sp in method_specs):
                        source_swapped[s].append(forward(source=s,source_operation=1-operation)-base_margin)
                    if off == 0 and s == rc['seed_pairs'][0][0]:
                        feature = int(a.square().sum((0, 1)).argmax())
                        delta = x[:, :, feature, None]*saes[t].decoder.weight[:, feature]
                        eps = .001
                        finite = (forward(test_delta=eps*delta)-forward(test_delta=-eps*delta))/(2*eps)
                        error = float((finite-a[:, :, feature]).abs().max())
                        bound = .025+.02*float(a[:, :, feature].abs().max())
                        check = dict(operation=operation,error=error,tolerance=bound,
                                     analytic=a[:,:,feature].cpu().tolist(),finite=finite.cpu().tolist())
                        (w.run/f'JACOBIAN_CHECK_{operation}.json').write_text(json.dumps(check,indent=2)+'\n')
                        w.progress('RESPONSE_JACOBIAN_CHECK', operation=operation,error=error,tolerance=bound)
                        assert error < bound, (error, bound)
                        w.checks[f'response_jacobian_operation{operation}'] = True
            cache.clear()
            budget()
        # Pad variable prefix lengths across batches, preserving zero inactive sites.
        def padcat(parts):
            max_sites = max(p.shape[1] for p in parts)
            return torch.cat([torch.nn.functional.pad(p, (0, 0, 0, max_sites-p.shape[1])) for p in parts])
        banks.append(dict(operation=operation, jacobians=[j.numpy() for j in bank_j], states=[h.numpy() for h in bank_h],
                          active=[a.numpy() for a in bank_active], base_margins=torch.cat(bank_base).numpy()))
        for s, t in rc['seed_pairs']:
            a = torch.cat(target_matrices[t]).reshape(-1, width).double()
            b = torch.cat(source_effects[s]).reshape(-1, 1).double()
            b_finite = b
            b_linear = torch.cat(source_linear[s]).reshape(-1, 1).double()
            x = padcat(current_codes[t]).reshape(-1, width).double()
            y = padcat(source_fields[s]).reshape(-1, saes[s].decoder.weight.shape[0]).double()
            d = saes[t].decoder.weight.T.double()
            field_energy = y.square().sum(1).mean().clamp_min(1e-10)
            bf = (x*(y@d.T)).mean(0)
            fe = x.square().mean(0)*d.square().sum(1)
            field_score = bf.clamp_min(0).square()/fe.clamp_min(1e-10)
            payload = dict(response_matrix=a.float().cpu().numpy(), source_response=b_finite.float().cpu().numpy(),
                           source_linear_response=b_linear.float().cpu().numpy())
            for spec in method_specs:
                name, anchor = spec['name'], spec['anchor']
                b = b_linear if spec.get('source_response') == 'linear' else b_finite
                if spec.get('source_response') == 'swapped':
                    b = torch.cat(source_swapped[s]).reshape(-1,1).double()
                    payload['swapped_source_response'] = b.float().cpu().numpy()
                response_energy = b.square().mean().clamp_min(1e-10)
                ar = a.T @ b / len(a)
                response_score = ar[:, 0].clamp_min(0).square()/a.square().mean(0).clamp_min(1e-10)
                direct = name == 'direct_profile' or spec.get('selection') == 'direct'
                score = field_score if anchor is None else a.mean(0) if direct else response_score
                pool = torch.argsort(score, descending=True, stable=True)[:cap if direct else rc['pool']]
                ap, xp, dp = a[:, pool], x[:, pool], d[pool]
                kf = (xp.T@xp/len(xp))*(dp@dp.T)/field_energy
                br = ar[pool]/response_energy
                kr = ap.T@ap/len(ap)/response_energy
                if anchor is None:
                    k, rhs = kf, bf[pool, None]/field_energy
                else:
                    k, rhs = kr+anchor*kf, br+anchor*bf[pool, None]/field_energy
                if spec.get('weights') == 'scalar':
                    assert direct
                    alpha = (rhs.sum()/(k.sum()+rc['ridge_fraction']*k.diag().mean()*len(pool)).clamp_min(1e-12)).clamp(0,1)
                    coef = torch.ones_like(rhs)*alpha
                    info = dict(scalar=float(alpha),closed_form=True,ridge_fraction=rc['ridge_fraction'])
                else:
                    coef, _ = fit(k, rhs, steps=rc['steps'], ridge_fraction=rc['ridge_fraction'], capacity=False)
                    keep = torch.argsort(coef[:, 0].square()*k.diag(), descending=True, stable=True)[:cap]
                    allowed = torch.zeros_like(coef, dtype=torch.bool); allowed[keep] = True
                    coef, info = fit(k, rhs, steps=rc['steps'], ridge_fraction=rc['ridge_fraction'], allowed=allowed, capacity=False)
                g = torch.zeros(width, device=w.device);g[pool] = coef[:, 0].float()
                key = (t, f'response_s{s}_{name}', cap)
                if key not in result:result[key] = torch.zeros(width, 2, device=w.device)
                result[key][:, operation] = g
                payload[name] = g.cpu().numpy()
                err = float((a@g.double()[:, None]-b).square().sum()/b.square().sum().clamp_min(1e-10))
                meta.append(dict(source=s, target=t, operation=operation, method=name, response_relative_mse=err, solver=info))
            np.savez_compressed(w.run/f'response_s{s}_t{t}_op{operation}.npz', **payload)
        w.progress('RESPONSE_RELATION_FIT', operation=operation, forward_batches=forward_count, backward_batches=backward_count)

    for s, t in rc['seed_pairs']:
        for name in [sp['name'] for sp in method_specs]:
            g = result[t, f'response_s{s}_{name}', cap]
            assert bool(((g>=0)&(g<=1)).all()) and bool(((g>0).sum(0)<=cap).all())
        payload = {sp['name']:result[t, f"response_s{s}_{sp['name']}", cap].cpu().numpy() for sp in method_specs}
        np.savez_compressed(w.run/f'response_s{s}_t{t}.npz', **payload, fit_pairs=np.array(pairs))
        result[s, 'source_views', cap] = source_gates[s]
        if (s, t) in original_fields:result[t, f'response_s{s}_field_prior', cap] = original_fields[s, t]
    if rc.get('direct_run'):
        direct = root / rc['direct_run']
        assert json.loads(w.checked(direct/'status.json').read_text())['status'] == 'PASS'
        dc = json.loads(w.checked(direct/'config.resolved.json').read_text())
        assert all(dc[k] == cfg[k] for k in ['training_run','checkpoint_step','model_revision','source_cache_run'])
        for s in cfg['seeds']:
            with np.load(w.checked(direct/f'counterfactual_seed{s}_k{cap}_target_gradient.npz')) as z:
                result[s,'direct_320',cap] = torch.tensor(z['updates_320'],device=w.device)
    # The bank is a list because mixed symbolic/English batches have three/four sites.
    arrays = {'fit_pairs':np.array(pairs)}
    for b in banks:
        op=b['operation'];arrays[f'base_margins_{op}']=b['base_margins']
        for key in ['jacobians','states','active']:
            for i,value in enumerate(b[key]):arrays[f'{key}_{op}_{i}']=value
    np.savez_compressed(w.run/'response_bank.npz', **arrays)
    (w.run/'RESPONSE_FREEZE.json').write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        forward_batches=forward_count, backward_batches=backward_count, source_fit_pairs=len(pairs), fits=meta,
        responses_per_pair=nresponses, methods=method_specs,
        scope='Known source-task digit roles; source finite responses and common model Jacobians. All methods share the fit bank. No new target-specific model backward pass during the convex correspondence solve. Evaluation uses free generation, with all failures retained.'),indent=2)+'\n')
    torch.set_float32_matmul_precision(previous_precision)
    return result, meta
