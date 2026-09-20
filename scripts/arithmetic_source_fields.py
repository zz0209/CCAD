import json
from pathlib import Path

import numpy as np
import torch

from ccad.source_field_inference import infer_source_fields
from run_causalgym_multisite import ROOT, write


def evaluate_source_fields(w, cfg, model, module, tok, saes, rows, pairs, generate, budget):
    spec = cfg['source_field_evaluation']
    source = spec['source_seed']
    payloads = {}
    for target in spec['target_seeds']:
        with np.load(w.checked(ROOT / spec['readout_run'] / f'readout_s{source}_t{target}.npz')) as data:
            payloads[target] = {k: data[k] for k in data.files}
    first = payloads[spec['target_seeds'][0]]
    gate = torch.tensor(first['source_gate'], device=w.device)
    for payload in payloads.values():
        assert np.array_equal(payload['source_gate'], first['source_gate'])
    cache_panel = json.loads(w.checked(ROOT / spec['cache_panel']).read_text())['rows']
    hidden = torch.tensor(np.concatenate([np.load(w.checked(ROOT / p))['hidden'] for p in spec['hidden_caches']]), device=w.device)
    assert len(hidden) == len(cache_panel)
    fit_pairs = first['fit_pairs'][:spec['profile_pairs']]
    assert all(cache_panel[int(i)]['split'] == 'fit' for i in fit_pairs.ravel())
    cache_by_prompt = {r['prompt']: i for i, r in enumerate(cache_panel)}
    donors = sorted({p['donor'] for p in pairs})
    donor_hidden = {}
    for offset in range(0, len(donors), cfg['batch_size']):
        batch = donors[offset:offset+cfg['batch_size']]
        missing = [i for i in batch if rows[i]['prompt'] not in cache_by_prompt]
        for i in batch:
            if i not in missing:
                donor_hidden[i] = hidden[cache_by_prompt[rows[i]['prompt']]]
        if missing:
            _, _, states, _ = generate(missing)
            donor_hidden.update({i: states[j] for j, i in enumerate(missing)})
    w.progress('DONOR_STATES', completed=len(donors), total=len(donors))
    for offset in range(0, len(pairs), cfg['batch_size']):
        selected = pairs[offset:offset+cfg['batch_size']]
        answers, texts, _, _ = generate([p['recipient'] for p in selected])
        for j, p in enumerate(selected):
            w.record(kind='base', task=f'template_{p["template"]}', row_id=offset+j,
                component=f'{rows[p["recipient"]]["a"]}+{rows[p["recipient"]]["b"]}',
                method='no_edit', answer=answers[j], generated_text=texts[j])
    roots = {}
    if spec.get('profile_cache'):
        roots = torch.load(w.checked(spec['profile_cache']), map_location=w.device, weights_only=True)
    observations = {}
    for op_index, operation in enumerate(['unit', 'tens']):
        si = torch.tensor(first[f'{operation}_source_indices'], device=w.device)
        decoder = saes[source].decoder.weight.T[si]
        assert all(np.array_equal(p[f'{operation}_source_indices'], si.cpu().numpy()) for p in payloads.values())
        if operation in roots:
            continue
        gradients = []
        batches = []
        for template in [0, 1]:
            ids = [i for i, pair in enumerate(fit_pairs) if cache_panel[int(pair[0])]['template'] == template]
            batches.extend(fit_pairs[ids[offset:offset+cfg['batch_size']]] for offset in range(0, len(ids), cfg['batch_size']))
        completed = 0
        for selected in batches:
            recipients = [cache_panel[int(i)] for i in selected[:, 0]]
            ds = hidden[selected[:, 1]]
            with torch.no_grad():
                donor_codes = saes[source].encode(ds)[..., si]
            tokenrows = [tok.encode(r['prompt'], add_special_tokens=False) for r in recipients]
            template = recipients[0]['template']
            assert all(r['template'] == template for r in recipients)
            count = len(selected)
            ix = torch.arange(count, device=w.device)[:, None]
            pos = torch.tensor([len(t)-1 for t in tokenrows], device=w.device)
            answers = [10*r['tens']+cache_panel[int(d)]['unit'] if operation == 'unit'
                       else 10*cache_panel[int(d)]['tens']+r['unit'] for r, d in zip(recipients, selected[:, 1])]
            for digit_step in [1, 2]:
                length = digit_step + template
                ids = torch.full((count, int(pos.max())+length), tok.eos_token_id, dtype=torch.long, device=w.device)
                attention = torch.zeros_like(ids)
                for j, tokens in enumerate(tokenrows):
                    prefix = tok.encode((' ' if template else '') + str(answers[j]), add_special_tokens=False)[:length-1]
                    assert len(tok.encode(str(answers[j]), add_special_tokens=False)) == 2
                    assert len(prefix) == length-1
                    combined = tokens + prefix
                    ids[j, :len(combined)] = torch.tensor(combined, device=w.device)
                    attention[j, :len(combined)] = 1
                sites = pos[:, None] + torch.arange(length, device=w.device)[None]
                roles = torch.arange(1-template, length+1-template, device=w.device)
                expected = torch.tensor([tok.encode(str(a), add_special_tokens=False)[digit_step-1] for a in answers], device=w.device)
                alternatives = []
                for j, (r, d) in enumerate(zip(recipients, selected[:, 1])):
                    alt = r['total'] if (operation == 'unit') == (digit_step == 2) else cache_panel[int(d)]['total']
                    alternatives.append(tok.encode(str(alt), add_special_tokens=False)[digit_step-1])
                other = torch.tensor(alternatives, device=w.device)
                assert bool((expected != other).all())
                for fraction in spec['profile_fractions']:
                    state = {}
                    def hook(_m, _a, out):
                        h = out[0] if isinstance(out, tuple) else out
                        current = h[ix, sites]
                        with torch.no_grad():
                            amplitude = (donor_codes[:, :length] - saes[source].encode(current)[..., si]) * gate[roles][:, si, op_index]
                            edited = current + fraction * (amplitude @ decoder)
                        edited = edited.detach().requires_grad_(True)
                        state['edited'] = edited
                        result = h.detach().clone()
                        result[ix, sites] = edited
                        return (result,) + out[1:] if isinstance(out, tuple) else result
                    handle = module.register_forward_hook(hook)
                    try:
                        logits = model(ids, attention_mask=attention, use_cache=False).logits[torch.arange(count, device=w.device), pos+length-1]
                        margin = logits[torch.arange(count, device=w.device), expected] - logits[torch.arange(count, device=w.device), other]
                        gradient = torch.autograd.grad(margin.sum(), state['edited'])[0]
                        assert torch.isfinite(gradient).all()
                        gradients.append(gradient.detach().reshape(-1, gradient.shape[-1]).cpu())
                    finally:
                        handle.remove()
                    del logits, margin, gradient, state
                    w.sequence_forwards += count
                    w.token_forwards += ids.numel()
                    budget()
            completed += count
            w.progress('SOURCE_PROFILE', operation=operation, completed=completed, total=len(fit_pairs))
        g = torch.cat(gradients).to(w.device)
        gram = g.T @ g / len(g)
        gram = gram / (gram.trace() / len(gram)).clamp_min(1e-12)
        gram = (1-spec['identity_fraction'])*gram + spec['identity_fraction']*torch.eye(len(gram), device=w.device)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram.double())
        assert eigenvalues.min() > 0
        roots[operation] = (eigenvectors * eigenvalues.sqrt()[None]).float()
        observations[operation] = dict(gradient_vectors=len(g), trace=float(gram.trace()), min_eigenvalue=float(eigenvalues.min()))
        torch.save({k: v.cpu() for k, v in roots.items()}, w.run/'source_profile.pt')
        np.savez_compressed(w.run/f'{operation}_profile_gradients.npz', gradients=g.cpu().numpy(), fit_pairs=fit_pairs)
    write(w.run/'PROFILE.json', dict(fit_pairs=fit_pairs.tolist(), observations=observations,
        output_information='Original source-fit hybrid and protected digit margins at four source-path fractions',
        target_response_fits=0, source_seed=source))
    query_rng = np.random.default_rng(spec['query_seed'])
    q = [np.ones(64, dtype=np.float32)]
    for _ in range(spec['partitions']):
        half = np.zeros(64, dtype=np.float32)
        half[query_rng.permutation(64)[:32]] = 1
        q.extend([half, 1-half])
    queries = torch.tensor(np.stack(q), device=w.device)
    np.save(w.run/'queries.npy', queries.cpu().numpy())
    for op_index, operation in enumerate(['unit', 'tens']):
        si = torch.tensor(first[f'{operation}_source_indices'], device=w.device)
        decoder = saes[source].decoder.weight.T[si]
        for target in spec['target_seeds']:
            payload = payloads[target]
            relation_path = ROOT / spec['relation_run'] / f'relation_s{source}_t{target}.npz'
            with np.load(w.checked(relation_path)) as data:
                relation = torch.tensor(data[f'{operation}_weights'], device=w.device)
                ti = torch.tensor(data[f'{operation}_target_indices'], device=w.device)
            ri = torch.tensor(payload[f'{operation}_target_indices'], device=w.device)
            reader = torch.tensor(payload[f'{operation}_full_activation'][:, ri.cpu().numpy()], device=w.device)
            for query_id, query in enumerate(queries):
                for method in spec['methods']:
                    if method == 'source' and target != spec['target_seeds'][0]:
                        continue
                    for offset in range(0, len(pairs), cfg['batch_size']):
                        pp = pairs[offset:offset+cfg['batch_size']]
                        recipients = [p['recipient'] for p in pp]
                        donor = torch.stack([donor_hidden[p['donor']] for p in pp])
                        with torch.no_grad():
                            source_donor = saes[source].encode(donor)[..., si]
                            target_donor = saes[target].encode(donor)
                        diagnostics = []
                        def patch(current, step):
                            roles = torch.arange(step+1, device=w.device)[None] + torch.tensor([1-rows[i]['template'] for i in recipients], device=w.device)[:, None]
                            if method in ['source', 'euclidean', 'profile']:
                                amplitudes = (source_donor[:, :step+1] - saes[source].encode(current)[..., si]) * gate[roles][..., si, op_index]
                                if method == 'source':
                                    return (amplitudes * query) @ decoder
                                columns = infer_source_fields(current, saes[target], decoder, amplitudes, spec['members'],
                                    roots[operation] if method == 'profile' else None, spec['inverse_steps'])
                                change = columns @ query
                                code = saes[target].encode(current)
                                violation = float((-code-change).clamp_min(0).max())
                                support = int((change != 0).sum(-1).max())
                                assert violation < 1e-3 and support <= spec['members'], (violation, support)
                                diagnostics.append(dict(step=step, capacity_violation=violation, support=support))
                                return change @ saes[target].decoder.weight.T
                            read_indices = ri if method == 'readout' else ti
                            difference = (target_donor[:, :step+1] - saes[target].encode(current))[..., read_indices]
                            if method == 'relation':
                                return (difference * (relation[roles] @ query)) @ saes[target].decoder.weight.T[ti]
                            if method == 'readout':
                                amplitude = torch.einsum('blj,blji->bli', difference, reader[roles])
                                return (amplitude * query) @ decoder
                            raise ValueError(method)
                        answers, texts, _, norms = generate(recipients, patch=patch)
                        for j, p in enumerate(pp):
                            w.record(kind='source_field', task=f'template_{p["template"]}', row_id=offset+j,
                                component=f'{rows[p["recipient"]]["a"]}+{rows[p["recipient"]]["b"]}',
                                seed=source, target_seed=target, mode=f'query_{query_id}', method=method, operation=operation,
                                answer=answers[j], generated_text=texts[j], expected_answer=p[operation+'_answer'],
                                exact_hybrid=answers[j] == p[operation+'_answer'], edit_norm=float(norms[j]))
                        with (w.run/'capacity.jsonl').open('a') as handle:
                            handle.write(json.dumps(dict(operation=operation, target=target, query=query_id,
                                method=method, offset=offset, diagnostics=diagnostics))+'\n')
                        budget()
                    w.progress('FIELD_GENERATION', operation=operation, target=target, query=query_id, method=method)
    w.checks['source_profile_fit_split'] = True
    w.checks['native_capacity_and_budget'] = True
