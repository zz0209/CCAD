"""Evaluate a fixed published functional distinction in new dictionaries.

The source node identities and roles come from SFC Figure18/annotations.
Controlled contexts are explicitly authored here, rather than relabeled as the
original public cluster. The relation solver reuses CCAD's existing field fit.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from run_causalgym_multisite import MultisiteWork, write
from run_shift_transfer import fit_members
from train_shift_dictionaries import site_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    work = MultisiteWork(cfg, args.config, ['scripts/run_published_parts.py',
        'scripts/run_shift_transfer.py', 'scripts/train_shift_dictionaries.py',
        'scripts/run_causalgym_multisite.py', 'scripts/run_r011s1_raw_hook_asset.py',
        'src/ccad/artifacts.py'])
    handle, error = None, None
    try:
        import torch
        import transformers
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        torch.cuda.set_device(cfg['device'])
        torch.cuda.reset_peak_memory_stats()
        device = torch.device(cfg['device'])
        work.device, work.torch = device, torch
        work.environment = dict(python=sys.executable, torch=torch.__version__,
                                transformers=transformers.__version__, numpy=np.__version__)
        panel = json.loads(work.checked(cfg['panel']).read_text())
        bank = np.load(work.checked(cfg['source_parameters']))
        s = {k: torch.tensor(bank[k], device=device) for k in ['encoder', 'encoder_bias', 'decoder', 'center']}
        for name in ['config.json', 'model.safetensors', 'tokenizer.json']:
            work.checked(Path(cfg['model_local_dir'])/name)
        model = transformers.AutoModelForCausalLM.from_pretrained(cfg['model_local_dir'],
            local_files_only=True, dtype=torch.float32, attn_implementation='eager').eval().to(device)
        model.requires_grad_(False)
        tokenizer = transformers.AutoTokenizer.from_pretrained(cfg['model_local_dir'], local_files_only=True)
        answer = tokenizer.encode(panel['answer'], add_special_tokens=False)
        assert len(answer) == 1
        token_rows = [tokenizer.encode(r['text'], add_special_tokens=False) for r in panel['rows']]
        method, q = 'none', None
        observed, relations, target, write_records = [], {}, None, []
        response_state, fit_alpha, capture_attention = None, 0., None
        functional_weight = cfg.get('functional_weight', 0.)
        queries = panel.get('queries', {'full': [1., 1., 1., 1.], 'predicate': [1., 1., 1., 0.], 'object': [0., 0., 0., 1.]})

        def source_z(h):
            return torch.relu((h-s['center']) @ s['encoder'].T+s['encoder_bias'])

        def hook(module, inputs, output):
            nonlocal response_state
            h = output[0] if isinstance(output, tuple) else output
            if method == 'capture':
                states = h.detach().flatten(0, 1)
                if capture_attention is not None:
                    states = states[capture_attention.flatten().bool()]
                observed.append(states.cpu().numpy().copy())
                if functional_weight:
                    h = (h-fit_alpha*(source_z(h) @ s['decoder'])).detach().requires_grad_(True)
                    response_state = h
            elif method == 'source':
                h = h-(source_z(h)*q) @ s['decoder']
            elif method != 'none':
                z = target.encode(h)
                if method in ['raw', 'raw_full']:
                    read = z if method == 'raw_full' else z[..., relations['candidates']]
                    h = h-(read @ relations[method]*q) @ s['decoder']
                elif method == 'raw_reconstruction':
                    reconstructed = z @ target.decoder.weight.T+target.b_dec
                    h = h-(source_z(reconstructed)*q) @ s['decoder']
                elif method.startswith(('native_response_relation', 'native_tangent_relation', 'native_response_gain')):
                    # One input-dependent column per source member. Selection and
                    # negative-capacity allocation happen before applying q, so
                    # the same relation supports every subsequent part request.
                    read_state = z @ target.decoder.weight.T+target.b_dec if method.endswith('_read') else h
                    source_codes = source_z(read_state)
                    if method.startswith('native_response_gain'):
                        columns = relations['response_gain']*source_codes.unsqueeze(-2)
                    elif method.startswith('native_tangent_relation'):
                        # Exact encoder derivative inside the current ReLU/TopK
                        # region, with the same source codes and decoder columns.
                        directions = -(target.encoder.weight @ s['decoder'].T)
                        columns = directions*source_codes.unsqueeze(-2)*(z>0).unsqueeze(-1)
                    else:
                        counterfactuals = h.unsqueeze(-2)-source_codes.unsqueeze(-1)*s['decoder']
                        columns = (target.encode(counterfactuals)-z.unsqueeze(-2)).transpose(-1, -2)
                        # Batched GEMM can round identical inputs differently
                        # from the clean call. A zero source edit is exactly zero.
                        columns = columns*(source_codes != 0).unsqueeze(-2)
                    if '_8' in method:
                        score = columns.abs().sum(-1)*target.decoder.weight.norm(dim=0)
                        selected = score.topk(8, dim=-1).indices
                        keep = torch.zeros_like(score).scatter(-1, selected, 1)
                        columns = columns*keep.unsqueeze(-1)
                    positive, negative = columns.clamp_min(0), (-columns).clamp_min(0)
                    scale = (z/negative.sum(-1).clamp_min(1e-20)).clamp_max(1)
                    columns = positive-negative*scale.unsqueeze(-1)
                    delta = columns @ q
                    write_records.append(dict(method=method,query_mask=q.cpu().tolist(),
                        states=delta.numel()//delta.shape[-1], changed_sum=int((delta!=0).sum()),
                        increased_sum=int((delta>0).sum()), decreased_sum=int((delta<0).sum()),
                        minimum_final_code=float((z+delta).min()),
                        capacity_scaled_rows=int(((scale<1)&(negative.sum(-1)>0)).sum())))
                    h = h+delta @ target.decoder.weight.T
                elif method.startswith('native_reencode'):
                    read_state = z @ target.decoder.weight.T+target.b_dec if method == 'native_reencode_read' else h
                    proposed = h-(source_z(read_state)*q) @ s['decoder']
                    edited_z = target.encode(proposed)
                    delta = edited_z-z
                    if method == 'native_reencode_down':
                        delta = delta.clamp_max(0)
                    elif method == 'native_reencode_up':
                        delta = delta.clamp_min(0)
                    elif method == 'native_reencode_top8':
                        selected = (delta.abs()*target.decoder.weight.norm(dim=0)).topk(8, dim=-1).indices
                        delta = torch.zeros_like(delta).scatter(-1, selected, delta.gather(-1, selected))
                    write_records.append(dict(method=method,query_mask=q.cpu().tolist(),
                        states=delta.numel()//delta.shape[-1], changed_sum=int((delta!=0).sum()),
                        increased_sum=int((delta>0).sum()), decreased_sum=int((delta<0).sum()),
                        minimum_final_code=float((z+delta).min())))
                    h = h+delta @ target.decoder.weight.T
                else:
                    h = h-(z*(relations[method] @ q)) @ target.decoder.weight.T
            return (h, *output[1:]) if isinstance(output, tuple) else h

        handle = site_module(model, 'resid_4').register_forward_hook(hook)
        if cfg.get('target_seed') is not None:
            sys.path.extend([cfg['dictionary_source_dir'], cfg['dictionary_overlay_dir']])
            from dictionary_learning.trainers.top_k import AutoEncoderTopK
            state = torch.load(work.checked(Path(cfg['target_directory'])/f'resid_4_seed{cfg["target_seed"]}.pt'),
                               map_location=device, weights_only=True)
            target = AutoEncoderTopK(512, state['encoder.weight'].shape[0], int(state['k'])).to(device)
            target.load_state_dict(state)
            target.requires_grad_(False)
            if cfg.get('relation_run'):
                saved = np.load(work.checked(Path(cfg['relation_run'])/'relation.npz'))
                relations = {k: torch.tensor(saved[k], device=device,
                    dtype=torch.long if k == 'candidates' else torch.float32) for k in saved.files}
            else:
                method = 'capture'
                if cfg.get('fit_panel'):
                    fitting = json.loads(work.checked(cfg['fit_panel']).read_text())
                    fit_rows = [tokenizer.encode(r['text'], add_special_tokens=False) for r in fitting['rows']]
                    natural = np.zeros((len(fit_rows), max(map(len, fit_rows))), dtype='int64')
                    fit_attention = np.zeros_like(natural)
                    for j, row in enumerate(fit_rows):
                        natural[j, :len(row)], fit_attention[j, :len(row)] = row, 1
                else:
                    natural = np.memmap(work.checked(cfg['paired_tokens']), dtype='<u2', mode='r').reshape(-1, 128)[:128]
                    fit_attention = np.ones_like(natural)
                gradients = []
                for fit_alpha in ([0., 1.] if functional_weight else [0.]):
                    for off in range(0, len(natural), 8):
                        ids = torch.tensor(np.array(natural[off:off+8], dtype='int64'), device=device)
                        capture_attention = torch.tensor(fit_attention[off:off+8], device=device)
                        with torch.set_grad_enabled(bool(functional_weight)):
                            if functional_weight:
                                values = model(ids, attention_mask=capture_attention, use_cache=False).logits
                                score = (torch.log_softmax(values, dim=-1)[..., answer[0]]*capture_attention).sum()
                                gradient = torch.autograd.grad(score, response_state)[0]
                                gradients.append(gradient.detach().flatten(0, 1)[capture_attention.flatten().bool()].cpu().numpy().copy())
                            else:
                                model.gpt_neox(ids, attention_mask=capture_attention, use_cache=False)
                        work.sequence_forwards += len(ids)
                        work.token_forwards += ids.numel()
                x = torch.tensor(np.concatenate(observed), device=device)
                observed.clear()
                capture_attention = None
                dt = target.decoder.weight.T.detach()
                cosine = (dt/dt.norm(dim=1)[:, None]) @ (s['decoder']/s['decoder'].norm(dim=1)[:, None]).T
                index = torch.topk(cosine, 16, dim=0).indices.flatten().unique()
                with torch.no_grad():
                    zs = source_z(x).cpu().numpy().astype('float64')
                    zt = np.concatenate([target.encode(xx)[:, index].cpu().numpy()
                        for xx in x.split(512)]).astype('float64')
                di, ds = dt[index].cpu().numpy().astype('float64'), s['decoder'].cpu().numpy().astype('float64')
                covariance, cross = zt.T @ zt/len(x), zt.T @ zs/len(x)
                ke, be = covariance*(di @ di.T), cross*(di @ ds.T)
                k, b = ke, be
                if functional_weight:
                    gs = np.concatenate(gradients).astype('float64')
                    at, ass = zt*(gs @ di.T), zs*(gs @ ds.T)
                    kf, bf = at.T @ at/len(at), at.T @ ass/len(at)
                    se, sf = max(np.trace(ke)/len(ke), 1e-15), max(np.trace(kf)/len(kf), 1e-15)
                    k = (1-functional_weight)*ke/se+functional_weight*kf/sf
                    b = (1-functional_weight)*be/se+functional_weight*bf/sf
                fitted = fit_members(k, b, 1200)
                chosen = np.argsort(-fitted.sum(1)*np.sqrt(np.maximum(np.diag(k), 0)), kind='stable')[:8]
                native = np.zeros((len(dt), 4))
                native[index.cpu().numpy()[chosen]] = fit_members(k[np.ix_(chosen, chosen)], b[chosen], 1200)
                geometry, used = np.zeros_like(native), set()
                for j in range(4):
                    for idx in index[torch.argsort(cosine[index, j], descending=True)].cpu().tolist():
                        if idx not in used:
                            geometry[idx, j] = 1.
                            used.add(idx)
                        if geometry[:, j].sum() == 2:
                            break
                gain = geometry.copy()
                lookup = {int(v): i for i, v in enumerate(index.cpu().tolist())}
                for j in range(4):
                    ids = np.flatnonzero(geometry[:, j])
                    loc = [lookup[int(i)] for i in ids]
                    gain[ids, j] *= np.clip(b[loc, j].sum()/max(k[np.ix_(loc, loc)].sum(), 1e-15), 0, 1)
                ridge = .001*max(np.trace(covariance)/len(covariance), 1e-12)
                raw = np.linalg.solve(covariance+ridge*np.eye(len(covariance)), cross)
                if functional_weight:
                    for j in range(4):
                        weights = (gs @ ds[j])**2
                        weights = (1-functional_weight)+functional_weight*weights/max(weights.mean(), 1e-15)
                        weighted_cov = zt.T @ (zt*weights[:, None])/len(zt)
                        weighted_cross = zt.T @ (zs[:, j]*weights)/len(zt)
                        regularizer = .001*max(np.trace(weighted_cov)/len(weighted_cov), 1e-12)
                        raw[:, j] = np.linalg.solve(weighted_cov+regularizer*np.eye(len(weighted_cov)), weighted_cross)
                arrays = dict(native=native, geometry=geometry, geometry_gain=gain, raw=raw, candidates=index.cpu().numpy())
                read_fit = {}
                if 'raw_full' in cfg['methods']:
                    if functional_weight:
                        raise ValueError('Full-code diagnostic currently uses the common field-only fit')
                    # The diagnostic exposes all available target codes. The old
                    # raw comparator deliberately remains restricted to candidates.
                    with torch.no_grad():
                        all_z = torch.cat([target.encode(xx) for xx in x.split(512)])
                        ys = source_z(x)
                        lam = .001*(all_z.square().mean(0).mean()).clamp_min(1e-12)
                        rhs = all_z.T @ ys/len(x)
                        diagonal = all_z.square().mean(0)[:, None]+lam
                        weights = torch.zeros_like(rhs)
                        residual = rhs.clone()
                        preconditioned = residual/diagonal
                        direction = preconditioned.clone()
                        rz = (residual*preconditioned).sum(0)
                        norm = rhs.square().sum(0).clamp_min(1e-30)
                        for iteration in range(400):
                            kd = all_z.T @ (all_z @ direction)/len(x)+lam*direction
                            step = rz/(direction*kd).sum(0).clamp_min(1e-30)
                            weights += direction*step
                            residual -= kd*step
                            rel = (residual.square().sum(0)/norm).sqrt()
                            if rel.max() < 1e-5:
                                break
                            preconditioned = residual/diagonal
                            next_rz = (residual*preconditioned).sum(0)
                            direction = preconditioned+direction*(next_rz/rz.clamp_min(1e-30))
                            rz = next_rz
                        exact_residual = rhs-(all_z.T @ (all_z @ weights)/len(x)+lam*weights)
                        read_fit = dict(all_target_codes=all_z.shape[1], iterations=iteration+1,
                            regularization=float(lam), normal_equation_relative_residual=(exact_residual.norm(dim=0)/rhs.norm(dim=0).clamp_min(1e-15)).tolist(),
                            source_coefficient_nrmse=((all_z@weights-ys).square().mean(0)/ys.square().mean(0).clamp_min(1e-15)).sqrt().tolist())
                        arrays['raw_full'] = weights.cpu().numpy()
                        np.savez_compressed(work.run/'fit_states.npz', h=x.cpu().numpy())
                        del all_z, ys
                np.savez_compressed(work.run/'relation.npz', **arrays)
                relations = {key: torch.tensor(value, device=device,
                    dtype=torch.long if key == 'candidates' else torch.float32) for key, value in arrays.items()}
                write(work.run/'FIT.json', dict(states=len(x), candidates=len(index), source_members=4,
                    target_allowance=8, target_row_sum=float(native.sum(1).max()), source_activation_rate=(zs>0).mean(0).tolist(),
                    functional_weight=functional_weight,
                    full_code_readout=read_fit,
                    fitting_contexts=cfg.get('fit_panel', cfg['paired_tokens']),
                    fitting='Existing field/source-response fit on specified clean/source-deleted states. Native/candidate-raw share geometric candidates; raw_full explicitly uses all target codes. Context coverage and source coefficients are available to every fitted method. No target task-response fitting.'))
                del x, zs, zt
            if cfg.get('response_gain_fit_states'):
                # Fixed target directions fitted to the same individual code
                # responses, with exact live source amplitudes at execution.
                # This isolates changing target membership from source reading.
                fitted_states = np.load(work.checked(cfg['response_gain_fit_states']))['h']
                numerator = torch.zeros((target.dict_size, 4), device=device)
                denominator = torch.zeros(4, device=device)
                with torch.no_grad():
                    for xx in np.array_split(fitted_states, max(1, len(fitted_states)//128)):
                        hx = torch.tensor(xx, device=device)
                        sx = source_z(hx)
                        active = sx.any(-1)
                        hx, sx = hx[active], sx[active]
                        if len(hx) == 0:
                            continue
                        zx = target.encode(hx)
                        edited = hx.unsqueeze(-2)-sx.unsqueeze(-1)*s['decoder']
                        response = (target.encode(edited)-zx.unsqueeze(-2)).transpose(-1, -2)
                        response *= (sx != 0).unsqueeze(-2)
                        numerator += (response*sx.unsqueeze(-2)).sum(0)
                        denominator += sx.square().sum(0)
                    relations['response_gain'] = numerator/denominator.clamp_min(1e-20)
                np.savez_compressed(work.run/'response_gain.npz', weights=relations['response_gain'].cpu().numpy())
                write(work.run/'RESPONSE_GAIN_FIT.json', dict(states=len(fitted_states),
                    source_second_moments=denominator.cpu().tolist(),
                    fit='Columnwise least squares of finite target-code changes against exact source activation on original natural fit states; no output responses.'))
        logits = np.full((len(cfg['methods']), len(queries), len(token_rows)), np.nan)
        probability, argmax = np.zeros_like(logits), np.zeros_like(logits, dtype=np.int64)
        for mi, method in enumerate(cfg['methods']):
            for qi, (name, mask) in enumerate(queries.items()):
                q = torch.tensor(mask, device=device)
                for off in range(0, len(token_rows), 16):
                    rows = token_rows[off:off+16]
                    ids = torch.zeros((len(rows), max(map(len, rows))), dtype=torch.long, device=device)
                    attention = torch.zeros_like(ids)
                    for j, row in enumerate(rows):
                        ids[j, :len(row)] = torch.tensor(row, device=device)
                        attention[j, :len(row)] = 1
                    with torch.no_grad():
                        result = model(ids, attention_mask=attention, use_cache=False).logits
                        last = result[torch.arange(len(rows), device=device), attention.sum(1)-1]
                        logp = torch.log_softmax(last, dim=-1)[:, answer[0]]
                    logits[mi, qi, off:off+len(rows)] = logp.cpu().numpy()
                    probability[mi, qi, off:off+len(rows)] = logp.exp().cpu().numpy()
                    argmax[mi, qi, off:off+len(rows)] = last.argmax(-1).cpu().numpy()
                    work.sequence_forwards += len(rows)
                    work.token_forwards += ids.numel()
                    if time.perf_counter()-work.wall_start > cfg['budget_seconds']:
                        raise TimeoutError('Bounded published-parts experiment')
                work.record(method=method, operation=name, component=name, mode='next_token',
                            seed=cfg.get('target_seed'), split=panel['evidence_level'],
                            value=float(logits[mi, qi].mean()), n=len(token_rows))
                work.progress('QUERY', method=method, query=name)
        np.savez_compressed(work.run/'responses.npz', log_probability=logits, probability=probability, argmax=argmax)
        write(work.run/'INDEX.json', dict(methods=cfg['methods'], queries=list(queries), query_masks=queries, rows=panel['rows'],
             answer_id=answer[0], evidence_level=panel['evidence_level']))
        if write_records:
            write(work.run/'WRITING.json', dict(records=write_records,
                scope='All evaluated token positions, including padding; nonnegative final codes checked. Reencoding permits input-dependent members.'))
        work.checks.update(finite_outputs=bool(np.isfinite(logits).all()), source_before_output_fixed=True,
                           reconstruction_residual_preserved=True)
    except Exception:
        error = traceback.format_exc()
    finally:
        if handle is not None:
            handle.remove()
        result = work.finish(error)
    if error:
        raise RuntimeError(error)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
