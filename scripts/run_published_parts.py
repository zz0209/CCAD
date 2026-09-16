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
        observed, relations, target = [], {}, None
        response_state, fit_alpha = None, 0.
        functional_weight = cfg.get('functional_weight', 0.)
        queries = {'full': [1., 1., 1., 1.], 'predicate': [1., 1., 1., 0.], 'object': [0., 0., 0., 1.]}

        def source_z(h):
            return torch.relu((h-s['center']) @ s['encoder'].T+s['encoder_bias'])

        def hook(module, inputs, output):
            nonlocal response_state
            h = output[0] if isinstance(output, tuple) else output
            if method == 'capture':
                observed.append(h.detach().flatten(0, 1).cpu().numpy().copy())
                if functional_weight:
                    h = (h-fit_alpha*(source_z(h) @ s['decoder'])).detach().requires_grad_(True)
                    response_state = h
            elif method == 'source':
                h = h-(source_z(h)*q) @ s['decoder']
            elif method != 'none':
                z = target.encode(h)
                if method == 'raw':
                    h = h-(z[..., relations['candidates']] @ relations['raw']*q) @ s['decoder']
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
                natural = np.memmap(work.checked(cfg['paired_tokens']), dtype='<u2', mode='r').reshape(-1, 128)[:128]
                gradients = []
                for fit_alpha in ([0., 1.] if functional_weight else [0.]):
                    for off in range(0, len(natural), 8):
                        ids = torch.tensor(np.array(natural[off:off+8], dtype='int64'), device=device)
                        with torch.set_grad_enabled(bool(functional_weight)):
                            if functional_weight:
                                values = model(ids, use_cache=False).logits
                                score = torch.log_softmax(values, dim=-1)[..., answer[0]].sum()
                                gradient = torch.autograd.grad(score, response_state)[0]
                                gradients.append(gradient.detach().flatten(0, 1).cpu().numpy().copy())
                            else:
                                model.gpt_neox(ids, use_cache=False)
                        work.sequence_forwards += len(ids)
                        work.token_forwards += ids.numel()
                x = torch.tensor(np.concatenate(observed), device=device)
                observed.clear()
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
                np.savez_compressed(work.run/'relation.npz', **arrays)
                relations = {key: torch.tensor(value, device=device,
                    dtype=torch.long if key == 'candidates' else torch.float32) for key, value in arrays.items()}
                write(work.run/'FIT.json', dict(states=len(x), candidates=len(index), source_members=4,
                    target_allowance=8, target_row_sum=float(native.sum(1).max()), source_activation_rate=(zs>0).mean(0).tolist(),
                    functional_weight=functional_weight,
                    fitting='Natural clean/source-deleted states; existing blended field/source-response fit with source log-probability of to. All comparators share candidates and source information; no task examples or target response fitting.'))
                del x, zs, zt
        logits = np.full((len(cfg['methods']), 3, len(token_rows)), np.nan)
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
        write(work.run/'INDEX.json', dict(methods=cfg['methods'], queries=list(queries), rows=panel['rows'],
             answer_id=answer[0], evidence_level=panel['evidence_level']))
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
