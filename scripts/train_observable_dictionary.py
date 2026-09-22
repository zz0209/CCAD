import argparse
import copy
import json
from pathlib import Path
import sys
import time
import traceback

from run_causalgym_multisite import MultisiteWork, ROOT, write

import numpy as np
import torch
import transformers


def adjoint_reconstruction(ae, h, g):
    z = ae.encode(h)
    active = (z > 0).to(h.dtype)
    reconstructed_gradient = ((g @ ae.decoder.weight) * active) @ ae.encoder.weight
    return ae.decode(z), reconstructed_gradient, z


def dictionary_losses(ae, h, g, gradient_energy=None):
    reconstructed, projected_gradient, z = adjoint_reconstruction(ae, h, g)
    rec = (reconstructed-h).square().sum()/h.square().sum().clamp_min(1e-12)
    denominator = g.square().sum().clamp_min(1e-12) if gradient_energy is None else len(g)*gradient_energy
    adjoint = (projected_gradient-g).square().sum()/denominator
    return rec, adjoint, z


def capture_scores(model, sites, ids, seed):
    observed, order, handles = {}, [], []

    def hook(name):
        def apply(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if name in observed:
                raise ValueError('A capture site was executed more than once')
            if not observed:
                h.requires_grad_(True)
            if not h.requires_grad:
                raise RuntimeError('A later capture site lost the model computation graph')
            h.retain_grad()
            observed[name] = h
            order.append(name)
            return output
        return apply

    try:
        for name, item in sites.items():
            handles.append(model.get_submodule(item['module']).register_forward_hook(hook(name)))
        logits = model(ids, use_cache=False).logits
        generator = torch.Generator(device=ids.device).manual_seed(seed)
        lp = logits.log_softmax(-1)
        sampled = torch.multinomial(lp.detach().exp().reshape(-1, lp.shape[-1]), 1,
                                    generator=generator).reshape(ids.shape)
        score = lp.gather(-1, sampled.unsqueeze(-1)).sum()
        score.backward()
        if set(observed) != set(sites):
            raise RuntimeError('Not all configured model sites executed')
        result = {}
        for name, h in observed.items():
            if h.grad is None or not bool(torch.isfinite(h.grad).all()):
                raise FloatingPointError('Invalid model score gradient at '+name)
            if not bool(torch.isfinite(h).all()) or float(h.grad.square().sum()) <= 0:
                raise FloatingPointError('Invalid activation or zero gradient energy at '+name)
            # 仅在完整模型 backward 完成以后保存独立缓存。
            result[name] = dict(h=h.detach().reshape(-1, h.shape[-1]).cpu(),
                                g=h.grad.detach().reshape(-1, h.shape[-1]).cpu())
        return result, sampled.detach().cpu(), order
    finally:
        for handle in handles:
            handle.remove()


def numerical_checks(ae_class):
    torch.manual_seed(1701)
    ae = ae_class(8, 13, 3).double()
    h = torch.randn(7, 8, dtype=torch.float64, requires_grad=True)
    g = torch.randn_like(h)
    y, analytic, _ = adjoint_reconstruction(ae, h, g)
    automatic = torch.autograd.grad(y, h, g, retain_graph=True)[0]
    error = float((automatic-analytic).abs().max().detach())
    torch.testing.assert_close(automatic, analytic, atol=1e-11, rtol=1e-10)
    initial = copy.deepcopy(ae.state_dict())
    rec, adjoint, _ = dictionary_losses(ae, h.detach(), g)
    (rec+adjoint).backward()
    for name, parameter in ae.named_parameters():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError('Numerical check missing finite gradient for '+name)
    optimizer = torch.optim.AdamW(ae.parameters(), lr=1e-4, weight_decay=0.)
    optimizer.step()
    for name in ['encoder.weight', 'encoder.bias', 'decoder.weight', 'b_dec']:
        assert not torch.equal(initial[name], ae.state_dict()[name]), name
    saved_ae, saved_optimizer = copy.deepcopy(ae.state_dict()), copy.deepcopy(optimizer.state_dict())
    rec, adjoint, _ = dictionary_losses(ae, h.detach(), g)
    optimizer.zero_grad(set_to_none=True)
    (rec+adjoint).backward()
    optimizer.step()
    continued = copy.deepcopy(ae.state_dict())
    ae.load_state_dict(saved_ae)
    optimizer.load_state_dict(saved_optimizer)
    rec, adjoint, _ = dictionary_losses(ae, h.detach(), g)
    optimizer.zero_grad(set_to_none=True)
    (rec+adjoint).backward()
    optimizer.step()
    assert all(torch.equal(ae.state_dict()[key], value) for key, value in continued.items())

    # 使用真实 GPT2 attention 路径核验未来位置对早期 hidden state 的梯度。
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=31,
        n_positions=8, n_embd=8, n_layer=2, n_head=2, bos_token_id=1, eos_token_id=2,
        resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.))
    model.eval().requires_grad_(False)
    ids = torch.tensor([[1, 7, 3, 9, 2]])
    sites = dict(early=dict(module='transformer.h.0'), late=dict(module='transformer.h.1'))
    first, samples, order = capture_scores(model, sites, ids, 53)
    repeat, repeated_samples, _ = capture_scores(model, sites, ids, 53)
    assert torch.equal(samples, repeated_samples)
    assert all(torch.equal(first[s]['g'], repeat[s]['g']) for s in sites)
    captured = {}

    def early_hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h.requires_grad_(True)
        captured['h'] = h
        return output

    handle = model.transformer.h[0].register_forward_hook(early_hook)
    try:
        lp = model(ids, use_cache=False).logits.log_softmax(-1)
        future_gradient = torch.autograd.grad(lp[0, -1, 4], captured['h'])[0]
    finally:
        handle.remove()
    future_norm = float(future_gradient[0, 0].norm())
    assert future_norm > 0 and order == ['early', 'late']
    return dict(adjoint_matches_autograd=True, maximum_adjoint_error=error,
        encoder_decoder_bias_gradients_finite=True, optimizer_resume_exact=True,
        score_sampling_repeatable=True, future_token_gradient_preserved=True,
        future_to_first_state_gradient_norm=future_norm)


def check_budget(work):
    if time.perf_counter()-work.wall_start > work.cfg['budget_seconds']:
        raise TimeoutError('Observable dictionary allocated time reached; saved outputs remain reusable')


def save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix('.partial')
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--cache-run', type=Path)
    parser.add_argument('--resume-run', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    if c.get('base_config'):
        c = json.loads(Path(c['base_config']).read_text()) | c
    defaults = dict(steps=512, lr=1e-4, batch_states=256, gradient_weight=1., seed=0,
        capture_batch=1, checkpoint_every=64, log_every=16, cpu_threads=2, device='cuda:0',
        fit_offset=0, eval_offset=0,
        tokens_file=str(ROOT/'artifacts/final_three_research_20260909/paired_material/discovery.uint16.bin'),
        eval_tokens_file=str(ROOT/'artifacts/final_three_research_20260909/paired_material/calibration.uint16.bin'),
        dictionary_source_dir='D:/CCAD_Storage/references/source/dictionary_learning_60ec6bf',
        dictionary_overlay_dir='D:/CCAD_Storage/environments/r005a_dictionary_overlay')
    c = defaults | c
    if args.smoke:
        c.update(run_id=c['run_id']+'_SMOKE', fit_sequences=8, eval_sequences=2,
                 steps=8, checkpoint_every=8, log_every=2)
    sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
    from dictionary_learning.trainers.top_k import AutoEncoderTopK
    torch.set_num_threads(c['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    if args.check_only:
        print(json.dumps(numerical_checks(AutoEncoderTopK)), flush=True)
        return 0
    if min(c['fit_sequences'], c['eval_sequences'], c['steps'], c['batch_states'], c['capture_batch']) <= 0:
        raise ValueError('Counts and training steps must be positive')
    if Path(c['tokens_file']).resolve() == Path(c['eval_tokens_file']).resolve():
        raise ValueError('Fit and evaluation must use their separate corpus split files')
    if args.cache_run and args.resume_run:
        raise ValueError('Use one cache or training resume source')
    c['generator_script'] = 'scripts/train_observable_dictionary.py'
    c['cache_run'] = str(args.cache_run) if args.cache_run else None
    c['resume_run'] = str(args.resume_run) if args.resume_run else None
    files = ['scripts/train_observable_dictionary.py', 'scripts/run_causalgym_multisite.py',
             'scripts/run_causalgym_native_transfer.py', 'scripts/run_r011s1_raw_hook_asset.py',
             'src/ccad/artifacts.py', 'src/ccad/activation_contract.py']
    w = MultisiteWork(c, args.config, files)
    error = None
    try:
        w.torch, w.device = torch, torch.device(c['device'])
        if w.device.type == 'cuda':
            torch.cuda.set_device(w.device)
            torch.cuda.reset_peak_memory_stats(w.device)
        w.environment = dict(python=sys.executable, torch=str(torch.__version__), numpy=np.__version__,
            transformers=transformers.__version__, device=str(w.device), threads=torch.get_num_threads())
        w.checked(Path(c['dictionary_source_dir'])/'dictionary_learning/trainers/top_k.py',
                  'Pinned dictionary_learning TopK implementation', 'MIT')
        checks = numerical_checks(AutoEncoderTopK)
        write(w.run/'numerical_checks.json', checks)
        w.checks.update({key: value for key, value in checks.items() if isinstance(value, bool)})
        w.checked(ROOT/'.aris/compute/local-r006b1-env-spec.json')
        for filename in ['config.json', 'model.safetensors']:
            w.checked(Path(c['model_local_dir'])/filename)
        if c.get('token_manifest'):
            w.checked(c['token_manifest'], 'Original document split provenance')
        token_files = {split: w.checked(c[key], split+' natural corpus tokens') for split, key in
                       [('fit', 'tokens_file'), ('eval', 'eval_tokens_file')]}
        input_hashes = {row['path']: row['sha256'] for row in w.inputs}
        identity = dict(model_local_dir=str(Path(c['model_local_dir']).resolve()),
            model_revision=c['model_revision'], modules={s: v['module'] for s, v in c['sites'].items()},
            model_hashes={name: input_hashes[str((Path(c['model_local_dir'])/name).resolve())]
                          for name in ['config.json', 'model.safetensors']},
            tokens={split: dict(path=str(path.resolve()), bytes=path.stat().st_size,
                sha256=input_hashes[str(path.resolve())]) for split, path in token_files.items()},
            counts={s: c[s+'_sequences'] for s in token_files},
            offsets={s: c[s+'_offset'] for s in token_files}, seed=c['seed'], capture_batch=c['capture_batch'])
        cache = dict(identity=identity, batches=[], complete=False)
        parent = args.resume_run or args.cache_run
        if parent:
            cache = json.loads(w.checked(parent/'cache_index.json').read_text())
            if cache['identity'] != identity:
                raise ValueError('Saved cache does not match model, sites, split membership or sampling')
        write(w.run/'cache_index.json', cache)
        write(w.run/'membership.json', dict(identity=identity, split_basis='Original discovery/calibration token files',
            token_manifest=c.get('token_manifest'), independent_documents_verified_here=False,
            sampling='One model-distribution token at each output position; one summed score backward per batch'))
        completed = {(r['split'], r['start']): r for r in cache['batches']}
        if not cache['complete']:
            model = transformers.AutoModelForCausalLM.from_pretrained(c['model_local_dir'],
                local_files_only=True, trust_remote_code=False, dtype=torch.float32,
                attn_implementation='eager').to(w.device).eval().requires_grad_(False)
            model.config.use_cache = False
            for split_index, split in enumerate(['fit', 'eval']):
                tokens = np.memmap(token_files[split], dtype='<u2', mode='r')
                if tokens.size % 128:
                    raise ValueError('Token file length is not a whole 128-token sequence')
                tokens = tokens.reshape(-1, 128)
                count, offset = c[split+'_sequences'], c[split+'_offset']
                if offset < 0 or offset+count > len(tokens):
                    raise ValueError('Requested token membership exceeds the split file')
                for start in range(0, count, c['capture_batch']):
                    check_budget(w)
                    stop = min(start+c['capture_batch'], count)
                    if (split, start) in completed:
                        if completed[split, start]['stop'] != stop:
                            raise ValueError('Saved batch boundary differs')
                        continue
                    ids = torch.from_numpy(np.array(tokens[offset+start:offset+stop], dtype=np.int64)).to(w.device)
                    batch_seed = c['seed']+split_index*1000000007+start
                    states, samples, order = capture_scores(model, c['sites'], ids, batch_seed)
                    path = w.run/'cache'/f'{split}_{start:07d}.pt'
                    save_checkpoint(path, dict(states=states, sampled_tokens=samples, split=split,
                        start=start, stop=stop, batch_seed=batch_seed, site_order=order))
                    row = dict(split=split, start=start, stop=stop, path=str(path.resolve()), seed=batch_seed)
                    cache['batches'].append(row)
                    completed[split, start] = row
                    write(w.run/'cache_index.json', cache)
                    w.sequence_forwards += len(ids)
                    w.token_forwards += ids.numel()
                    w.progress('CAPTURE', split=split, sequences=stop, total=count, site_order=order)
                    del states, ids
                del tokens
            w.checks['base_model_frozen'] = all(not p.requires_grad for p in model.parameters())
            del model
            if w.device.type == 'cuda':
                torch.cuda.empty_cache()
            cache['complete'] = True
            write(w.run/'cache_index.json', cache)
        w.checks['cache_complete'] = cache['complete']
        training_identity = dict(cache=identity, steps=c['steps'], lr=c['lr'], batch_states=c['batch_states'],
                                 gradient_weight=c['gradient_weight'], seed=c['seed'])
        checkpoint_index, quality = [], []
        prior_checkpoints = []
        if args.resume_run and (args.resume_run/'checkpoints.json').exists():
            prior_checkpoints = json.loads(w.checked(args.resume_run/'checkpoints.json').read_text())
        for site, site_config in c['sites'].items():
            initial_path = w.checked(site_config['checkpoint'])
            initial = torch.load(initial_path, map_location='cpu', weights_only=True)
            dim = initial['encoder.weight'].shape[1]
            data = {}
            for split in ['fit', 'eval']:
                chunks = [torch.load(r['path'], map_location='cpu', weights_only=True)['states'][site]
                          for r in sorted(cache['batches'], key=lambda r: r['start']) if r['split'] == split]
                data[split] = {k: torch.cat([r[k] for r in chunks]) for k in ['h', 'g']}
                assert data[split]['h'].shape == data[split]['g'].shape == (c[split+'_sequences']*128, dim)
                assert all(bool(torch.isfinite(v).all()) for v in data[split].values())
                del chunks
            gradient_energy = float(data['fit']['g'].double().square().sum()/len(data['fit']['g']))
            if gradient_energy <= 0:
                raise ValueError('Fit score gradients have zero energy')
            ae = AutoEncoderTopK(dim, len(initial['encoder.weight']), int(initial['k'])).to(w.device)

            def evaluate(arm):
                totals = dict(rec=0., energy=0., gradient_error=0., gradient_energy=0., l0=0)
                alive = torch.zeros(len(initial['encoder.weight']), dtype=torch.bool, device=w.device)
                with torch.no_grad():
                    for start in range(0, len(data['eval']['h']), c['batch_states']):
                        h, g = [data['eval'][k][start:start+c['batch_states']].to(w.device) for k in ['h', 'g']]
                        reconstructed, projected, z = adjoint_reconstruction(ae, h, g)
                        for key, value in [('rec', (reconstructed-h).square().sum()), ('energy', h.square().sum()),
                            ('gradient_error', (projected-g).square().sum()), ('gradient_energy', g.square().sum())]:
                            totals[key] += float(value)
                        totals['l0'] += int((z > 0).sum())
                        alive |= (z > 0).any(0)
                variance = float((data['eval']['h']-data['eval']['h'].mean(0)).square().sum())
                if min(totals['energy'], totals['gradient_energy'], variance) <= 0:
                    raise ValueError('Evaluation normalization has zero energy')
                row = dict(kind='quality', task=site, component='natural', row_id=0, method=arm,
                    reconstruction_error=totals['rec']/totals['energy'], fve=1-totals['rec']/variance,
                    adjoint_error=totals['gradient_error']/totals['gradient_energy'],
                    states=len(data['eval']['h']), l0=totals['l0']/len(data['eval']['h']), alive=int(alive.sum()),
                    decoder_norm_error=float((ae.decoder.weight.norm(dim=0)-1).abs().max().detach()))
                quality.append(row)
                w.record(**row)
                write(w.run/'quality.json', quality)

            ae.load_state_dict(initial)
            evaluate('initial')
            for arm in ['rec', 'adjoint']:
                ae.load_state_dict(initial)
                ae.train().requires_grad_(True)
                optimizer = torch.optim.AdamW(ae.parameters(), lr=c['lr'], weight_decay=0.)
                start_step = 0
                inherited = [r for r in prior_checkpoints if r['site'] == site and r['arm'] == arm]
                if inherited:
                    prior = max(inherited, key=lambda r: r['step'])
                    saved = torch.load(w.checked(prior['path']), map_location=w.device, weights_only=True)
                    if saved['training_identity'] != training_identity or saved['initial_checkpoint'] != str(initial_path.resolve()):
                        raise ValueError('Training resume configuration or initial dictionary differs')
                    ae.load_state_dict(saved['dictionary'])
                    optimizer.load_state_dict(saved['optimizer'])
                    start_step = saved['step']
                    checkpoint_index.append(prior)
                for step in range(start_step, c['steps']):
                    check_budget(w)
                    generator = torch.Generator().manual_seed(c['seed']+step)
                    indices = torch.randint(len(data['fit']['h']), (c['batch_states'],), generator=generator)
                    h, g = [data['fit'][key][indices].to(w.device) for key in ['h', 'g']]
                    rec, adjoint, _ = dictionary_losses(ae, h, g, gradient_energy)
                    loss = rec+(c['gradient_weight']*adjoint if arm == 'adjoint' else 0.)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Nonfinite dictionary training loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    for name, parameter in ae.named_parameters():
                        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                            raise FloatingPointError('Invalid gradient for '+name)
                    optimizer.step()
                    with torch.no_grad():
                        ae.decoder.weight.div_(ae.decoder.weight.norm(dim=0).clamp_min(1e-12))
                    if (step+1) % c['log_every'] == 0 or step+1 == c['steps']:
                        w.record(kind='training', task=site, component='natural', row_id=step+1, method=arm,
                            loss=float(loss.detach()), reconstruction=float(rec.detach()), adjoint=float(adjoint.detach()))
                        w.progress('TRAIN', site=site, arm=arm, step=step+1, total=c['steps'], loss=float(loss.detach()))
                    if (step+1) % c['checkpoint_every'] == 0 or step+1 == c['steps']:
                        path = w.run/arm/f'{site}_step{step+1}.pt'
                        save_checkpoint(path, dict(dictionary=ae.state_dict(), optimizer=optimizer.state_dict(),
                            step=step+1, arm=arm, site=site, training_identity=training_identity,
                            fit_gradient_mean_squared_norm=gradient_energy,
                            initial_checkpoint=str(initial_path.resolve())))
                        checkpoint_index.append(dict(arm=arm, site=site, step=step+1, path=str(path.resolve())))
                        write(w.run/'checkpoints.json', checkpoint_index)
                evaluate(arm)
                w.checks[site+'_'+arm+'_encoder_updated'] = not torch.equal(
                    ae.encoder.weight.detach().cpu(), initial['encoder.weight'])
                w.checks[site+'_'+arm+'_decoder_updated'] = not torch.equal(
                    ae.decoder.weight.detach().cpu(), initial['decoder.weight'])
                del optimizer
            del data, ae
        w.checks['all_sites_evaluated'] = len(quality) == 3*len(c['sites'])
        w.checks['same_training_sample_order'] = True
        write(w.run/'checkpoints.json', checkpoint_index)
        write(w.run/'method_summary.json', dict(training_identity=training_identity, checkpoints=checkpoint_index,
            sites=c['sites'], query_supervision='Natural model-distribution score gradients only',
            evaluation='Held-out natural reconstruction and adjoint fidelity; intervention evaluation is separate'))
    except Exception:
        error = traceback.format_exc()
    return w.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
