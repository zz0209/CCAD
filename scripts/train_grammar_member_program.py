import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import numpy as np
import torch
import transformers

from run_causalgym_multisite import MultisiteWork, ROOT, write
from run_shift_transfer import input_member_delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--target-seed', type=int)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    if args.target_seed is not None:
        assert args.target_seed in c['target_seeds']
        c['target_seed'] = args.target_seed
        c['seeds'] = [1, args.target_seed]
        c['run_id'] = c['run_id_template'].format(seed=args.target_seed)
        c['target_checkpoint'] = c['target_checkpoint_template'].format(seed=args.target_seed)
        if str(args.target_seed) in c.get('resume_by_seed', {}):
            c['resume_checkpoints'] = c['resume_by_seed'][str(args.target_seed)]
    w = MultisiteWork(c, args.config, [
        'scripts/train_grammar_member_program.py', 'scripts/run_shift_transfer.py',
        'scripts/run_shift_explanation.py', 'scripts/train_shift_dictionaries.py',
        'scripts/run_causalgym_multisite.py', 'scripts/run_r011s1_raw_hook_asset.py',
        'src/ccad/artifacts.py', 'src/ccad/activation_contract.py'])
    handle, error = None, None
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        w.torch, w.device = torch, torch.device(c['device'])
        torch.cuda.set_device(w.device)
        torch.cuda.reset_peak_memory_stats()
        sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK

        w.checked(Path(c['dictionary_source_dir'])/'dictionary_learning/trainers/top_k.py',
                  'Pinned dictionary_learning TopK implementation', 'MIT')
        w.environment = dict(python=sys.executable, torch=torch.__version__, numpy=np.__version__,
                             transformers=transformers.__version__, threads=2,
                             model=c['model_revision'], hook=c['hook_module_path'])
        for filename in ['model.safetensors', 'config.json', 'tokenizer.json']:
            w.checked(Path(c['model_local_dir'])/filename)
        model = transformers.AutoModelForCausalLM.from_pretrained(c['model_local_dir'],
            local_files_only=True, dtype=torch.float32, attn_implementation='eager').eval().to(w.device)
        model.requires_grad_(False)
        model.config.use_cache = False
        tok = transformers.AutoTokenizer.from_pretrained(c['model_local_dir'], local_files_only=True)
        dim = model.config.hidden_size
        source_state = torch.load(w.checked(c['source_checkpoint']), map_location=w.device, weights_only=True)
        initial = torch.load(w.checked(c['target_checkpoint']), map_location=w.device, weights_only=True)
        source_ae = AutoEncoderTopK(dim, source_state['encoder.weight'].shape[0], int(source_state['k'])).to(w.device)
        source_ae.load_state_dict(source_state)
        source_ae.eval().requires_grad_(False)
        target = AutoEncoderTopK(dim, initial['encoder.weight'].shape[0], int(initial['k'])).to(w.device)
        target.load_state_dict(initial)
        parent = Path(c['source_run'])
        gate = torch.tensor(np.load(w.checked(parent/'topk_s1_source.npz'))['gate'], device=w.device)
        members = gate.sum(1).nonzero().flatten()
        part_ids = gate[members].argmax(1)
        assert len(members) == 192 and bool((gate.sum(1) <= 1).all())
        source = dict(sae=source_ae, member_ids=members,
                      decoder=source_ae.decoder.weight[:, members].T)
        old = np.load(w.checked(parent/f'topk_s1_t{c["target_seed"]}_map.npz')) if c.get('fixed_control') else None
        fixed_ids = torch.tensor(old['target_members'], device=w.device) if old is not None else None
        fixed_gate = torch.tensor(old['partition64'], device=w.device) if old is not None else None
        natural = torch.tensor(np.load(w.checked(c['natural_states']))['hidden'], device=w.device)
        natural_fit, natural_eval = natural[:-1024], natural[-1024:]
        panel = json.loads(w.checked(c['panel']).read_text())
        fit = [r for task in c['tasks'] for r in [r for r in panel['rows'] if r['task'] == task and r['split'] == 'fit'][:c['fit_pairs_per_task']]]
        evaluation = json.loads(w.checked(c['evaluation_panel']).read_text()) if c.get('evaluation_panel') else panel
        evaluation_split = c.get('evaluation_split', 'development')
        rows = [r for task in c['tasks'] for r in [r for r in evaluation['rows'] if r['task'] == task and r['split'] == evaluation_split][:c['eval_pairs_per_task']]]
        assert len(fit) == len(c['tasks'])*c['fit_pairs_per_task']
        assert len(rows) == len(c['tasks'])*c['eval_pairs_per_task']
        assert not {r[key] for r in fit for key in ['sentence_good', 'sentence_bad']} & {r[key] for r in rows for key in ['sentence_good', 'sentence_bad']}
        queries = c['queries']
        write(w.run/'panel.json', dict(fit=fit, rows=rows, queries=queries,
            query_order=list(queries), scope=c['scope']))
        mode, q = 'none', torch.ones(len(members), device=w.device)
        request = torch.ones(3, device=w.device)
        gain = torch.nn.Parameter(torch.ones(len(members), device=w.device))
        positions, cache = None, {}

        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            idx = torch.arange(len(h), device=w.device)
            x = h[idx, positions]
            cache['input'] = x.detach()
            if mode == 'none':
                return output
            if mode == 'source':
                delta = -(source_ae.encode(x)[:, members]*q)@source['decoder']
            elif mode == 'fixed':
                delta = -(target.encode(x)[:, fixed_ids]*(fixed_gate@request))@target.decoder.weight[:, fixed_ids].T
            else:
                sp = source
                if mode == 'gain':
                    sp = dict(source, transport_basis=-(target.encoder.weight@source['decoder'].T)*gain)
                operation = 'raw_reconstruction' if mode == 'readout' else 'input_tangent_budget'
                delta, counts = input_member_delta(x, target, sp, q, operation,
                    c['members_per_source']*len(members), torch.ones(len(x), device=w.device))
                if operation != 'raw_reconstruction':
                    assert counts['minimum_final_code'] >= -1e-5
            hh = h.clone()
            hh[idx, positions] = x+delta
            cache['delta'] = delta.detach()
            return (hh, *output[1:]) if isinstance(output, tuple) else hh

        handle = model.get_submodule(c['hook_module_path']).register_forward_hook(hook)

        def forward(rr):
            nonlocal positions
            length = max(len(r[key]) for r in rr for key in ['good', 'bad'])
            ids = torch.full((2*len(rr), length), tok.eos_token_id, device=w.device, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, row in enumerate(rr):
                for j, key in enumerate(['good', 'bad']):
                    tokens = row[key]
                    ids[2*i+j, :len(tokens)] = torch.tensor(tokens, device=w.device)
                    mask[2*i+j, :len(tokens)] = 1
            positions = torch.tensor([r['position'] for r in rr], device=w.device).repeat_interleave(2)
            hidden = model.transformer(ids, attention_mask=mask, use_cache=False).last_hidden_state
            logits = model.get_output_embeddings()(hidden[:, :-1])
            per_token = logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
            total = (per_token*mask[:, 1:]).double().sum(1).reshape(-1, 2)
            margins = total[:, 0]-total[:, 1]
            w.sequence_forwards += len(ids)
            w.token_forwards += int(mask.sum())
            assert float((cache['input'][::2]-cache['input'][1::2]).abs().max()) < .001
            if time.perf_counter()-w.wall_start > c['budget_seconds']:
                raise TimeoutError('Declared grammar program budget exceeded')
            return hidden, mask, margins

        def state_mse(a, b, mask):
            return ((a-b).square()*mask[:, :, None]).sum()/(mask.sum()*a.shape[-1])

        values, quality = {}, []

        @torch.no_grad()
        def evaluate(name, execution):
            nonlocal mode, q, request
            mode = execution
            result = np.empty((len(queries), len(rows)), dtype=np.float64)
            for qi, (query, vector) in enumerate(queries.items()):
                request = torch.tensor(vector, device=w.device, dtype=torch.float32)
                q = request[part_ids]
                for off in range(0, len(rows), c['eval_batch_pairs']):
                    rr = rows[off:off+c['eval_batch_pairs']]
                    _, _, margins = forward(rr)
                    result[qi, off:off+len(rr)] = margins.cpu().numpy()
                    for row, value in zip(rr, margins.cpu().tolist()):
                        w.record(kind='grammar_program', task=row['task'], row_id=row['row_id'],
                            component=row['task']+':'+str(row['row_id']), mode=query, operation=query,
                            method=name, seed=1, target_seed=c['target_seed'], split=evaluation_split,
                            margin=value, accuracy=value > 0)
                w.progress('EVALUATION', method=name, query=query)
            values[name] = result
            np.savez_compressed(w.run/'responses.npz', **values)
            z = target.encode(natural_eval)
            rec = target.decode(z)
            quality.append(dict(method=name, fve=float(1-(rec-natural_eval).square().sum()/
                (natural_eval-natural_eval.mean(0)).square().sum()), l0=float((z > 0).sum(1).float().mean())))
            write(w.run/'natural_quality.json', quality)

        evaluate('none', 'none')
        evaluate('source', 'source')
        evaluate('initial', 'tangent')
        evaluate('readout_initial', 'readout')
        if c.get('fixed_control'):
            evaluate('fixed', 'fixed')
        endpoints = torch.cat([torch.eye(3, device=w.device), torch.ones(1, 3, device=w.device)])
        energy = []
        calibration = [fit[int(i)] for i in np.linspace(0, len(fit)-1, min(24, len(fit)), dtype=int)]
        with torch.no_grad():
            for off in range(0, len(calibration), c['batch_pairs']):
                rr = calibration[off:off+c['batch_pairs']]
                mode = 'none'
                clean_h, mask, clean_m = forward(rr)
                for request in endpoints:
                    q, mode = request[part_ids], 'source'
                    source_h, _, source_m = forward(rr)
                    energy.append([float(state_mse(source_h, clean_h, mask)),
                                   float((source_m-clean_m).square().mean())])
        scales = np.maximum(np.mean(energy, axis=0), 1e-8)
        write(w.run/'loss_scales.json', dict(hidden=float(scales[0]), response=float(scales[1])))
        rng = np.random.default_rng(c['training_seed'])
        requests = rng.random((c['steps'], 3)).astype('float32')
        for step in range(0, c['steps'], 2):
            requests[step] = endpoints[(step//2) % 4].cpu().numpy()
        order = rng.permutation(len(fit))
        natural_indices = rng.integers(len(natural_fit), size=(c['steps'], c['natural_batch_states']))
        np.savez_compressed(w.run/'training_schedule.npz', requests=requests, rows=order, natural=natural_indices)
        for variant in c['variants']:
            target.load_state_dict(initial)
            target.requires_grad_(variant in ['program', 'whole'])
            gain.data.fill_(1.)
            gain.requires_grad_(variant == 'gain')
            parameters = [gain] if variant == 'gain' else list(target.parameters())
            optimizer = torch.optim.AdamW(parameters, lr=c['gain_lr'] if variant == 'gain' else c['dictionary_lr'], weight_decay=0.)
            first_step = 0
            if variant in c.get('resume_checkpoints', {}):
                saved = torch.load(w.checked(c['resume_checkpoints'][variant]), map_location=w.device, weights_only=True)
                target.load_state_dict(saved['dictionary'])
                gain.data.copy_(saved['gain'])
                optimizer.load_state_dict(saved['optimizer'])
                first_step = saved['step']
            for step in range(first_step, c['steps']):
                request = torch.tensor(requests[step], device=w.device)
                if variant == 'whole':
                    request = torch.ones_like(request)
                q = request[part_ids]
                rr = [fit[order[(step*c['batch_pairs']+j) % len(fit)]] for j in range(c['batch_pairs'])]
                mode = 'source'
                with torch.no_grad():
                    th, mask, tm = forward(rr)
                mode = 'gain' if variant == 'gain' else 'tangent'
                sh, _, sm = forward(rr)
                hidden_loss = state_mse(sh, th, mask)/scales[0]
                response_loss = (sm-tm).square().mean()/scales[1]
                natural_h = natural_fit[natural_indices[step]]
                reconstruction = (target(natural_h)-natural_h).square().mean()/natural_h.square().mean()
                loss = (1-c['response_weight'])*hidden_loss+c['response_weight']*response_loss+c['reconstruction_weight']*reconstruction
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
                with torch.no_grad():
                    if variant in ['program', 'whole']:
                        target.decoder.weight.div_(target.decoder.weight.norm(dim=0).clamp_min(1e-10))
                    gain.clamp_(min=0)
                assert bool(torch.isfinite(loss))
                if (step+1) % c['checkpoint_every'] == 0 or step+1 == c['steps']:
                    torch.save(dict(dictionary=target.state_dict(), gain=gain.detach(),
                        optimizer=optimizer.state_dict(), step=step+1), w.run/f'{variant}_step{step+1}.pt')
                    w.progress('TRAINING', method=variant, step=step+1, loss=float(loss.detach()),
                               response=float(response_loss.detach()), reconstruction=float(reconstruction.detach()),
                               peak_cuda_bytes=torch.cuda.max_memory_allocated())
            evaluate(variant, 'gain' if variant == 'gain' else 'tangent')
            if variant == 'program':
                evaluate('readout_program', 'readout')
        w.checks['source_frozen'] = all(torch.equal(source_ae.state_dict()[key], value) for key, value in source_state.items())
        w.checks['base_model_frozen'] = all(not p.requires_grad for p in model.parameters())
        write(w.run/'method_summary.json', dict(quality=quality, source_members=members.cpu().tolist(),
            source_parts=part_ids.cpu().tolist(), same_rule='run_shift_transfer.input_member_delta'))
    except Exception:
        error = traceback.format_exc()
    finally:
        if handle is not None:
            handle.remove()
    return w.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
