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

from run_causalgym_multisite import MultisiteWork, write
from run_shift_transfer import input_member_delta
from train_grammar_member_program import balanced_source_schedule
from ccad.shared_action_encoder import SharedActionEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--source-columns', action='store_true')
    args = parser.parse_args()
    overlay = json.loads(args.config.read_text())
    c = json.loads(Path(overlay['base_config']).read_text()) | overlay
    evaluation_only = c.get('evaluation_only', False)
    assert isinstance(evaluation_only, bool)
    assert not evaluation_only or (not args.source_columns and args.resume is None)
    c['training_method'] = 'source_columns' if args.source_columns else 'shared'
    if args.source_columns:
        c['run_id'] += '_SOURCE_COLUMNS'
    if args.smoke:
        c.update(run_id=c['run_id']+'_SMOKE', steps=8, checkpoint_every=8,
                 fit_pairs_per_task=8, eval_pairs_per_task=2, budget_seconds=240)
    if args.resume:
        c['run_id'] += '_RESUME'
        c['resume'] = args.resume.as_posix()
    source_seed = c.get('evaluation_source_seed', 1) if evaluation_only else 1
    c['source_seed'] = source_seed
    if evaluation_only:
        c['steps'] = 0
        c['training_method'] = 'evaluation_only'
    w = MultisiteWork(c, args.config, ['scripts/train_shared_action_rule.py',
        'scripts/train_grammar_member_program.py', 'scripts/run_shift_transfer.py',
        'scripts/run_causalgym_multisite.py', 'scripts/run_r011s1_raw_hook_asset.py',
        'src/ccad/shared_action_encoder.py', 'src/ccad/intervention_transport.py', 'src/ccad/artifacts.py'])
    error, handle = None, None
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        w.torch, w.device = torch, torch.device(c['device'])
        torch.cuda.set_device(w.device)
        torch.cuda.reset_peak_memory_stats()
        sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        w.checked(c['base_config'])
        w.checked(Path(c['dictionary_source_dir'])/'dictionary_learning/trainers/top_k.py', 'Pinned TopK', 'MIT')
        w.environment = dict(python=sys.executable, torch=torch.__version__, numpy=np.__version__,
                             transformers=transformers.__version__, threads=2)
        for filename in ['model.safetensors', 'config.json', 'tokenizer.json']:
            w.checked(Path(c['model_local_dir'])/filename)
        model = transformers.AutoModelForCausalLM.from_pretrained(c['model_local_dir'], local_files_only=True,
            dtype=torch.float32, attn_implementation='eager').eval().to(w.device).requires_grad_(False)
        model.config.use_cache = False
        tok = transformers.AutoTokenizer.from_pretrained(c['model_local_dir'], local_files_only=True)
        dim = model.config.hidden_size
        states, dictionaries = {}, {}
        for seed in dict.fromkeys([1, source_seed, *c['target_seeds']]):
            state = torch.load(w.checked(c['dictionary_template'].format(seed=seed)), map_location=w.device, weights_only=True)
            ae = AutoEncoderTopK(dim, len(state['encoder.weight']), int(state['k'])).to(w.device)
            ae.load_state_dict(state)
            ae.eval().requires_grad_(False)
            states[seed], dictionaries[seed] = state, ae
        source_ae = dictionaries[source_seed]
        gate = torch.tensor(np.load(w.checked(Path(c['source_run'])/f'topk_s{source_seed}_source.npz'))['gate'], device=w.device)
        members = gate.sum(1).nonzero().flatten()
        parts = gate[members].argmax(1)
        assert len(members) == 192 and bool((gate.sum(1) <= 1).all())
        source = dict(sae=source_ae, member_ids=members, part_ids=parts,
                      decoder=source_ae.decoder.weight[:, members].T)
        panel = json.loads(w.checked(c['panel']).read_text())
        fit = [r for task in c['tasks'] for r in [v for v in panel['rows']
            if v['task'] == task and v['split'] == 'fit'][:c['fit_pairs_per_task']]]
        evaluation = json.loads(w.checked(c['evaluation_panel']).read_text()) if c.get('evaluation_panel') else panel
        evaluation_split = c.get('evaluation_split', 'development')
        rows = [r for task in c['tasks'] for r in [v for v in evaluation['rows']
            if v['task'] == task and v['split'] == evaluation_split][:c['eval_pairs_per_task']]]
        assert len(fit) == 3*c['fit_pairs_per_task'] and len(rows) == 3*c['eval_pairs_per_task']
        assert not {r[k] for r in fit for k in ['sentence_good', 'sentence_bad']} & {
            r[k] for r in rows for k in ['sentence_good', 'sentence_bad']}
        write(w.run/'panel.json', dict(fit=fit, rows=rows, queries=c['queries'], query_order=list(c['queries'])))
        action = SharedActionEncoder(dim, c['recurrence_steps']).to(w.device)
        analytic = SharedActionEncoder(dim, c['recurrence_steps']).to(w.device).requires_grad_(False)
        learned_columns = torch.nn.Parameter(source['decoder'].clone(), requires_grad=args.source_columns)
        transferred_columns, transfer_identity = {}, {}
        initial_action = None
        if evaluation_only:
            saved = torch.load(w.checked(c['shared_checkpoint']), map_location='cpu', weights_only=True)
            assert saved['training_method'] == 'shared' and saved['recurrence_steps'] == c['recurrence_steps']
            assert saved['training_targets'] == c['training_targets']
            action.load_state_dict(saved['action'])
            action.requires_grad_(False)
            initial_action = action.weight.detach().clone()
            old_gate = torch.tensor(np.load(w.checked(Path(c['source_run'])/'topk_s1_source.npz'))['gate'])
            old_members = old_gate.sum(1).nonzero().flatten()
            old_decoder = dictionaries[1].decoder.weight[:, old_members].T.detach().cpu().double()
            new_decoder = source['decoder'].detach().cpu().double()
            cutoff = max(old_decoder.shape)*torch.finfo(torch.float32).eps
            inverse = torch.linalg.pinv(old_decoder, atol=0., rtol=cutoff)
            mapping = new_decoder @ inverse
            for name, path in c['source_column_checkpoints'].items():
                assert name.startswith('source_columns_')
                checkpoint = torch.load(w.checked(path), map_location='cpu', weights_only=True)
                assert checkpoint['training_method'] == 'source_columns'
                assert checkpoint['training_targets'] == saved['training_targets']
                assert checkpoint['step'] == saved['step']
                for key in ('requests', 'row_batches', 'targets'):
                    assert torch.equal(checkpoint[key], saved[key]), key
                correction = checkpoint['source_columns'].double() - old_decoder
                extended = new_decoder + mapping @ correction
                assert torch.isfinite(extended).all()
                transferred_columns[name] = extended.to(device=w.device, dtype=source['decoder'].dtype)
                transfer_identity[name] = dict(checkpoint=path, original_training_source_seed=1,
                    training_targets=checkpoint['training_targets'], original_updates=checkpoint['step'],
                    mapping_relative_cutoff=cutoff,
                    old_source_correction_replay_relative_error=float((old_decoder @ inverse @ correction-correction).norm()/correction.norm().clamp_min(1e-30)),
                    new_column_correction_norm=float((extended-new_decoder).norm()))
            write(w.run/'source_transfer.json', dict(evaluation_source_seed=source_seed,
                source_checkpoint=c['dictionary_template'].format(seed=source_seed),
                source_gate=str(Path(c['source_run'])/f'topk_s{source_seed}_source.npz'),
                source_members=members.cpu().tolist(), source_parts=parts.cpu().tolist(),
                training_source_checkpoint=c['dictionary_template'].format(seed=1),
                training_source_gate=str(Path(c['source_run'])/'topk_s1_source.npz'),
                training_source_members=old_members.tolist(), shared_checkpoint=c['shared_checkpoint'],
                shared_original_updates=saved['step'], column_methods=transfer_identity,
                mapping='U_s = D_s + (D_s pinv(D_1)) (U_1-D_1)',
                mapping_information='Source decoder geometry and saved learned columns; no new target responses',
                evaluation_training_steps=0))
        target = dictionaries[c['target_seeds'][0]]
        mode, q, positions = 'none', torch.ones(len(members), device=w.device), None
        counts = []
        allowance = c['members_per_source']*len(members)

        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if mode == 'none':
                return output
            ix = torch.arange(len(h), device=w.device)
            x = h[ix, positions]
            if mode == 'source':
                delta = -(source_ae.encode(x)[:, members]*q)@source['decoder']
            elif mode == 'source_columns' or mode in transferred_columns:
                translated = dict(source, decoder=transferred_columns[mode] if mode in transferred_columns else learned_columns)
                delta, checks = input_member_delta(x, target, translated, q, 'input_tangent_budget', allowance,
                                                   torch.ones(len(x), device=w.device))
                assert checks['minimum_final_code'] >= -1e-5
                counts.append(checks)
            elif mode in ('shared', 'analytic'):
                z = target.encode(x)
                zs = source_ae.encode(x)[:, members]
                basis = -(target.encoder.weight@source['decoder'].T)
                score = (zs@basis.abs().T)*(z > 0)*target.decoder.weight.norm(dim=0)
                chosen = torch.zeros_like(score).scatter(1, score.topk(allowance, dim=-1).indices, 1)*(z > 0)
                ids = chosen.topk(min(allowance, int(target.k)), dim=-1).indices
                active = chosen.gather(1, ids)
                decoder = target.decoder.weight.T[ids].transpose(-1, -2)*active[:, None, :]
                capacity = z.gather(1, ids)*active
                initial = basis[ids]*zs[:, None, :]*active[..., None]
                negative = (-initial).clamp_min(0)
                scale = (capacity/negative.sum(-1).clamp_min(1e-20)).clamp_max(1)
                initial = initial.clamp_min(0)-negative*scale[..., None]
                fields = -source['decoder'].T[None]*zs[:, None, :]
                solver = action if mode == 'shared' else analytic
                delta, checks = solver(fields, capacity, decoder, initial, q)
                assert checks['minimum_final_code'] >= -1e-5
                assert checks['minimum_capacity_margin'] >= -1e-5
                assert checks['max_changed_per_state'] <= int(target.k)
                counts.append(checks)
            else:
                operation = 'raw_reconstruction' if mode == 'readout' else 'input_tangent_budget'
                delta, _ = input_member_delta(x, target, source, q, operation, allowance,
                                               torch.ones(len(x), device=w.device))
            result = h.clone()
            result[ix, positions] = x+delta
            return (result, *output[1:]) if isinstance(output, tuple) else result

        handle = model.get_submodule(c['hook_module_path']).register_forward_hook(hook)

        def forward(batch):
            nonlocal positions
            width = max(len(row[key]) for row in batch for key in ['good', 'bad'])
            ids = torch.full((2*len(batch), width), tok.eos_token_id, device=w.device, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, row in enumerate(batch):
                for j, key in enumerate(['good', 'bad']):
                    ids[2*i+j, :len(row[key])] = torch.tensor(row[key], device=w.device)
                    mask[2*i+j, :len(row[key])] = 1
            positions = torch.tensor([row['position'] for row in batch], device=w.device).repeat_interleave(2)
            hidden = model.transformer(ids, attention_mask=mask, use_cache=False).last_hidden_state
            logits = model.get_output_embeddings()(hidden[:, :-1])
            lp = logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
            total = (lp*mask[:, 1:]).double().sum(1).reshape(-1, 2)
            w.sequence_forwards += len(ids)
            w.token_forwards += int(mask.sum())
            if time.perf_counter()-w.wall_start > c['budget_seconds']:
                raise TimeoutError('Shared action rule compute budget exceeded')
            return hidden, mask, total[:, 0]-total[:, 1]

        def state_mse(a, b, mask):
            return ((a-b).square()*mask[..., None]).sum()/(mask.sum()*a.shape[-1])

        endpoints = torch.cat([torch.eye(3), torch.ones(1, 3)]).to(w.device)
        calibration = [] if evaluation_only else [fit[int(i)] for i in np.linspace(0, len(fit)-1, min(24, len(fit)), dtype=int)]
        energy = []
        with torch.no_grad():
            for off in range(0, len(calibration), c['batch_pairs']):
                batch = calibration[off:off+c['batch_pairs']]
                mode = 'none'
                clean_h, mask, clean_m = forward(batch)
                for request in endpoints:
                    q, mode = request[parts], 'source'
                    source_h, _, source_m = forward(batch)
                    energy.append([float(state_mse(source_h, clean_h, mask)),
                                   float((source_m-clean_m).square().mean())])
        if not evaluation_only:
            scales = np.maximum(np.mean(energy, axis=0), 1e-8)
            write(w.run/'loss_scales.json', dict(hidden=float(scales[0]), response=float(scales[1])))
        rng = np.random.default_rng(c['training_seed'])
        requests = rng.random((c['steps'], 3)).astype('float32')
        for step in range(0, c['steps'], 2):
            requests[step] = endpoints[(step//2) % 4].cpu().numpy()
        order = rng.permutation(len(fit))
        batches = np.array([[order[(step*c['batch_pairs']+j) % len(fit)]
            for j in range(c['batch_pairs'])] for step in range(c['steps'])], dtype=np.int64).reshape(c['steps'], c['batch_pairs'])
        targets = balanced_source_schedule(c['training_targets'], c['steps'], c['training_seed'])
        np.savez_compressed(w.run/'training_schedule.npz', requests=requests, row_batches=batches, targets=targets)
        parameters = [] if evaluation_only else [learned_columns] if args.source_columns else list(action.parameters())
        action.requires_grad_(not args.source_columns and not evaluation_only)
        optimizer = None if evaluation_only else torch.optim.AdamW(parameters, lr=c['learning_rate'], weight_decay=0.)
        first_step = 0
        if args.resume:
            saved = torch.load(w.checked(args.resume), map_location=w.device, weights_only=True)
            assert saved['training_targets'] == c['training_targets'] and saved['recurrence_steps'] == c['recurrence_steps']
            assert saved['training_method'] == c['training_method']
            assert torch.equal(saved['requests'].cpu(), torch.from_numpy(requests))
            assert torch.equal(saved['row_batches'].cpu(), torch.from_numpy(batches))
            assert torch.equal(saved['targets'].cpu(), torch.from_numpy(targets))
            action.load_state_dict(saved['action'])
            learned_columns.data.copy_(saved['source_columns'])
            optimizer.load_state_dict(saved['optimizer'])
            first_step = saved['step']
        trace = []
        for step in range(first_step, c['steps']):
            target_seed = int(targets[step])
            target = dictionaries[target_seed]
            batch = [fit[i] for i in batches[step]]
            q = torch.tensor(requests[step], device=w.device)[parts]
            mode = 'source'
            with torch.no_grad():
                teacher_h, mask, teacher_m = forward(batch)
            mode = c['training_method']
            student_h, _, student_m = forward(batch)
            hidden_loss = state_mse(student_h, teacher_h, mask)/scales[0]
            response_loss = (student_m-teacher_m).square().mean()/scales[1]
            loss = (1-c['response_weight'])*hidden_loss+c['response_weight']*response_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            assert bool(torch.isfinite(loss)) and bool(torch.isfinite(grad))
            optimizer.step()
            trace.append(dict(step=step+1, target_seed=target_seed, loss=float(loss.detach()),
                              hidden=float(hidden_loss.detach()), response=float(response_loss.detach()), grad=float(grad)))
            if (step+1) % c['checkpoint_every'] == 0 or step+1 == c['steps']:
                torch.save(dict(action=action.state_dict(), optimizer=optimizer.state_dict(), step=step+1,
                    source_columns=learned_columns.detach(), training_method=c['training_method'],
                    training_targets=c['training_targets'], recurrence_steps=c['recurrence_steps'],
                    requests=torch.from_numpy(requests), row_batches=torch.from_numpy(batches),
                    targets=torch.from_numpy(targets)), w.run/f'action_step{step+1}.pt')
                write(w.run/'training.json', trace)
                w.progress('SHARED_RULE_TRAINING', **trace[-1], peak_cuda_bytes=torch.cuda.max_memory_allocated())
        response_arrays = {}
        action.requires_grad_(False)
        learned_columns.requires_grad_(False)
        for target_seed in c['target_seeds']:
            target = dictionaries[target_seed]
            methods = [('none', 'none'), ('source', 'source'), ('initial', 'tangent'),
                       ('readout_initial', 'readout'), ('analytic', 'analytic')]
            methods.extend([('shared', 'shared'), *[(name, name) for name in transferred_columns]] if evaluation_only
                           else [(c['training_method'], c['training_method'])])
            program = torch.load(w.checked(c['program_checkpoints'][str(target_seed)]), map_location=w.device, weights_only=True)
            program_target = AutoEncoderTopK(dim, len(states[target_seed]['encoder.weight']), int(target.k)).to(w.device)
            program_target.load_state_dict(program['dictionary'])
            program_target.eval().requires_grad_(False)
            methods.append(('program', 'tangent'))
            for name, execution in methods:
                mode = execution
                target = program_target if name == 'program' else dictionaries[target_seed]
                values = np.empty((len(c['queries']), len(rows)), dtype=np.float64)
                counts = []
                with torch.no_grad():
                    for qi, (query, vector) in enumerate(c['queries'].items()):
                        q = torch.tensor(vector, device=w.device, dtype=torch.float32)[parts]
                        for off in range(0, len(rows), c['eval_batch_pairs']):
                            batch = rows[off:off+c['eval_batch_pairs']]
                            _, _, margins = forward(batch)
                            values[qi, off:off+len(batch)] = margins.cpu().numpy()
                            for row, value in zip(batch, margins.cpu().tolist()):
                                w.record(kind='grammar_program', task=row['task'], row_id=row['row_id'],
                                    component=row['task']+':'+str(row['row_id']), mode=query, operation=query,
                                    method=name, seed=source_seed, target_seed=target_seed, split=evaluation_split, margin=value)
                response_arrays[f't{target_seed}_{name}'] = values
                np.savez_compressed(w.run/'responses.npz', **response_arrays)
                if counts:
                    write(w.run/f't{target_seed}_{name}_execution.json', counts)
                w.progress('SHARED_RULE_EVALUATION', target_seed=target_seed, method=name)
            del program_target, program
        w.checks.update(dictionaries_frozen=all(torch.equal(dictionaries[s].state_dict()[k], value)
            for s, state in states.items() for k, value in state.items()),
            heldout_targets_not_optimized=not bool(set(c['heldout_targets']) & set(targets.tolist())),
            finite_parameter=bool(torch.isfinite(action.weight).all()), base_model_frozen=not any(p.requires_grad for p in model.parameters()))
        if evaluation_only:
            w.checks.update(shared_parameter_unchanged=torch.equal(action.weight, initial_action),
                no_evaluation_training=c['steps'] == 0 and len(trace) == 0 and optimizer is None,
                evaluation_parameters_frozen=not any(p.requires_grad for p in action.parameters()),
                source_columns_frozen=all(not value.requires_grad for value in transferred_columns.values()))
        else:
            w.checks['shared_parameter_updated'] = (not torch.equal(learned_columns, source['decoder']) if args.source_columns
                                                  else not torch.equal(action.weight, torch.eye(dim, device=w.device)))
        write(w.run/'rule_identity.json', dict(parameters=sum(p.numel() for p in parameters),
            evaluation_only=evaluation_only, evaluation_source_seed=source_seed, updates_performed=len(trace),
            shared_parameter_count=action.weight.numel(),
            training_method=c['training_method'], recurrence_steps=c['recurrence_steps'] if not args.source_columns else 0,
            training_target_counts={str(s): int((targets == s).sum()) for s in c['training_targets']},
            heldout_targets=c['heldout_targets'], dictionary_parameters_updated=0,
            weight_change_frobenius=float((action.weight-torch.eye(dim, device=w.device)).norm())))
    except Exception:
        error = traceback.format_exc()
    finally:
        if handle is not None:
            handle.remove()
    return w.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
