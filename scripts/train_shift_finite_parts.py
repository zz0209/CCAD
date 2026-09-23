from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time
import traceback

import numpy as np

from run_causalgym_multisite import MultisiteWork, write
from run_shift_explanation import source_groups
from train_shift_dictionaries import site_module
from train_shift_program import project_rows
from ccad.artifacts import sha256


def load_config(path):
    current = json.loads(Path(path).read_text())
    result = load_config(current['base_config']) if current.get('base_config') else {}
    result.update(current)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    c = load_config(args.config)
    variants = ['joint_vector', 'independent_vector', 'joint_mixed']
    assert c.get('variants', variants) == variants
    work = MultisiteWork(c, args.config, ['scripts/train_shift_finite_parts.py',
        'scripts/refine_shift_program.py', 'scripts/run_shift_conditional_restore.py',
        'scripts/train_shift_program.py', 'scripts/run_shift_explanation.py',
        'scripts/train_shift_dictionaries.py', 'scripts/run_causalgym_multisite.py', 'src/ccad/artifacts.py'])
    hooks, error = [], None
    backward_sequences = 0
    try:
        import torch
        import transformers
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        work.torch, work.device = torch, torch.device(c['device'])
        torch.cuda.set_device(work.device)
        torch.cuda.reset_peak_memory_stats(work.device)
        work.environment = dict(python=sys.executable, python_version=platform.python_version(),
            torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__, cpu_threads=2)
        if c.get('base_config'):
            work.checked(c['base_config'], 'Inherited configuration')
        sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        source_path = work.checked(c['source_manifest'], 'Published source members', 'MIT')
        source = json.loads(source_path.read_text())
        groups, _ = source_groups(work.checked(c['notebook'], 'Published source annotations', 'MIT'), source['members'])
        group_names = ['pronouns', 'names', 'associated_words']
        parameter_path = work.checked(c['source_parameters'], 'Published source parameters', 'MIT')
        source_bank = np.load(parameter_path)
        relation_path = work.checked(Path(c['relation_run'])/'relation.npz', 'Original R59 relations')
        old = np.load(relation_path)
        selection_path = work.checked(c['vector_selection_file'], 'Frozen finite-vector W group')
        selected = json.loads(selection_path.read_text())['finite_vector_n8']
        sites = list(source['members'])
        parameters, targets, supports, partitions, active_parts, initial = {}, {}, {}, {}, {}, {}
        raw_indices, raw_maps, target_hashes = {}, {}, {}
        for site in sites:
            parameters[site] = {k: torch.tensor(source_bank[site+'__'+k], device=work.device)
                                for k in ['encoder', 'encoder_bias', 'decoder', 'center']}
            partition = np.array([[i in groups[g].get(site, []) for g in group_names]
                                  for i in source['members'][site]], dtype=np.float32)
            assert np.all(partition.sum(1) == 1)
            partitions[site] = torch.tensor(partition, device=work.device)
            active_parts[site] = partitions[site].sum(0) > 0
            native = old[site+'__native']
            candidate = old[site+'__candidates']
            added = selected.get(site, [])
            assert set(added) <= set(candidate.tolist())
            support = np.union1d(np.flatnonzero(native.sum(1) > 0), added).astype(np.int64)
            supports[site] = torch.tensor(support, device=work.device)
            initial[site] = torch.tensor(native[support]@partition, device=work.device, dtype=torch.float32)
            assert float(initial[site].sum(1).max()) <= 1.000001
            raw_indices[site] = torch.tensor(candidate, device=work.device, dtype=torch.long)
            raw_maps[site] = torch.tensor(old[site+'__raw'], device=work.device, dtype=torch.float32)
            path = work.checked(Path(c['target_directory'])/f'{site}_seed{c["target_seed"]}.pt', 'Frozen target SAE')
            target_hashes[site] = sha256(path)
            state = torch.load(path, map_location=work.device, weights_only=True)
            target = AutoEncoderTopK(512, state['encoder.weight'].shape[0], int(state['k'])).to(work.device)
            target.load_state_dict(state)
            targets[site] = target.eval().requires_grad_(False)
        assert sum(map(len, selected.values())) == 22
        probe_path = work.checked(Path(c['frozen_source_run'])/'probe.npz', 'Original fixed occupation head')
        probe = np.load(probe_path)
        pw = torch.tensor(probe['weight'].reshape(-1), device=work.device)
        pb = torch.tensor(probe['bias'].reshape(-1)[0], device=work.device)
        for name in ['config.json', 'tokenizer.json', 'model.safetensors']:
            work.checked(Path(c['model_local_dir'])/name, 'Pinned Pythia70M', 'Apache-2.0')
        model = transformers.AutoModelForCausalLM.from_pretrained(c['model_local_dir'], local_files_only=True,
            dtype=torch.float32, attn_implementation='eager').eval().to(work.device)
        model.requires_grad_(False)
        model.config.use_cache = False
        operation_names = ['0', 'P', 'N', 'W', 'PN', 'PW', 'NW', 'PNW']
        request_values = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                   [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], np.float32)
        requests = torch.tensor(request_values, device=work.device)
        mode, q, mask, pooled, coefficients = 'none', requests[0], None, None, initial
        capture_capacity, capacity_current = False, {}

        def hook(site):
            def apply(module, inputs, output):
                nonlocal pooled
                x = output[0] if isinstance(output, tuple) else output
                if mode == 'source':
                    s = parameters[site]
                    z = torch.relu((x-s['center'])@s['encoder'].T+s['encoder_bias'])
                    x = x-(z*(partitions[site]@q))@s['decoder']
                elif mode == 'raw':
                    z = targets[site].encode(x)[..., raw_indices[site]]
                    x = x-((z@raw_maps[site])*(partitions[site]@q))@parameters[site]['decoder']
                elif mode == 'target':
                    z = targets[site].encode(x)[..., supports[site]]
                    requested = coefficients[site]@q
                    fraction = requested.clamp(max=1.)
                    if capture_capacity:
                        clipped = (z>0)&(requested>1.)&mask[..., None].bool()
                        item = capacity_current[site]
                        item['active_member_tokens_clipped'] += int(clipped.sum())
                        item['tokens_with_clip'] += int(clipped.any(-1).sum())
                        item['removed_code_excess'] += float((z*(requested-1.).clamp_min(0)*mask[..., None]).sum())
                    x = x-(z*fraction)@targets[site].decoder.weight[:, supports[site]].T
                if site == 'resid_4':
                    pooled = (x*mask[..., None]).sum(1)/mask.sum(1)[:, None]
                return (x, *output[1:]) if isinstance(output, tuple) else x
            return apply
        for site in sites:
            hooks.append(site_module(model, site).register_forward_hook(hook(site)))

        def forward(rows, indices):
            nonlocal mask
            if time.perf_counter()-work.wall_start > c['budget_seconds']:
                raise TimeoutError('Finite-part training budget exceeded')
            length = max(len(rows[i]['tokens']) for i in indices)
            assert len(indices)*length <= c['token_budget']
            ids = torch.zeros((len(indices), length), device=work.device, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, i in enumerate(indices):
                tokens = rows[i]['tokens']
                ids[j, :len(tokens)] = torch.tensor(tokens, device=work.device)
                mask[j, :len(tokens)] = 1
            model.gpt_neox(ids, attention_mask=mask, use_cache=False)
            work.sequence_forwards += len(indices)
            work.token_forwards += ids.numel()
            assert torch.isfinite(pooled).all()
            if torch.cuda.max_memory_allocated(work.device) > c.get('maximum_cuda_bytes', 4*1024**3):
                raise RuntimeError('CUDA allocation budget exceeded')
            return pooled

        def evaluate_rows(rows):
            values = np.empty((len(rows), 512), np.float32)
            ordered = sorted(range(len(rows)), key=lambda i: len(rows[i]['tokens']))
            offset = 0
            with torch.no_grad():
                while offset < len(ordered):
                    ix = ordered[offset:offset+c.get('eval_batch_size', c['batch_size'])]
                    while len(ix)>1 and len(ix)*max(len(rows[i]['tokens']) for i in ix)>c['token_budget']:
                        ix = ix[:-1]
                    values[ix] = forward(rows, ix).cpu().numpy()
                    offset += len(ix)
            return values

        calibration_path = work.checked(c['calibration_panel'], 'Exact prior calibration documents')
        fit = json.loads(calibration_path.read_text())['rows']
        assert len({r['document_sha256'] for r in fit}) == len(fit) and len(fit) >= c['batch_size']
        identity = dict(source_sha256=sha256(parameter_path), relation_sha256=sha256(relation_path),
            selection_sha256=sha256(selection_path), target_checkpoint_hashes=target_hashes,
            calibration_sha256=sha256(calibration_path), probe_sha256=sha256(probe_path),
            steps=c['steps'], learning_rate=c['learning_rate'], training_seed=c['training_seed'], batch_size=c['batch_size'])
        training_identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        write(work.run/'training_identity.json', dict(identity=training_identity, inputs=identity,
            requests=operation_names[1:4], fixed_last_step=True, support='positive native union frozen finite-vector W22'))
        fitted, traces = {}, {}
        if c.get('eval_only'):
            frozen = np.load(work.checked(Path(c['frozen_relation_run'])/'relations.npz', 'Frozen complete-part coefficients'))
            assert str(frozen['training_identity']) == training_identity
            for site in sites:
                np.testing.assert_array_equal(frozen[site+'__support'], supports[site].cpu().numpy())
                np.testing.assert_array_equal(frozen[site+'__partition'], partitions[site].cpu().numpy())
            fitted = {method: {site: torch.tensor(frozen[site+'__'+method], device=work.device)
                              for site in sites} for method in variants}
        else:
            mode, q = 'none', requests[0]
            clean = evaluate_rows(fit)
            teacher = []
            for q in requests[1:4]:
                mode = 'source'
                teacher.append(evaluate_rows(fit))
            teacher = torch.tensor(np.stack(teacher, axis=1), device=work.device)
            clean_tensor = torch.tensor(clean, device=work.device)
            effects = teacher-clean_tensor[:, None, :]
            raw_pool_scale = effects.square().mean((0, 2))
            raw_head_scale = (effects@pw).square().mean(0)
            assert bool(torch.isfinite(raw_pool_scale).all()) and bool(torch.isfinite(raw_head_scale).all())
            pool_scale = raw_pool_scale.clamp_min(1e-4)
            head_scale = raw_head_scale.clamp_min(1e-4)
            rng = np.random.default_rng(c['training_seed'])
            schedule = np.array([rng.choice(len(fit), c['batch_size'], replace=False) for _ in range(c['steps'])])
            np.savez_compressed(work.run/'teacher.npz', baseline_pooled512=clean,
                source_pooled512=teacher.cpu().numpy(), pool_scale=pool_scale.cpu().numpy(),
                head_scale=head_scale.cpu().numpy(), document_sha256=np.array([r['document_sha256'] for r in fit]),
                raw_pool_scale=raw_pool_scale.cpu().numpy(), raw_head_scale=raw_head_scale.cpu().numpy(),
                requests=request_values[1:4], schedule=schedule)
            write(work.run/'calibration_membership.json', dict(rows=fit, panel=str(calibration_path)))
            checkpoint_dir = work.run/'checkpoints'
            checkpoint_dir.mkdir()
            for variant in variants:
                coefficients = {site: torch.nn.Parameter(initial[site].clone()) for site in sites}
                optimizer = torch.optim.Adam(coefficients.values(), lr=c['learning_rate'])
                start, trace = 0, []
                if c.get('resume_run'):
                    saved_paths = sorted((Path(c['resume_run'])/'checkpoints').glob(variant+'_step*.pt'))
                    if saved_paths:
                        saved = torch.load(work.checked(saved_paths[-1], 'Resume completed training checkpoint'),
                                           map_location=work.device, weights_only=True)
                        assert saved['training_identity'] == training_identity
                        for site in sites:
                            coefficients[site].data.copy_(saved['coefficients'][site])
                        optimizer.load_state_dict(saved['optimizer'])
                        start, trace = saved['step'], saved['trace']
                mode = 'target'
                for step in range(start, c['steps']):
                    part = step % 3
                    q = requests[part+1]
                    ix = schedule[step].tolist()
                    optimizer.zero_grad(set_to_none=True)
                    prediction = forward(fit, ix)
                    difference = prediction-teacher[ix, part]
                    pool_loss = difference.square().mean()/pool_scale[part]
                    head_loss = (difference@pw).square().mean()/head_scale[part]
                    loss = .5*pool_loss+.5*head_loss if variant == 'joint_mixed' else pool_loss
                    assert bool(torch.isfinite(loss))
                    loss.backward()
                    backward_sequences += len(ix)
                    optimizer.step()
                    with torch.no_grad():
                        for site, value in coefficients.items():
                            value.mul_(active_parts[site])
                            if variant == 'independent_vector':
                                value.clamp_(0, 1)
                            else:
                                project_rows(value)
                    item = dict(step=step+1, part=group_names[part], rows=ix,
                        loss=float(loss.detach()), pooled_loss=float(pool_loss.detach()), head_loss=float(head_loss.detach()))
                    trace.append(item)
                    if (step+1) % c['checkpoint_every'] == 0 or step+1 == c['steps']:
                        temporary = checkpoint_dir/f'{variant}_step{step+1:04d}.partial.pt'
                        torch.save(dict(step=step+1, training_identity=training_identity,
                            coefficients={s: v.detach().cpu() for s, v in coefficients.items()},
                            optimizer=optimizer.state_dict(), trace=trace), temporary)
                        temporary.replace(checkpoint_dir/f'{variant}_step{step+1:04d}.pt')
                        traces[variant] = trace
                        write(work.run/'training_trace.json', traces)
                        work.progress('TRAIN_COMPLETE_PARTS', method=variant, step=step+1, total_steps=c['steps'],
                                      loss=item['loss'], backward_sequences=backward_sequences)
                fitted[variant] = {site: value.detach().clone() for site, value in coefficients.items()}
                traces[variant] = trace
        export = dict(group_names=np.array(group_names), operation_names=np.array(operation_names), requests=request_values,
                      training_identity=np.array(training_identity), original_relation_path=np.array(str(relation_path)))
        relation_summary = {}
        for site in sites:
            export.update({site+'__support': supports[site].cpu().numpy(),
                site+'__source_ids': np.array(source['members'][site]),
                site+'__partition': partitions[site].cpu().numpy(), site+'__native': initial[site].cpu().numpy()})
            for method in variants:
                export[site+'__'+method] = fitted[method][site].cpu().numpy()
        for method, values in {**fitted, 'native': initial}.items():
            relation_summary[method] = {}
            for site, matrix in values.items():
                value = matrix.cpu().numpy()
                excess = np.maximum(value@request_values.T-1., 0.)
                relation_summary[method][site] = dict(members=len(value), minimum=float(value.min()),
                    maximum=float(value.max()), maximum_row_sum=float(value.sum(1).max()),
                    part_fraction_sums=value.sum(0).tolist(), clipped_rows=(excess>0).sum(0).tolist(),
                    excess_fraction_sum=excess.sum(0).tolist())
        np.savez_compressed(work.run/'relations.npz', **export)
        write(work.run/'relation_summary.json', relation_summary)
        destination = work.run/'evaluation'
        destination.mkdir()
        method_names = ['source', *variants, 'native', 'raw']
        evaluation_index = {}
        for name, path in c['evaluation_panels'].items():
            panel_path = work.checked(path, 'Fixed complete-program evaluation panel')
            panel = json.loads(panel_path.read_text())
            rows = panel['rows']
            assert rows and len({r['document_sha256'] for r in rows}) == len(rows)
            write(destination/f'{name}_membership.json', dict(rows=rows, tasks=panel.get('tasks', []), panel=str(panel_path)))
            mode, q = 'none', requests[0]
            baseline = evaluate_rows(rows)
            values = np.empty((len(rows), len(method_names), len(requests), 512), np.float32)
            values[:, :, 0] = baseline[:, None, :]
            capacity = {}
            for mi, method in enumerate(method_names):
                mode = method if method in ['source', 'raw'] else 'target'
                coefficients = fitted.get(method, initial)
                capture_capacity = mode == 'target'
                if capture_capacity:
                    capacity[method] = {'0': {site: dict(active_member_tokens_clipped=0,
                        tokens_with_clip=0, removed_code_excess=0.) for site in sites}}
                for ri in range(1, len(requests)):
                    q = requests[ri]
                    if capture_capacity:
                        capacity_current = {site: dict(active_member_tokens_clipped=0,
                            tokens_with_clip=0, removed_code_excess=0.) for site in sites}
                    values[:, mi, ri] = evaluate_rows(rows)
                    if capture_capacity:
                        capacity[method][operation_names[ri]] = capacity_current
                    work.record(kind='finite_part_evaluation', task=name, row_id=ri, component=name,
                        method=method, mode=name, operation=operation_names[ri], seed=c['target_seed'],
                        target_seed=c['target_seed'], documents=len(rows), path=str(destination/f'{name}.npz'))
                work.progress('EVALUATE_COMPLETE_PARTS', panel=name, method=method, documents=len(rows))
            capture_capacity = False
            write(destination/f'{name}_capacity.json', capacity)
            metadata = dict(row_ids=np.array([r['row_id'] for r in rows]),
                document_sha256=np.array([r['document_sha256'] for r in rows]),
                genders=np.array([r['gender'] for r in rows]),
                splits=np.array([r.get('selection_split', r['split']) for r in rows]))
            for field, key in [('label', 'labels'), ('profession', 'professions')]:
                if all(field in row for row in rows):
                    metadata[key] = np.array([row[field] for row in rows])
            np.savez_compressed(destination/f'{name}.npz', **metadata, pooled512=values,
                logits=values@pw.cpu().numpy()+float(pb), baseline_pooled512=baseline,
                baseline_logits=baseline@pw.cpu().numpy()+float(pb), method_names=np.array(method_names),
                operation_names=np.array(operation_names), requests=request_values)
            evaluation_index[name] = str((destination/f'{name}.npz').resolve())
            write(work.run/'evaluation_index.json', evaluation_index)
        work.checks.update(frozen_dictionaries=True, frozen_model=True, same_training_schedule=True,
            three_single_part_teacher=True, fixed_last_step=True, original_raw_code_readout=True,
            complete_dynamic_program_evaluation=True)
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr)
    write(work.run/'execution_cost.json', dict(backward_sequences=backward_sequences,
        sequence_forwards=work.sequence_forwards, token_forwards=work.token_forwards))
    for handle in hooks:
        handle.remove()
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
