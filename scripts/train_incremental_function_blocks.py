from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
import transformers
from transformers import get_constant_schedule_with_warmup

from ccad.artifacts import sha256
from run_causalgym_multisite import MultisiteWork, write
from train_grammar_material_support import load_config
from train_grammar_functional_blocks import variance_statistics


PARTS = ('verb', 'number', 'gender')
TASKS = ('regular_plural_subject_verb_agreement_1', 'anaphor_number_agreement', 'anaphor_gender_agreement')


def collect(work, cfg, api):
    panel_path = work.checked(cfg['panel'])
    original = json.loads(panel_path.read_text())['rows']
    fit = [row for task in TASKS for row in [r for r in original if r['task'] == task and r['split'] == 'fit'][:cfg['fit_pairs_per_task']]]
    evaluation = [row for task in TASKS for row in [r for r in original if r['task'] == task and r['split'] == 'development'][:cfg['eval_pairs_per_task']]]
    assert len(fit) == 3*cfg['fit_pairs_per_task']
    assert len(evaluation) == 3*cfg['eval_pairs_per_task']
    assert not {r[key] for r in fit for key in ('sentence_good', 'sentence_bad')} & {r[key] for r in evaluation for key in ('sentence_good', 'sentence_bad')}
    write(work.run/'panel.json', dict(fit=fit, rows=evaluation, source_panel=str(panel_path), source_panel_sha256=sha256(panel_path)))
    for filename in ('config.json', 'tokenizer.json', 'model.safetensors'):
        work.checked(Path(cfg['model_local_dir'])/filename)
    model = transformers.AutoModelForCausalLM.from_pretrained(cfg['model_local_dir'], local_files_only=True,
        dtype=torch.float32, attn_implementation='eager').eval().to(work.device)
    model.requires_grad_(False)
    model.config.use_cache = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg['model_local_dir'], local_files_only=True)
    source_path = work.checked(cfg['source_checkpoint'])
    source = api.load_dictionary(source_path, work.device).eval().requires_grad_(False)
    gate_path = work.checked(Path(cfg['source_run'])/'topk_s1_source.npz')
    gate = np.load(gate_path, allow_pickle=False)['gate']
    assert gate.shape == (8192, 3) and np.array_equal(gate.sum(0), [64, 64, 64])
    assert np.max(gate.sum(1)) == 1
    members = {name: np.flatnonzero(gate[:, index]) for index, name in enumerate(PARTS)}
    hidden = np.empty((len(fit), 1024), np.float32)
    positions, captured = None, None
    def hook(module, arguments, output):
        nonlocal captured
        values = output[0] if isinstance(output, tuple) else output
        captured = values[torch.arange(len(values), device=work.device), positions].detach()
    handle = model.get_submodule(cfg['hook_module_path']).register_forward_hook(hook)
    maximum_shared_error = 0.
    try:
        with torch.no_grad(), torch.autocast(device_type=work.device.type, enabled=False):
            for start in range(0, len(fit), cfg['capture_batch_pairs']):
                rows = fit[start:start+cfg['capture_batch_pairs']]
                length = max(len(row[key]) for row in rows for key in ('good', 'bad'))
                tokens = torch.full((2*len(rows), length), tokenizer.eos_token_id, dtype=torch.long, device=work.device)
                mask = torch.zeros_like(tokens)
                for i, row in enumerate(rows):
                    assert row['good'][:row['position']+1] == row['bad'][:row['position']+1]
                    for j, name in enumerate(('good', 'bad')):
                        tokens[2*i+j, :len(row[name])] = torch.tensor(row[name], device=work.device)
                        mask[2*i+j, :len(row[name])] = 1
                positions = torch.tensor([row['position'] for row in rows], device=work.device).repeat_interleave(2)
                model.transformer(tokens, attention_mask=mask, use_cache=False)
                difference = float((captured[::2]-captured[1::2]).abs().max())
                assert difference < .001
                maximum_shared_error = max(maximum_shared_error, difference)
                hidden[start:start+len(rows)] = captured[::2].cpu().numpy()
                work.sequence_forwards += 2*len(rows)
                work.token_forwards += int(mask.sum())
                work.progress('CAPTURE', pairs=start+len(rows), total_pairs=len(fit))
                if time.perf_counter()-work.wall_start > cfg['budget_seconds']:
                    raise TimeoutError('Incremental capture budget exceeded')
    finally:
        handle.remove()
    source_decoder = source.decoder.weight.detach().cpu().numpy().T
    with torch.no_grad():
        codes = source.encode(torch.tensor(hidden, device=work.device)).cpu().numpy()
    teachers = {name: -(codes[:, ids] @ source_decoder[ids]) for name, ids in members.items()}
    task_ids = np.array([TASKS.index(row['task']) for row in fit], dtype=np.int64)
    index = dict(arrivals={}, panel=str(work.run/'panel.json'), natural_states=cfg['natural_states'],
        source_checkpoint_sha256=sha256(source_path), source_gate_sha256=sha256(gate_path),
        model_revision=cfg['model_revision'], hook=cfg['hook_module_path'],
        positions='last common prefix token in both complete good/bad sentences',
        shared_prefix_max_error=maximum_shared_error, source_teacher='negative complete natural source group contribution')
    for arrival in cfg['arrivals']:
        old_parts = [name for name in PARTS if name != arrival]
        old_rows = np.flatnonzero(task_ids != PARTS.index(arrival))
        arrival_dir = work.run/arrival
        arrival_dir.mkdir()
        stage1_path, stage2_path = arrival_dir/'stage1.npz', arrival_dir/'stage2.npz'
        np.savez_compressed(stage1_path, hidden=hidden[old_rows],
            teachers=np.stack([teachers[name][old_rows] for name in old_parts], axis=1),
            source_decoders=np.stack([source_decoder[members[name]] for name in old_parts]),
            source_members=np.stack([members[name] for name in old_parts]), part_names=np.array(old_parts),
            row_indices=old_rows, task_ids=task_ids[old_rows])
        np.savez_compressed(stage2_path, hidden=hidden, teacher=teachers[arrival],
            source_decoder=source_decoder[members[arrival]], source_members=members[arrival],
            part_name=np.array(arrival), row_indices=np.arange(len(hidden)), task_ids=task_ids)
        index['arrivals'][arrival] = dict(stage1=str(stage1_path), stage1_sha256=sha256(stage1_path),
            stage2=str(stage2_path), stage2_sha256=sha256(stage2_path))
        work.record(kind='cache', task=arrival, component=arrival, method='source', row_id=0,
                    mode='capture', pairs=len(hidden), old_pairs=len(old_rows), maximum_shared_error=maximum_shared_error)
    work.checked(cfg['natural_states'])
    write(work.run/'cache_index.json', index)
    work.checks.update(exact_fit_count=True, complete_source_groups=True, shared_prefix_checked=True,
                       stage_specific_files=True, evaluation_disjoint=True)


def quality(ae, values, groups, allocation, api):
    statistics = variance_statistics(values)
    squared_error, l0 = 0., 0
    alive = torch.zeros(8192, dtype=torch.bool, device=ae.b_dec.device)
    with torch.no_grad():
        for start in range(0, len(values), 256):
            x = torch.tensor(values[start:start+256], device=ae.b_dec.device)
            codes = api.encode(ae, x, groups, allocation)
            reconstruction = ae.decode(codes)
            squared_error += float((reconstruction-x).double().square().sum())
            active = codes > 0
            l0 += int(active.sum())
            alive |= active.any(0)
    return dict(states=len(values), fve=1-squared_error/statistics['centered_sum_squares'],
                squared_error=squared_error, total_variance=statistics['centered_sum_squares'],
                l0=l0/len(values), alive=int(alive.sum()), dead=int((~alive).sum()),
                decoder_norm_max_error=float((ae.decoder.weight.detach().norm(dim=0)-1).abs().max()))


def train(work, cfg, api):
    cache_path = work.checked(Path(cfg['cache_run'])/'cache_index.json')
    cache_index = json.loads(cache_path.read_text())
    assert cache_index['model_revision'] == cfg['model_revision']
    natural_path = work.checked(cache_index['natural_states'])
    with np.load(natural_path, allow_pickle=False) as bank:
        natural = bank['hidden'].copy()
    natural_fit, natural_quality = natural[:-1024], natural[-1024:]
    natural_statistics = variance_statistics(natural_fit)
    assert int(cfg['batch_function_states']) == 32 and int(cfg['natural_batch_states']) == 256
    steps = int(cfg['steps'])
    assert float(cfg['learning_rate']) == 1e-4 and int(cfg['warmup_steps']) == 32
    initial = {int(item['seed']): Path(item['path']) for item in cfg['initial_checkpoints']}
    initial_hashes = {str(seed): sha256(work.checked(path)) for seed, path in initial.items()}
    identity = dict(cache_index_sha256=sha256(cache_path), natural_states_sha256=sha256(natural_path),
        initial_hashes=initial_hashes, steps=steps, training_seed=cfg['training_seed'],
        arrivals=cfg['arrivals'], batch_function_states=32, natural_batch_states=256,
        learning_rate=1e-4, warmup_steps=32, b_dec='fixed at original initialization',
        precision='highest FP32 autocast disabled')
    write(work.run/'training_identity.json', identity)
    write(work.run/'natural_scale.json', natural_statistics)
    index = dict(cells={}, training_identity=identity)
    resume_cells = {}
    if cfg.get('resume_run'):
        parent = Path(cfg['resume_run'])
        assert json.loads(work.checked(parent/'training_identity.json').read_text()) == identity
        resume_cells = json.loads(work.checked(parent/'checkpoint_index.json').read_text())['cells']
    encoded_states = backward_calls = quality_states = 0

    def read_bank(arrival, stage):
        spec = cache_index['arrivals'][arrival]
        path = work.checked(spec[stage])
        assert sha256(path) == spec[stage+'_sha256']
        with np.load(path, allow_pickle=False) as bank:
            return {key: bank[key].copy() for key in bank.files}

    def run_cell(ae, groups, metadata, values, task_ids, teachers, masks, functional_scales, frozen_parts):
        nonlocal encoded_states, backward_calls, quality_states
        seed, arrival, stage, method = metadata['target_seed'], metadata['arrival'], metadata['stage'], metadata['method']
        allocation = metadata['allocation']
        key = f'seed{seed}__{arrival}__{stage}__{method}'
        old = resume_cells.get(key)
        if old and old['step'] == steps:
            index['cells'][key] = old
            write(work.run/'checkpoint_index.json', index)
            work.record(kind='complete', task=arrival, component=key, row_id=steps,
                        mode=stage, method=method, seed=seed, reused=True)
            return old['path']
        groups = {name: ids.to(work.device) for name, ids in groups.items()}
        ae = ae.to(work.device).train()
        if old:
            ae, saved_groups, saved_metadata = api.load_checkpoint(old['path'], work.device)
            assert saved_metadata['method'] == method
            assert all(torch.equal(groups[name], saved_groups[name]) for name in groups)
        trainable, frozen = api.configure_trainable(ae, groups, frozen_parts)
        optimizer = torch.optim.Adam([p for p in ae.parameters() if p.requires_grad], lr=1e-4)
        scheduler = get_constant_schedule_with_warmup(optimizer, 32)
        start_step = 0
        if old:
            state = torch.load(work.checked(old['state_path']), map_location=work.device, weights_only=True)
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            start_step = int(state['step'])
        cell_dir = work.run/key
        cell_dir.mkdir()
        calibration_statistics = variance_statistics(values)
        write(cell_dir/'loss_scales.json', dict(functional=functional_scales,
            calibration=calibration_statistics, natural_mean_square_variance=natural_statistics['mean_square_variance']))
        # 各grammar循环覆盖，三个grammar的11/11/10名额逐步轮换。
        rng = np.random.default_rng(int(cfg['training_seed']))
        task_names = sorted(set(task_ids.tolist()))
        task_orders = {task: rng.permutation(np.flatnonzero(task_ids == task)) for task in task_names}
        cursors = dict.fromkeys(task_names, 0)
        function_schedule = []
        for step in range(steps):
            counts = [16, 16] if len(task_names) == 2 else [10 if index == step % 3 else 11 for index in range(3)]
            batch = []
            for task, count in zip(task_names, counts):
                order = task_orders[task]
                batch.extend(order[(cursors[task]+np.arange(count)) % len(order)].tolist())
                cursors[task] += count
            function_schedule.append(batch)
        function_schedule = np.asarray(function_schedule, dtype=np.int64)
        natural_schedule = np.random.default_rng(int(cfg['training_seed'])+1).integers(len(natural_fit), size=(steps, 256))
        np.savez_compressed(cell_dir/'schedule.npz', function=function_schedule, natural=natural_schedule)
        x_all = torch.tensor(values, device=work.device)
        teacher_tensors = {name: torch.tensor(value, device=work.device) for name, value in teachers.items()}
        mask_tensors = {name: torch.tensor(value, device=work.device, dtype=torch.bool) for name, value in masks.items()}
        checkpoint_steps = sorted(set(cfg.get('checkpoint_steps', [256, steps])+[steps]))
        for step in range(start_step+1, steps+1):
            if time.perf_counter()-work.wall_start > cfg['budget_seconds']:
                raise TimeoutError('Incremental training budget exceeded; checkpoints retained')
            ids = function_schedule[step-1]
            x = x_all[ids]
            natural_x = torch.tensor(natural_fit[natural_schedule[step-1]], device=work.device)
            api.normalize_decoder(ae, trainable)
            with torch.autocast(device_type=work.device.type, enabled=False):
                codes = api.encode(ae, x, groups, allocation)
                reconstruction = ae.decode(codes)
                losses = {}
                for name, teacher in teacher_tensors.items():
                    valid = mask_tensors[name][ids]
                    assert valid.any(), (method, step, name)
                    prediction = -codes[valid][:, groups[name]] @ ae.decoder.weight[:, groups[name]].T
                    losses[name] = (prediction-teacher[ids][valid]).square().mean()/functional_scales[name]
                function_loss = (.5*losses[arrival]+sum(.25*losses[name] for name in metadata['old_parts'])
                    if method == 'global_replay' else torch.stack(list(losses.values())).mean())
                natural_codes = api.encode(ae, natural_x, groups, allocation)
                natural_loss = (ae.decode(natural_codes)-natural_x).square().mean()/natural_statistics['mean_square_variance']
                calibration_loss = (reconstruction-x).square().mean()/calibration_statistics['mean_square_variance']
                loss = .5*function_loss+.25*natural_loss+.25*calibration_loss
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in ae.parameters() if p.grad is not None)
            api.constrain_gradients(ae, trainable)
            lr_used = float(optimizer.param_groups[0]['lr'])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            assert api.check_frozen(ae, frozen), (key, step, 'frozen parameters changed before restoration')
            api.restore_frozen(ae, frozen)
            encoded_states += 288
            backward_calls += 1
            item = dict(step=step, loss=float(loss.detach()), function_loss=float(function_loss.detach()),
                natural_fvu=float(natural_loss.detach()), calibration_fvu=float(calibration_loss.detach()),
                factor_losses={name: float(value.detach()) for name, value in losses.items()}, learning_rate=lr_used)
            with (cell_dir/'loss_trace.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(item)+'\n')
            if step in checkpoint_steps:
                api.normalize_decoder(ae, trainable)
                assert api.check_frozen(ae, frozen)
                checkpoint = cell_dir/f'step_{step}.pt'
                api.save_checkpoint(checkpoint, ae, groups, dict(metadata, step=step,
                    frozen_parts=frozen_parts, b_dec_frozen=True, exact_frozen_parameters=True))
                measured = quality(ae, natural_quality, groups, allocation, api)
                quality_states += len(natural_quality)
                write(cell_dir/f'quality_{step}.json', measured)
                state_path = cell_dir/'resume_state.pt'
                temporary = cell_dir/'resume_state.pt.tmp'
                torch.save(dict(step=step, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()), temporary)
                os.replace(temporary, state_path)
                saved = dict(metadata, path=str(checkpoint), step=step, state_path=str(state_path),
                    sha256=sha256(checkpoint), quality=measured, frozen_parameters_exact=True)
                index['cells'][key] = saved
                write(work.run/'checkpoint_index.json', index)
                work.record(kind='quality', task=arrival, component=key, row_id=step,
                    mode=stage, method=method, seed=seed, **measured)
            if step % 32 == 0 or step in checkpoint_steps:
                work.progress('TRAINING', target_seed=seed, arrival=arrival, stage_name=stage,
                              method=method, step=step, steps=steps, loss=item['loss'])
        write(work.run/'execution_cost.json', dict(encoded_training_states=encoded_states,
            sae_backward_calls=backward_calls, quality_states=quality_states, model_sequence_forwards=0,
            model_token_forwards=0, source_acquisition_run=cfg['cache_run']))
        return index['cells'][key]['path']

    for seed, initial_path in initial.items():
        for arrival in cfg['arrivals']:
            stage1 = read_bank(arrival, 'stage1')
            old_parts = stage1['part_names'].tolist()
            assert old_parts == [name for name in PARTS if name != arrival]
            initial_ae = api.load_dictionary(initial_path, 'cpu')
            groups, scores = api.assign_groups(initial_ae.decoder.weight,
                {name: stage1['source_decoders'][i] for i, name in enumerate(old_parts)})
            api.validate_groups(groups)
            np.savez_compressed(work.run/f'seed{seed}__{arrival}__stage1_groups.npz', scores=scores,
                                **{name: ids.numpy() for name, ids in groups.items()})
            del initial_ae
            old_teachers = {name: stage1['teachers'][:, i] for i, name in enumerate(old_parts)}
            old_scales = {name: float(np.square(value.astype(np.float64)).mean()) for name, value in old_teachers.items()}
            assert all(value > 0 for value in old_scales.values())
            stage1_paths = {}
            for allocation in ('global', 'group'):
                ae = api.load_dictionary(initial_path, 'cpu')
                metadata = dict(target_seed=seed, arrival=arrival, stage='stage1', method=allocation,
                    allocation=allocation, old_parts=old_parts, initial_sha256=initial_hashes[str(seed)],
                    source_cache=cache_index['arrivals'][arrival]['stage1'])
                stage1_paths[allocation] = run_cell(ae, groups, metadata, stage1['hidden'], stage1['task_ids'], old_teachers,
                    {name: np.ones(len(stage1['hidden']), bool) for name in old_parts}, old_scales, [])
                del ae
                torch.cuda.empty_cache()
            # 新功能的信息在两个第一阶段训练完成之后才读取。
            stage2 = read_bank(arrival, 'stage2')
            assert stage2['part_name'].item() == arrival
            new_scale = float(np.square(stage2['teacher'].astype(np.float64)).mean())
            assert new_scale > 0
            for allocation in ('global', 'group'):
                ae, previous_groups, previous_metadata = api.load_checkpoint(stage1_paths[allocation], 'cpu')
                additions, new_scores = api.assign_groups(ae.decoder.weight,
                    {arrival: stage2['source_decoder']}, previous_groups['rest'].numpy())
                next_groups = {name: previous_groups[name] for name in old_parts}
                next_groups.update(additions)
                api.validate_groups(next_groups)
                np.savez_compressed(work.run/f'seed{seed}__{arrival}__{allocation}__stage2_groups.npz', scores=new_scores,
                                    **{name: ids.numpy() for name, ids in next_groups.items()})
                methods = [allocation+'_frozen']+(['global_replay'] if allocation == 'global' else [])
                for method in methods:
                    target, _, _ = api.load_checkpoint(stage1_paths[allocation], 'cpu')
                    teacher_map = {arrival: stage2['teacher']}
                    masks = {arrival: np.ones(len(stage2['hidden']), bool)}
                    scales = {arrival: new_scale}
                    if method == 'global_replay':
                        for name in old_parts:
                            values = np.zeros_like(stage2['hidden'])
                            values[stage1['row_indices']] = old_teachers[name]
                            teacher_map[name] = values
                            masks[name] = np.isin(stage2['row_indices'], stage1['row_indices'])
                            scales[name] = old_scales[name]
                    metadata = dict(target_seed=seed, arrival=arrival, stage='stage2', method=method,
                        allocation=allocation, old_parts=old_parts, parent_checkpoint=stage1_paths[allocation],
                        source_cache=cache_index['arrivals'][arrival]['stage2'],
                        old_teacher_replay=method == 'global_replay')
                    run_cell(target, next_groups, metadata, stage2['hidden'], stage2['task_ids'], teacher_map, masks, scales,
                             [] if method == 'global_replay' else old_parts)
                    del target
                    torch.cuda.empty_cache()
                del ae
    write(work.run/'final_checkpoints.json', dict(checkpoints=list(index['cells'].values())))
    work.checks.update(all_cells_complete=len(index['cells']) == len(initial)*len(cfg['arrivals'])*5,
                      all_steps=all(item['step'] == steps for item in index['cells'].values()),
                      center_bias_fixed=True, frozen_old_parameters_exact=True, stage1_new_function_unread=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['collect', 'train'])
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    sys.path[:0] = [cfg['dictionary_overlay_dir'], cfg['dictionary_source_dir']]
    import ccad.incremental_function_blocks as api
    cfg.setdefault('generator_script', 'scripts/train_incremental_function_blocks.py')
    work = MultisiteWork(cfg, args.config, ['scripts/train_incremental_function_blocks.py',
        'src/ccad/incremental_function_blocks.py', 'scripts/train_grammar_functional_blocks.py',
        'scripts/train_grammar_material_support.py', 'scripts/run_causalgym_multisite.py', 'src/ccad/artifacts.py'])
    error = None
    try:
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest')
        torch.use_deterministic_algorithms(True)
        work.torch, work.device = torch, torch.device(cfg['device'])
        torch.cuda.set_device(work.device)
        torch.cuda.reset_peak_memory_stats(work.device)
        work.environment = dict(python=sys.executable, torch=torch.__version__, transformers=transformers.__version__,
            dtype='float32', autocast=False, matmul_precision='highest', cpu_threads=4,
            encoder='ordinary linear relu topk', command=args.command)
        work.checked(Path(cfg['dictionary_source_dir'])/'dictionary_learning/trainers/top_k.py', 'Pinned AutoEncoderTopK source', 'MIT')
        assert set(cfg['arrivals']) <= set(PARTS)
        if args.command == 'collect':
            collect(work, cfg, api)
        else:
            train(work, cfg, api)
    except Exception:
        error = traceback.format_exc()
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
