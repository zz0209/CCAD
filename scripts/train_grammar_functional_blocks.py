from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
from transformers import get_constant_schedule_with_warmup

from run_causalgym_multisite import MultisiteWork, write
from train_grammar_material_support import load_config, load_array
from ccad.artifacts import sha256


def variance_statistics(states):
    count, mean, centered_sum = 0, np.zeros(states.shape[1], np.float64), 0.
    for start in range(0, len(states), 4096):
        values = np.asarray(states[start:start+4096], dtype=np.float64)
        batch_mean = values.mean(0)
        delta = batch_mean-mean
        new_count = count+len(values)
        centered_sum += float(np.square(values-batch_mean).sum())
        centered_sum += float(np.square(delta).sum())*count*len(values)/new_count
        mean += delta*len(values)/new_count
        count = new_count
    scale = centered_sum/(count*states.shape[1])
    assert np.isfinite(scale) and scale > 0
    return dict(states=count, dimensions=states.shape[1], centered_sum_squares=centered_sum,
                mean_square_variance=scale, mean=mean.tolist())


def quality(sae, states, groups, allocation, encode, batch_size):
    statistics = variance_statistics(states)
    squared_error, l0_total = 0., 0
    alive = torch.zeros(8192, dtype=torch.bool, device=sae.device)
    group_l0 = dict.fromkeys(groups, 0)
    membership = torch.empty(8192, dtype=torch.long, device=sae.device)
    for index, name in enumerate(groups):
        membership[groups[name]] = index
    with torch.no_grad(), torch.autocast(device_type=sae.device.type, enabled=False):
        for start in range(0, len(states), batch_size):
            x = torch.as_tensor(np.array(states[start:start+batch_size]), device=sae.device)
            out = encode(sae, x, groups, allocation)
            reconstruction = sae.decode(out.top_acts, out.top_indices)
            squared_error += float((reconstruction-x).double().square().sum())
            active = out.top_acts > 0
            l0_total += int(active.sum())
            alive[out.top_indices[active]] = True
            for index, name in enumerate(groups):
                group_l0[name] += int((active & (membership[out.top_indices] == index)).sum())
    return dict(states=len(states), squared_error=squared_error,
        total_variance=statistics['centered_sum_squares'],
        fve=1-squared_error/statistics['centered_sum_squares'], l0=l0_total/len(states),
        alive=int(alive.sum()), dead=int((~alive).sum()),
        group_l0={name: value/len(states) for name, value in group_l0.items()},
        decoder_norm_max_error=float((sae.W_dec.detach().norm(dim=1)-1).abs().max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    sys.path.insert(0, cfg['sparsify_overlay_dir'])
    sys.path.insert(0, cfg['sparsify_source_dir'])
    from sparsify import SparseCoder
    from ccad.functional_blocks import assign_groups, encode, save_checkpoint, load_checkpoint

    cfg.setdefault('generator_script', 'scripts/train_grammar_functional_blocks.py')
    work = MultisiteWork(cfg, args.config, ['scripts/train_grammar_functional_blocks.py',
        'src/ccad/functional_blocks.py', 'scripts/train_grammar_material_support.py',
        'scripts/run_causalgym_multisite.py', 'src/ccad/artifacts.py'])
    error = None
    encoded_states = backward_calls = quality_states = 0
    try:
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest')
        torch.use_deterministic_algorithms(True)
        work.torch, work.device = torch, torch.device(cfg['device'])
        torch.cuda.set_device(work.device)
        torch.cuda.reset_peak_memory_stats(work.device)
        work.environment = dict(python=sys.executable, torch=torch.__version__, numpy=np.__version__,
            dtype='float32', autocast=False, matmul_precision='highest',
            encoder='ordinary torch linear relu topk', optimizer='torch.optim.Adam',
            teacher_capture_matmul_precision='high', cpu_threads=4)
        steps = int(cfg['steps'])
        assert cfg['arms'] == ['global', 'group']
        assert int(cfg['batch_pairs']) == 32 and int(cfg['natural_batch_size']) == 256
        assert float(cfg['learning_rate']) == 1e-4 and int(cfg['warmup_steps']) == 32
        manifest_path = work.checked(cfg['natural_manifest'])
        manifest = json.loads(manifest_path.read_text())
        natural = load_array(work, manifest['natural_states'])
        validation = load_array(work, manifest['quality_states'])
        cal_run = Path(cfg['calibration_run'])
        cal_path = work.checked(cal_run/'relations/calibration_information.npz')
        membership_path = work.checked(cal_run/'calibration_membership.json')
        cal = json.loads(membership_path.read_text())
        with np.load(cal_path, allow_pickle=False) as bank:
            hidden = bank['hidden'].copy()
            teachers = {name: bank[name+'_teacher'].copy() for name in ('number', 'time')}
        assert hidden.shape == (64, 2048) and len(cal['rows']) == 64
        assert all(row['cue_id'] == 0 for row in cal['rows'])
        recipients = np.tile(np.arange(64), 2)
        donors = np.array([pair[name] for name in ('number', 'time') for pair in cal['pairs']])
        for index, name in enumerate(('number', 'time')):
            assert teachers[name].shape == (128, 2048)
            assert np.count_nonzero(teachers[name][64*(1-index):64*(2-index)]) == 0
        source_path = Path(cfg['source_checkpoint'])
        source_hash = sha256(work.checked(source_path/'sae.safetensors'))
        components_path = work.checked(cfg['source_components'])
        component_rows = json.loads(components_path.read_text())['rows']
        members = {name: next(row['members'] for row in component_rows
            if row['seed'] == 1 and row['factor'] == name and row['budget'] == budget)
            for name, budget in [('number', 16), ('time', 32)]}
        source = SparseCoder.load_from_disk(source_path, device='cpu').float()
        source_decoder = source.W_dec.detach().clone()
        del source
        scales = dict(natural=variance_statistics(natural), calibration=variance_statistics(hidden),
            functional={name: float(np.square(values[index*64:(index+1)*64].astype(np.float64)).mean())
                        for index, (name, values) in enumerate(teachers.items())},
            functional_on_factor_pairs=64, functional_zero_pairs=64)
        assert all(np.isfinite(value) and value > 0 for value in scales['functional'].values())
        write(work.run/'loss_scales.json', scales)
        work.progress('FIXED_SCALES', natural_states=len(natural), calibration_states=len(hidden))
        # 两种纯轴每步各16对，固定循环顺序在两臂和所有目标间共用。
        rng = np.random.default_rng(int(cfg['training_seed']))
        pair_schedule = []
        for cycle in range((max(steps, 512)+3)//4):
            number_order, time_order = rng.permutation(64), rng.permutation(64)+64
            for part in range(4):
                pair_schedule.append(np.concatenate([number_order[part*16:(part+1)*16],
                                                     time_order[part*16:(part+1)*16]]))
        pair_schedule = np.asarray(pair_schedule[:steps], dtype=np.int64)
        natural_schedule = np.random.default_rng(int(cfg['training_seed'])+1).integers(
            len(natural), size=(steps, 256), dtype=np.int64)
        np.savez_compressed(work.run/'training_schedule.npz', pairs=pair_schedule, natural=natural_schedule)
        initial = {int(item['seed']): Path(item['path']) for item in cfg['initial_checkpoints']}
        assert sorted(initial) == sorted(cfg['seeds'])
        initial_hashes = {str(seed): sha256(work.checked(path/'sae.safetensors')) for seed, path in initial.items()}
        identity = dict(calibration_sha256=sha256(cal_path), membership_sha256=sha256(membership_path),
            natural_manifest_sha256=sha256(manifest_path), source_checkpoint_sha256=source_hash,
            source_components_sha256=sha256(components_path), initial_hashes=initial_hashes,
            steps=steps, training_seed=int(cfg['training_seed']), learning_rate=1e-4, warmup_steps=32,
            allocation=['global', 'group'], functional_scale='64 on-factor mean square per coordinate',
            fvu_scale='fixed complete pool centered mean square per coordinate',
            source_cache_precision='high', train_precision='highest float32 autocast disabled')
        write(work.run/'training_identity.json', identity)
        resume_cells = {}
        if cfg.get('resume_run'):
            parent = Path(cfg['resume_run'])
            assert json.loads(work.checked(parent/'training_identity.json').read_text()) == identity
            resume_cells = json.loads(work.checked(parent/'checkpoint_index.json').read_text())['cells']
        index = dict(cells={}, training_identity=identity)
        checkpoint_steps = sorted(set(cfg.get('checkpoint_steps', [256, steps])+[steps]))
        hidden_tensor = torch.as_tensor(hidden, device=work.device)
        teacher_tensors = {name: torch.as_tensor(value, device=work.device) for name, value in teachers.items()}
        for seed, initial_path in initial.items():
            initial_sae = SparseCoder.load_from_disk(initial_path, device='cpu').float()
            groups_cpu, scores = assign_groups(initial_sae.W_dec, source_decoder, members)
            group_info = dict(groups={name: ids.tolist() for name, ids in groups_cpu.items()},
                initial_checkpoint_sha256=initial_hashes[str(seed)], source_checkpoint_sha256=source_hash,
                source_members=members, ranking='descending positive cosine, feature ID, number before time')
            write(work.run/f'seed{seed}__groups.json', group_info)
            np.savez_compressed(work.run/f'seed{seed}__groups.npz', scores=scores,
                                **{name: ids.numpy() for name, ids in groups_cpu.items()})
            del initial_sae
            for allocation in cfg['arms']:
                key = f'seed{seed}__{allocation}'
                old = resume_cells.get(key)
                if old and old['step'] == steps:
                    index['cells'][key] = old
                    work.record(kind='completed', task='training', component=key, mode=allocation,
                                method=allocation, seed=seed, row_id=steps, reused=True)
                    continue
                torch.manual_seed(int(cfg['training_seed']))
                sae = SparseCoder.load_from_disk(initial_path, device=str(work.device)).float().train()
                groups = {name: ids.to(work.device) for name, ids in groups_cpu.items()}
                if old:
                    sae, saved_groups, metadata = load_checkpoint(old['path'], str(work.device))
                    assert all(torch.equal(groups[name], saved_groups[name]) for name in groups)
                    assert metadata['allocation'] == allocation
                assert sae.num_latents == 8192 and sae.d_in == 2048 and sae.cfg.normalize_decoder
                optimizer = torch.optim.Adam(sae.parameters(), lr=1e-4)
                scheduler = get_constant_schedule_with_warmup(optimizer, 32)
                start_step = 0
                if old:
                    state = torch.load(work.checked(old['state_path']), map_location=work.device, weights_only=True)
                    optimizer.load_state_dict(state['optimizer'])
                    scheduler.load_state_dict(state['scheduler'])
                    start_step = int(state['step'])
                    assert start_step == old['step']
                cell_dir = work.run/key
                cell_dir.mkdir()
                for step in range(start_step+1, steps+1):
                    if time.perf_counter()-work.wall_start > cfg['budget_seconds']:
                        raise TimeoutError('Functional training budget exceeded; saved checkpoints remain available')
                    selected = pair_schedule[step-1]
                    ids = np.concatenate([recipients[selected], donors[selected]])
                    cal_x = hidden_tensor[ids]
                    nat_x = torch.as_tensor(np.array(natural[natural_schedule[step-1]]), device=work.device)
                    sae.set_decoder_norm_to_unit_norm()
                    with torch.autocast(device_type=work.device.type, enabled=False):
                        cal_out = encode(sae, cal_x, groups, allocation)
                        cal_reconstruction = sae.decode(cal_out.top_acts, cal_out.top_indices)
                        codes = torch.zeros_like(cal_out.pre_acts).scatter(1, cal_out.top_indices, cal_out.top_acts)
                        factor_losses = {}
                        for name in ('number', 'time'):
                            delta = (codes[32:, groups[name]]-codes[:32, groups[name]]) @ sae.W_dec[groups[name]]
                            factor_losses[name] = (delta-teacher_tensors[name][selected]).square().mean()/scales['functional'][name]
                        functional_loss = (factor_losses['number']+factor_losses['time'])/2
                        nat_out = encode(sae, nat_x, groups, allocation)
                        nat_reconstruction = sae.decode(nat_out.top_acts, nat_out.top_indices)
                        nat_fvu = (nat_reconstruction-nat_x).square().mean()/scales['natural']['mean_square_variance']
                        cal_fvu = (cal_reconstruction-cal_x).square().mean()/scales['calibration']['mean_square_variance']
                        loss = .5*functional_loss+.25*nat_fvu+.25*cal_fvu
                    assert torch.isfinite(loss)
                    lr_used = float(optimizer.param_groups[0]['lr'])
                    loss.backward()
                    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in sae.parameters())
                    sae.remove_gradient_parallel_to_decoder_directions()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    encoded_states += 320
                    backward_calls += 1
                    item = dict(step=step, loss=float(loss.detach()), functional_loss=float(functional_loss.detach()),
                        number_loss=float(factor_losses['number'].detach()), time_loss=float(factor_losses['time'].detach()),
                        natural_fvu=float(nat_fvu.detach()), calibration_fvu=float(cal_fvu.detach()), learning_rate=lr_used)
                    with (cell_dir/'loss_trace.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(item)+'\n')
                    if step in checkpoint_steps:
                        sae.set_decoder_norm_to_unit_norm()
                        directory = cell_dir/f'step_{step}'
                        save_checkpoint(sae, directory, groups, dict(group_info, allocation=allocation,
                            seed=seed, step=step, precision='highest float32 autocast disabled'))
                        quality_bank = {name: quality(sae, values, groups, allocation, encode,
                            int(cfg.get('quality_batch_size', 256))) for name, values in [('natural', validation), ('calibration', hidden)]}
                        quality_states += len(validation)+64
                        write(directory/'quality.json', quality_bank)
                        state_path = cell_dir/'resume_state.pt'
                        temp_path = cell_dir/'resume_state.pt.tmp'
                        torch.save(dict(step=step, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()), temp_path)
                        os.replace(temp_path, state_path)
                        saved = dict(seed=seed, arm=allocation, allocation=allocation, step=step, path=str(directory),
                            state_path=str(state_path), weights_sha256=sha256(directory/'sae.safetensors'), quality=quality_bank)
                        index['cells'][key] = saved
                        write(work.run/'checkpoint_index.json', index)
                        for name, values in quality_bank.items():
                            work.record(kind='quality', task=name, component=key, mode=allocation, method=allocation,
                                        seed=seed, row_id=step, **values)
                    if step % 32 == 0 or step in checkpoint_steps:
                        work.progress('TRAINING', seed=seed, allocation=allocation, step=step, steps=steps,
                                      loss=item['loss'], encoded_states=encoded_states, backward_calls=backward_calls)
                del sae, optimizer, scheduler, codes, cal_out, nat_out, cal_reconstruction, nat_reconstruction
                torch.cuda.empty_cache()
        write(work.run/'final_checkpoints.json', dict(checkpoints=list(index['cells'].values())))
        work.checks.update(all_cells_complete=len(index['cells']) == 2*len(initial),
            exact_steps=all(item['step'] == steps for item in index['cells'].values()),
            matched_pair_schedule=True, fixed_source_teacher=True, target_only_parameters=True)
    except Exception:
        error = traceback.format_exc()
    write(work.run/'execution_cost.json', dict(encoded_training_states=encoded_states,
        sae_backward_calls=backward_calls, quality_states=quality_states, model_sequence_forwards=0,
        model_token_forwards=0, cached_source_teacher=True,
        source_teacher_acquisition_run=cfg['calibration_run'], natural_capture_manifest=cfg['natural_manifest']))
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
