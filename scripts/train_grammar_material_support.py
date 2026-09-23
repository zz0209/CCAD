from pathlib import Path
import argparse
import json
import os
import platform
import sys
import time
import traceback

import numpy as np

from run_causalgym_multisite import MultisiteWork, write
from run_r006b_topk_capacity import state_hash
from ccad.artifacts import sha256


ARMS = ['natural_replay', 'correlated', 'factorial']


def load_config(path):
    current = json.loads(Path(path).read_text())
    result = load_config(current['base_config']) if current.get('base_config') else {}
    result.update(current)
    return result


def load_array(work, spec):
    path = work.checked(spec['path'], 'Frozen material cache or batch schedule')
    assert sha256(path) == spec['sha256'], str(path)
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    assert isinstance(array, np.ndarray)
    return array


def material_batch(natural, grammar, indices):
    # 只物化当前批次，统一索引在自然缓存之后连接 grammar 缓存。
    result = np.empty((len(indices), natural.shape[1]), dtype=np.float32)
    is_natural = indices < len(natural)
    result[is_natural] = natural[indices[is_natural]]
    result[~is_natural] = grammar[indices[~is_natural]-len(natural)]
    assert np.isfinite(result).all()
    return result


def quality(sae, states, torch, device, batch_size):
    count = len(states)
    assert count > 1
    mean = np.mean(states, axis=0, dtype=np.float64)
    squared_error, variance, l0_total = 0., 0., 0
    alive = np.zeros(sae.num_latents, dtype=bool)
    sae.eval()
    with torch.no_grad():
        for start in range(0, count, batch_size):
            values = np.asarray(states[start:start+batch_size], dtype=np.float32).copy()
            x = torch.from_numpy(values).to(device)
            out = sae(x)
            assert torch.isfinite(out.sae_out).all()
            squared_error += float((out.sae_out-x).double().square().sum())
            variance += float(np.square(values.astype(np.float64)-mean).sum())
            active = out.latent_acts > 0
            l0_total += int(active.sum())
            alive[out.latent_indices[active].cpu().numpy()] = True
    assert variance > 0
    sae.train()
    return dict(states=count, fve=1-squared_error/variance, squared_error=squared_error,
                total_variance=variance, l0=l0_total/count, alive=int(alive.sum()),
                dead=int((~alive).sum()), decoder_norm_max_error=float(
                    (sae.W_dec.detach().norm(dim=1)-1).abs().max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault('generator_script', 'scripts/train_grammar_material_support.py')
    assert cfg.get('arms', ARMS) == ARMS
    work = MultisiteWork(cfg, args.config, ['scripts/train_grammar_material_support.py',
        'scripts/run_causalgym_multisite.py', 'scripts/run_r006b_topk_capacity.py',
        'src/ccad/artifacts.py'])
    error = None
    completed = {}
    trained_states = backward_calls = quality_states = 0
    try:
        import torch
        from transformers import get_linear_schedule_with_warmup
        sys.path.insert(0, cfg['sparsify_overlay_dir'])
        sys.path.insert(0, cfg['sparsify_source_dir'])
        from sparsify import SparseCoder

        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('high')
        work.torch, work.device = torch, torch.device(cfg['device'])
        torch.cuda.set_device(work.device)
        torch.cuda.reset_peak_memory_stats(work.device)
        work.environment = dict(python=sys.executable, python_version=platform.python_version(),
            torch=torch.__version__, numpy=np.__version__, dtype='float32',
            optimizer='torch.optim.Adam', matmul_precision='high', cpu_threads=4,
            model_forward_calls=0, input='frozen actual layer15 residual states')
        for filename in ['sparse_coder.py', 'trainer.py']:
            work.checked(Path(cfg['sparsify_source_dir'])/'sparsify'/filename,
                         'Sparsify fixed source', 'MIT')
        cache_path = work.checked(cfg['cache_manifest'], 'Frozen material identity')
        manifest = json.loads(cache_path.read_text())
        natural = load_array(work, manifest['natural_states'])
        grammar = load_array(work, manifest['grammar_states'])
        validation = load_array(work, manifest['quality_states'])
        for array in [natural, grammar, validation]:
            assert array.ndim == 2 and array.shape[1] == 2048 and array.dtype == np.float32
        steps = int(cfg['steps'])
        batch_size = int(cfg.get('batch_size_states', 1024))
        assert steps > 0 and batch_size == 1024
        schedules = {arm: load_array(work, manifest['schedules'][arm]) for arm in ARMS}
        for arm, indices in schedules.items():
            assert indices.dtype == np.int64 and indices.ndim == 2
            assert indices.shape[0] >= steps and indices.shape[1] == batch_size
            assert indices[:steps].min() >= 0 and indices[:steps].max() < len(natural)+len(grammar)
            if arm == 'natural_replay':
                assert np.all(indices[:steps] < len(natural))
            else:
                assert np.all(indices[:steps, :512] < len(natural))
                assert np.all(indices[:steps, 512:] >= len(natural))
        assert np.array_equal(schedules['correlated'][:steps, :512], schedules['factorial'][:steps, :512])
        assert np.array_equal(schedules['natural_replay'][:steps, :512], schedules['factorial'][:steps, :512])
        checkpoint_steps = sorted(set(cfg.get('checkpoint_steps', [256, 512, steps])+[steps]))
        checkpoint_steps = [step for step in checkpoint_steps if 0 < step <= steps]
        initial = {int(item['seed']): Path(item['path']) for item in cfg['initial_checkpoints']}
        assert sorted(initial) == sorted(cfg['seeds'])
        initial_hashes = {}
        for seed, directory in initial.items():
            work.checked(directory/'cfg.json', 'Initial SAE configuration')
            weights = work.checked(directory/'sae.safetensors', 'Original fixed SAE initialization')
            initial_hashes[str(seed)] = sha256(weights)
        identity = dict(cache_manifest_sha256=sha256(cache_path), initial_hashes=initial_hashes,
            steps=steps, batch_size_states=batch_size, learning_rate=float(cfg['learning_rate']),
            warmup_steps=int(cfg['warmup_steps']), auxk_alpha=.03125, dead_feature_threshold=8192,
            optimizer='torch.optim.Adam', firing_counters='zero_at_start',
            rng_seed=int(cfg.get('training_seed', 20260923)), arms=ARMS,
            sparse_coder_sha256=sha256(Path(cfg['sparsify_source_dir'])/'sparsify'/'sparse_coder.py'))
        write(work.run/'training_identity.json', identity)
        resume_index = {}
        if cfg.get('resume_run'):
            parent = Path(cfg['resume_run'])
            assert json.loads(work.checked(parent/'training_identity.json').read_text()) == identity
            resume_index = json.loads(work.checked(parent/'checkpoint_index.json').read_text())['cells']
        index = dict(cells={}, training_identity=identity)
        write(work.run/'checkpoint_index.json', index)
        for seed in sorted(initial):
            for arm in ARMS:
                key = f'seed{seed}__{arm}'
                old = resume_index.get(key)
                if old and old['step'] == steps:
                    index['cells'][key] = old
                    completed[key] = old
                    write(work.run/'checkpoint_index.json', index)
                    for name, values in old['quality'].items():
                        work.record(kind='quality', task=name, component=key, row_id=steps,
                            mode=arm, method='continued_sae', seed=seed, reused=True, **values)
                    continue
                torch.manual_seed(identity['rng_seed'])
                torch.cuda.manual_seed_all(identity['rng_seed'])
                sae = SparseCoder.load_from_disk(initial[seed], device=str(work.device)).float().train()
                assert sae.num_latents == 8192 and sae.cfg.k == 64 and sae.d_in == 2048
                assert sae.cfg.normalize_decoder and not sae.cfg.multi_topk and not sae.cfg.transcode
                optimizer = torch.optim.Adam(sae.parameters(), lr=identity['learning_rate'])
                scheduler = get_linear_schedule_with_warmup(optimizer, identity['warmup_steps'], steps)
                counters = torch.zeros(sae.num_latents, dtype=torch.int64, device=work.device)
                initial_state_hash = state_hash(sae.state_dict())
                start_step = 0
                if old:
                    directory = Path(old['path'])
                    del sae
                    sae = SparseCoder.load_from_disk(directory, device=str(work.device)).float().train()
                    optimizer = torch.optim.Adam(sae.parameters(), lr=identity['learning_rate'])
                    scheduler = get_linear_schedule_with_warmup(optimizer, identity['warmup_steps'], steps)
                    saved = torch.load(work.checked(old['state_path']), map_location=work.device, weights_only=True)
                    optimizer.load_state_dict(saved['optimizer'])
                    scheduler.load_state_dict(saved['scheduler'])
                    counters = saved['counters'].to(work.device)
                    torch.set_rng_state(saved['cpu_rng'].cpu())
                    torch.cuda.set_rng_state_all([value.cpu() for value in saved['cuda_rng']])
                    start_step = int(saved['step'])
                    assert start_step == old['step']
                cell_dir = work.run/key
                cell_dir.mkdir()
                write(cell_dir/'initialization.json', dict(seed=seed, arm=arm,
                    initial_state_hash=initial_state_hash, initial_checkpoint=str(initial[seed]),
                    initial_checkpoint_sha256=initial_hashes[str(seed)], resumed_step=start_step))
                trace_path = cell_dir/'loss_trace.jsonl'
                for step in range(start_step+1, steps+1):
                    if time.perf_counter()-work.wall_start > cfg['budget_seconds']:
                        raise TimeoutError('Material training budget exceeded; saved checkpoints remain available')
                    values = material_batch(natural, grammar, schedules[arm][step-1])
                    x = torch.from_numpy(values).to(work.device)
                    sae.set_decoder_norm_to_unit_norm()
                    out = sae(x, dead_mask=counters > 8192)
                    loss = out.fvu + .03125*out.auxk_loss
                    assert torch.isfinite(loss)
                    lr_used = float(optimizer.param_groups[0]['lr'])
                    loss.backward()
                    sae.remove_gradient_parallel_to_decoder_directions()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    with torch.no_grad():
                        counters += batch_size
                        counters[out.latent_indices.flatten()] = 0
                    trained_states += batch_size
                    backward_calls += 1
                    item = dict(step=step, states=step*batch_size, fvu=float(out.fvu.detach()),
                        auxk_loss=float(out.auxk_loss.detach()), loss=float(loss.detach()),
                        learning_rate=lr_used, dtype=str(x.dtype),
                        natural_states=batch_size if arm == 'natural_replay' else 512,
                        grammar_final_token_states=0 if arm == 'natural_replay' else 512)
                    with trace_path.open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(item)+'\n')
                    del out, loss, x
                    if step in checkpoint_steps:
                        directory = cell_dir/f'step_{step}'
                        assert not directory.exists()
                        sae.save_to_disk(directory)
                        quality_bank = {'natural': quality(sae, validation, torch, work.device,
                                            int(cfg.get('quality_batch_size', 256))),
                                        'grammar': quality(sae, grammar, torch, work.device,
                                            int(cfg.get('quality_batch_size', 256)))}
                        quality_states += len(validation)+len(grammar)
                        write(directory/'quality.json', quality_bank)
                        # 恢复文件只保留当前训练状态，独立的历史权重和质量指标持续保留。
                        state_path = cell_dir/'resume_state.pt'
                        temp_path = cell_dir/'resume_state.pt.tmp'
                        torch.save(dict(step=step, optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), counters=counters.cpu(),
                            cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), temp_path)
                        os.replace(temp_path, state_path)
                        saved_item = dict(seed=seed, arm=arm, step=step, path=str(directory),
                            state_path=str(state_path),
                            weights_sha256=sha256(directory/'sae.safetensors'), quality=quality_bank)
                        index['cells'][key] = saved_item
                        write(work.run/'checkpoint_index.json', index)
                        for name, values in quality_bank.items():
                            work.record(kind='quality', task=name, component=key, row_id=step,
                                mode=arm, method='continued_sae', seed=seed, reused=False, **values)
                    if step % 32 == 0 or step in checkpoint_steps:
                        work.progress('TRAINING', seed=seed, arm=arm, step=step, steps=steps,
                            trained_states=trained_states, backward_calls=backward_calls,
                            loss=item['loss'], peak_cuda_bytes=torch.cuda.max_memory_allocated())
                completed[key] = index['cells'][key]
                del sae, optimizer, scheduler, counters
                torch.cuda.empty_cache()
        write(work.run/'final_checkpoints.json', dict(checkpoints=list(completed.values())))
        work.checks.update(all_cells_complete=len(completed)==len(initial)*len(ARMS),
            exact_steps=all(item['step']==steps for item in completed.values()),
            matched_natural_order=True, cached_states_float32=True)
    except Exception:
        error = traceback.format_exc()
    write(work.run/'execution_cost.json', dict(trained_states=trained_states,
        sae_backward_calls=backward_calls, quality_states=quality_states,
        model_sequence_forwards=0, model_token_forwards=0,
        scope='Training on previously collected hidden states; cache extraction has separate cost'))
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
