from pathlib import Path
import argparse
import json
import platform
import sys
import time
import traceback

import numpy as np

from run_causalgym_multisite import MultisiteWork, write
from ccad.activation_contract import HookPointContract, extract_primary_hook_tensor
from ccad.artifacts import sha256
from ccad.sae_quality import ce_recovered


def load_config(path):
    current = json.loads(Path(path).read_text())
    result = load_config(current['base_config']) if current.get('base_config') else {}
    result.update(current)
    return result


def eligible_positions(tokens):
    # 缓存按序列、位置保存非 EOS 输入；末位置没有同序列 next-token 标签。
    sequence_ids, positions = np.nonzero(tokens != 0)
    eligible = positions < tokens.shape[1]-1
    cache_indices = np.flatnonzero(eligible)
    sequence_ids, positions = sequence_ids[eligible], positions[eligible]
    labels = tokens[sequence_ids, positions+1].astype(np.int64)
    return cache_indices, sequence_ids, positions, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault('generator_script', 'scripts/evaluate_material_ce.py')
    source_files = ['scripts/evaluate_material_ce.py',
        'scripts/run_causalgym_multisite.py', 'src/ccad/activation_contract.py',
        'src/ccad/sae_quality.py', 'src/ccad/artifacts.py']
    if cfg.get('functional_blocks', False):
        source_files.append('src/ccad/functional_blocks.py')
    work = MultisiteWork(cfg, args.config, source_files)
    error = None
    tail_tokens = sae_tokens = 0
    try:
        import torch
        import transformers
        sys.path.insert(0, cfg['sparsify_overlay_dir'])
        sys.path.insert(0, cfg['sparsify_source_dir'])
        from sparsify import SparseCoder
        if cfg.get('functional_blocks', False):
            from ccad.functional_blocks import load_checkpoint, encode

        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        work.torch, work.device = torch, torch.device(cfg['device'])
        torch.cuda.set_device(work.device)
        torch.cuda.reset_peak_memory_stats(work.device)
        work.environment = dict(python=sys.executable, python_version=platform.python_version(),
            torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
            matmul_precision='highest', dtype='float32', cpu_threads=4)
        manifest_path = work.checked(cfg['quality_manifest'], 'Current final-layer activation cache')
        manifest = json.loads(manifest_path.read_text())
        assert manifest['hook'] == 'gpt_neox.layers.15' and manifest['hidden_size'] == 2048
        assert manifest['model_revision'] == cfg['model_revision']
        capture_oracle = json.loads(work.checked(manifest_path.parent/'final_layer_oracle.json',
            'Current cache capture oracle').read_text())
        assert capture_oracle['status'] == 'PASS'
        capture_environment = json.loads(work.checked(manifest_path.parent/'environment.json').read_text())
        assert capture_environment['matmul_precision'] == 'highest' and capture_environment['dtype'] == 'float32'
        input_records = json.loads(work.checked(manifest_path.parent/'inputs.json').read_text())['inputs']
        spec = manifest['quality_states']
        state_path = work.checked(spec['path'], 'Current validation pre-final-LN states')
        assert sha256(state_path) == spec['sha256']
        states = np.load(state_path, mmap_mode='r', allow_pickle=False)
        assert states.dtype == np.float32 and states.shape[1] == 2048
        token_spec = manifest['quality_states_input']
        token_path = work.checked(token_spec['path'], 'Original validation token stream')
        previous_hash = next(item['sha256'] for item in input_records
                             if Path(item['path']).resolve() == token_path.resolve())
        assert sha256(token_path) == previous_hash
        count = int(token_spec['sequences'])
        assert int(token_spec['tokens_per_sequence']) == 128
        tokens = np.memmap(token_path, dtype=np.uint16, mode='r').reshape(-1, 128)[:count]
        assert len(tokens) == count and len(states) == np.count_nonzero(tokens)
        cache_indices, sequence_ids, positions, labels = eligible_positions(tokens)
        weights = np.bincount(sequence_ids, minlength=count).astype(np.int64)
        assert np.all(weights > 0)
        identity = dict(cache_manifest_sha256=sha256(manifest_path), states_sha256=spec['sha256'],
            token_file_sha256=previous_hash, sequences=count, cached_states=len(states),
            eligible_positions=len(cache_indices), excluded_input_token_id=0,
            excluded_final_position=127, next_token_zero_allowed=True,
            definition='Same fixed non-EOS input positions before position127; next token remains the original token',
            original_cache_oracle=capture_oracle, matmul_precision='highest')
        write(work.run/'position_identity.json', identity)
        np.savez_compressed(work.run/'positions.npz', cache_indices=cache_indices,
            sequence_ids=sequence_ids, positions=positions, next_token_ids=labels, sequence_weight=weights)
        model_path = Path(cfg['model_local_dir'])
        for filename in ['config.json', 'model.safetensors']:
            work.checked(model_path/filename, 'Current pinned Pythia1B', 'Apache-2.0')
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
            dtype=torch.float32, attn_implementation='eager').to(work.device).eval()
        model.requires_grad_(False)
        model.config.use_cache = False
        assert model.config.num_hidden_layers == 16 and model.config.hidden_size == 2048
        normalization, head = model.gpt_neox.final_layer_norm, model.get_output_embeddings()
        contract = HookPointContract('gpt_neox.layers.15', 15, 'resid_post', 2048)
        observed = []
        def hook(_module, _args, output):
            observed.append(extract_primary_hook_tensor(output, contract).detach())
        handle = model.gpt_neox.layers[15].register_forward_hook(hook)
        try:
            with torch.no_grad():
                ids = torch.as_tensor(np.array(tokens[:1], dtype=np.int64), device=work.device)
                full_logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        finally:
            handle.remove()
        work.sequence_forwards += 1
        work.token_forwards += 128
        assert len(observed) == 1
        selected = np.flatnonzero(sequence_ids == 0)[:64]
        assert len(selected) > 0
        actual_positions = torch.as_tensor(positions[selected], device=work.device)
        target = torch.as_tensor(labels[selected], device=work.device)
        with torch.no_grad():
            cached = torch.as_tensor(np.array(states[cache_indices[selected]]), device=work.device)
            cached_logits = head(normalization(cached))
            current = full_logits[0, actual_positions]
            current_tail = head(normalization(observed[0][0, actual_positions]))
            full_ce = torch.nn.functional.cross_entropy(current, target, reduction='none')
            cached_ce = torch.nn.functional.cross_entropy(cached_logits, target, reduction='none')
            current_tail_ce = torch.nn.functional.cross_entropy(current_tail, target, reduction='none')
            ce_error = float((full_ce-cached_ce).abs().max())
            oracle = dict(sequence_id=0, full_context_tokens=128, compared_tokens=len(selected),
                cache_hidden_max_absolute_error=float((cached-observed[0][0, actual_positions]).abs().max()),
                cache_logits_max_absolute_error=float((cached_logits-current).abs().max()),
                current_tail_logits_max_absolute_error=float((current_tail-current).abs().max()),
                cache_ce_max_absolute_error=ce_error,
                current_tail_ce_max_absolute_error=float((current_tail_ce-full_ce).abs().max()),
                full_ce_mean=float(full_ce.mean()), cached_ce_mean=float(cached_ce.mean()),
                ce_tolerance=float(cfg.get('oracle_ce_tolerance', 1e-3)),
                matmul_precision='highest', status='PASS' if ce_error <= cfg.get('oracle_ce_tolerance', 1e-3) else 'FAIL')
            np.savez_compressed(work.run/'full_model_oracle.npz', positions=positions[selected],
                labels=labels[selected], full_ce=full_ce.cpu().numpy(), cached_ce=cached_ce.cpu().numpy(),
                current_tail_ce=current_tail_ce.cpu().numpy())
        tail_tokens += 2*len(selected)
        write(work.run/'full_model_oracle.json', oracle)
        assert oracle['status'] == 'PASS', oracle
        del model, full_logits, observed, cached_logits, current, current_tail, cached, ids
        torch.cuda.empty_cache()
        batch_size = int(cfg.get('batch_size_tokens', 256))
        assert batch_size > 0

        def measure(sae=None, zero=False, input_variance=None, groups=None, allocation=None):
            nonlocal tail_tokens, sae_tokens
            losses = np.empty(len(cache_indices), np.float64)
            squared_error, l0_total = 0., 0
            alive = torch.zeros(sae.num_latents, dtype=torch.bool, device=work.device) if sae is not None else None
            seen, variance = 0, 0.
            mean = np.zeros(2048, np.float64)
            with torch.no_grad():
                for start in range(0, len(cache_indices), batch_size):
                    if time.perf_counter()-work.wall_start > cfg['budget_seconds']:
                        raise TimeoutError('Material CE budget exceeded; finished checkpoint outputs remain available')
                    stop = min(start+batch_size, len(cache_indices))
                    inputs = np.array(states[cache_indices[start:stop]])
                    x = torch.as_tensor(inputs, device=work.device)
                    if sae is not None:
                        if groups is None:
                            values, indices, _ = sae.encode(x)
                        else:
                            values, indices, _ = encode(sae, x, groups, allocation)
                        reconstruction = sae.decode(values, indices)
                        squared_error += float((reconstruction-x).double().square().sum())
                        positive = values > 0
                        l0_total += int(positive.sum())
                        alive[indices[positive]] = True
                        x = reconstruction
                        sae_tokens += stop-start
                    elif zero:
                        x = torch.zeros_like(x)
                    else:
                        # 合并每批中心二阶矩，固定全部有效位置的输入方差。
                        block = inputs.astype(np.float64)
                        block_mean = block.mean(0)
                        difference = block_mean-mean
                        size = len(block)
                        variance += float(np.square(block-block_mean).sum())
                        variance += float(np.square(difference).sum())*seen*size/(seen+size)
                        mean += difference*size/(seen+size)
                        seen += size
                    logits = head(normalization(x))
                    y = torch.as_tensor(labels[start:stop], device=work.device)
                    loss = torch.nn.functional.cross_entropy(logits, y, reduction='none')
                    assert torch.isfinite(loss).all()
                    losses[start:stop] = loss.cpu().numpy()
                    tail_tokens += stop-start
            sums = np.bincount(sequence_ids, weights=losses, minlength=count)
            quality = None
            if sae is not None:
                assert input_variance is not None and input_variance > 0
                quality = dict(states=len(cache_indices), squared_error=squared_error,
                    total_variance=input_variance, fve=1-squared_error/input_variance,
                    l0=l0_total/len(cache_indices), nonzero_activations=l0_total,
                    alive=int(alive.sum()), dead=int((~alive).sum()),
                    decoder_norm_max_error=float((sae.W_dec.norm(dim=1)-1).abs().max()))
            elif not zero:
                assert seen == len(cache_indices) and variance > 0
                quality = dict(states=seen, total_variance=variance, input_mean=mean.tolist())
            return sums, float(sums.sum()/weights.sum()), quality

        clean_sum, clean_ce, input_statistics = measure()
        zero_sum, zero_ce, _ = measure(zero=True)
        write(work.run/'input_statistics.json', dict(**input_statistics,
            scope='Same CE eligible positions; training quality also includes saved position127 states'))
        np.savez_compressed(work.run/'reference_ce.npz', sequence_ids=np.arange(count),
            sequence_weight=weights, clean_loss_sum=clean_sum, zero_loss_sum=zero_sum,
            clean_sequence_ce=clean_sum/weights, zero_sequence_ce=zero_sum/weights)
        results = []
        checkpoints = cfg['checkpoints']
        assert len({(item['seed'], item['arm']) for item in checkpoints}) == len(checkpoints)
        for spec in checkpoints:
            directory = Path(spec['path'])
            work.checked(directory/'cfg.json', 'SAE checkpoint configuration')
            weights_path = work.checked(directory/'sae.safetensors', 'SAE checkpoint weights')
            groups = allocation = None
            if cfg.get('functional_blocks', False):
                work.checked(directory/'functional_groups.json', 'Fixed functional group identity')
                sae, groups, metadata = load_checkpoint(directory, device=str(work.device))
                allocation = metadata['allocation']
            else:
                sae = SparseCoder.load_from_disk(directory, device=str(work.device)).float().eval()
            sae.requires_grad_(False)
            assert sae.d_in == 2048 and sae.num_latents == 8192 and sae.cfg.k == 64
            loss_sum, reconstruction_ce, quality = measure(sae=sae,
                input_variance=input_statistics['total_variance'], groups=groups, allocation=allocation)
            recovery = ce_recovered(clean_ce, reconstruction_ce, zero_ce)
            key = f"{spec['arm']}_s{spec['seed']}"
            output_path = work.run/(key+'_ce.npz')
            np.savez_compressed(output_path, sequence_ids=np.arange(count), sequence_weight=weights,
                reconstruction_loss_sum=loss_sum, reconstruction_sequence_ce=loss_sum/weights,
                **quality)
            item = dict(seed=int(spec['seed']), arm=spec['arm'], checkpoint_path=str(directory),
                checkpoint_sha256=sha256(weights_path), eligible_positions=int(weights.sum()),
                clean_ce=clean_ce, zero_ce=zero_ce, reconstruction_ce=reconstruction_ce,
                ce_recovered=recovery, quality=quality, sequence_result_path=str(output_path))
            results.append(item)
            work.record(kind='material_ce', task='validation', component=key, row_id=0,
                mode='cached_final_layer', method=spec['arm'], seed=int(spec['seed']),
                clean_ce=clean_ce, zero_ce=zero_ce, reconstruction_ce=reconstruction_ce,
                ce_recovered=recovery, positions=int(weights.sum()), **quality)
            write(work.run/'ce_results.json', dict(position_identity=identity, results=results))
            work.progress('CHECKPOINT_CE_COMPLETE', completed=len(results), total=len(checkpoints),
                seed=int(spec['seed']), arm=spec['arm'], ce_recovered=recovery,
                cached_tail_tokens=tail_tokens, reconstructed_states=sae_tokens)
            del sae
            torch.cuda.empty_cache()
        work.checks.update(all_checkpoints=len(results)==len(checkpoints), current_full_model_oracle=True,
            same_eligible_positions=True, cache_token_identity=True, fixed_highest_precision=True)
    except Exception:
        error = traceback.format_exc()
    write(work.run/'execution_cost.json', dict(cached_tail_tokens=tail_tokens,
        reconstructed_states=sae_tokens, full_model_sequences=work.sequence_forwards,
        full_model_tokens=work.token_forwards))
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
