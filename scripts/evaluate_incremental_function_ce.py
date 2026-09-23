import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')

import numpy as np
import torch
import torch.nn.functional as F
import transformers

from run_causalgym_multisite import MultisiteWork, write
from ccad.artifacts import sha256
from ccad.sae_quality import ce_recovered


def load_config(path):
    current = json.loads(Path(path).read_text(encoding='utf-8'))
    cfg = load_config(current['base_config']) if current.get('base_config') else {}
    cfg.update(current)
    return cfg


def checkpoint_specs(cfg, index):
    entries = index['checkpoints'] if isinstance(index, dict) else index
    selected = [dict(item) for item in entries if item['stage'] == 'stage2'
                and item['target_seed'] in cfg['target_seeds']
                and item['arrival'] in cfg['arrivals']]
    for seed in cfg['target_seeds']:
        for arrival in cfg['arrivals']:
            matches = [item for item in selected
                       if item['target_seed'] == seed and item['arrival'] == arrival]
            assert len(matches) == 3 and len({item['method'] for item in matches}) == 3
    for item in selected:
        item['key'] = f"s{item['target_seed']}__{item['arrival']}__{item['method']}"
        item['original'] = False
    for seed in cfg['target_seeds']:
        selected.append(dict(target_seed=seed, arrival=None, stage=0, method='original',
            path=cfg['target_checkpoint_template'].format(seed=seed),
            key=f'original_s{seed}', original=True))
    assert len({item['key'] for item in selected}) == len(selected)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    work = MultisiteWork(cfg, args.config, ['scripts/evaluate_incremental_function_ce.py',
        'src/ccad/incremental_function_blocks.py', 'scripts/run_causalgym_multisite.py',
        'src/ccad/sae_quality.py', 'src/ccad/artifacts.py'])
    reconstructed_states = 0
    try:
        sys.path.extend([cfg['dictionary_source_dir'], cfg['dictionary_overlay_dir']])
        from ccad.incremental_function_blocks import load_dictionary, load_checkpoint, encode

        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        work.torch, work.device = torch, torch.device(cfg['device'])
        torch.cuda.set_device(work.device)
        torch.cuda.reset_peak_memory_stats(work.device)
        work.environment = dict(python=sys.executable, python_version=platform.python_version(),
            torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
            dtype='float32', matmul_precision='highest', autocast=False, cpu_threads=2)
        manifest_path = work.checked(cfg['token_manifest'], 'R19 natural validation token manifest')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        spec = manifest['outputs']['validation']
        token_path = work.checked(spec['path'], 'R19 natural validation tokens', 'ODC-By-1.0')
        assert sha256(token_path) == spec['sha256']
        assert int(manifest['context_length']) == 128
        tokens = np.memmap(token_path, mode='r', dtype=spec.get('dtype', manifest.get('token_dtype', '<u2')))
        assert len(tokens) == int(spec['tokens']) == 32768
        tokens = np.asarray(tokens).reshape(-1, 128)
        count = int(cfg.get('validation_sequences', 256))
        assert 0 < count <= len(tokens)
        assert count == 256 or cfg['evidence_level'] == 'smoke'
        tokens = tokens[:count]
        identity = dict(token_manifest_path=str(manifest_path), token_manifest_sha256=sha256(manifest_path),
            token_path=str(token_path), token_sha256=spec['sha256'], sequences=count,
            context_length=128, reconstruction_positions=count*128, ce_positions=count*127,
            definition='Reconstruct every hook position; score all 127 next-token labels per original sequence, including EOS labels',
            original_document_count=int(spec['documents']), subset='first sequences' if count < 256 else 'complete validation')
        write(work.run/'position_identity.json', identity)
        index_path = work.checked(cfg.get('checkpoint_index', str(Path(cfg['training_run'])/'final_checkpoints.json')))
        specs = checkpoint_specs(cfg, json.loads(index_path.read_text(encoding='utf-8')))
        write(work.run/'evaluated_checkpoints.json', dict(checkpoints=specs))
        for name in ('config.json', 'model.safetensors'):
            work.checked(Path(cfg['model_local_dir'])/name, 'Pinned GPT2-medium model', 'MIT')
        model = transformers.AutoModelForCausalLM.from_pretrained(cfg['model_local_dir'],
            local_files_only=True, dtype=torch.float32, attn_implementation='eager').to(work.device).eval()
        model.requires_grad_(False)
        model.config.use_cache = False
        assert model.config.hidden_size == 1024
        module = model.get_submodule(cfg['hook_module_path'])
        batch_size = int(cfg.get('batch_size_sequences', 4))
        assert batch_size > 0

        @torch.no_grad()
        def forward(ids, replacement=None, capture=False):
            if time.perf_counter()-work.wall_start > cfg['budget_seconds']:
                raise TimeoutError('Natural CE budget exceeded; completed checkpoint outputs retained')
            observed = []

            def hook(_module, _args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if capture:
                    observed.append(hidden.detach().cpu().numpy())
                if replacement is None:
                    return output
                assert replacement.shape == hidden.shape and replacement.dtype == hidden.dtype
                return (replacement, *output[1:]) if isinstance(output, tuple) else replacement

            handle = module.register_forward_hook(hook)
            try:
                with torch.autocast(device_type=work.device.type, enabled=False):
                    logits = model(input_ids=ids, use_cache=False).logits
                    losses = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                        ids[:, 1:].reshape(-1), reduction='none').reshape(len(ids), 127)
            finally:
                handle.remove()
            assert torch.isfinite(losses).all()
            work.sequence_forwards += len(ids)
            work.token_forwards += ids.numel()
            values = losses.double().mean(1).cpu().numpy()
            if capture:
                assert len(observed) == 1
            return values, observed[0] if capture else None

        hidden_path = work.run/'validation_hidden.npy'
        hidden = np.lib.format.open_memmap(hidden_path, mode='w+', dtype=np.float32,
            shape=(count, 128, 1024))
        clean, zero = np.empty(count, np.float64), np.empty(count, np.float64)
        mean = np.zeros(1024, np.float64)
        total_variance, seen = 0., 0
        for start in range(0, count, batch_size):
            stop = min(start+batch_size, count)
            ids = torch.as_tensor(np.array(tokens[start:stop], dtype=np.int64), device=work.device)
            clean[start:stop], current = forward(ids, capture=True)
            hidden[start:stop] = current
            zero[start:stop], _ = forward(ids, replacement=torch.zeros_like(torch.as_tensor(current, device=work.device)))
            if start == 0:
                repeated, _ = forward(ids, replacement=torch.as_tensor(current, device=work.device))
                np.testing.assert_array_equal(repeated, clean[start:stop])
                work.checks['current_hook_replacement_exact'] = True
            # 固定全部自然状态的中心方差，与原训练质量定义一致。
            block = current.reshape(-1, 1024).astype(np.float64)
            block_mean = block.mean(0)
            difference = block_mean-mean
            total_variance += float(np.square(block-block_mean).sum())
            total_variance += float(np.square(difference).sum())*seen*len(block)/(seen+len(block))
            mean += difference*len(block)/(seen+len(block))
            seen += len(block)
            work.progress('NATURAL_REFERENCE', completed_sequences=stop, total_sequences=count)
        hidden.flush()
        assert seen == count*128 and total_variance > 0
        write(work.run/'input_statistics.json', dict(states=seen, total_variance=total_variance,
            input_mean=mean.tolist(), validation_hidden_path=str(hidden_path),
            validation_hidden_sha256=sha256(hidden_path)))
        weights = np.full(count, 127, dtype=np.int64)
        np.savez_compressed(work.run/'reference_ce.npz', sequence_ids=np.arange(count),
            sequence_weight=weights, clean_sequence_ce=clean, zero_sequence_ce=zero,
            clean_loss_sum=clean*weights, zero_loss_sum=zero*weights)
        clean_ce, zero_ce = float(clean.mean()), float(zero.mean())
        results = []
        for spec in specs:
            path = work.checked(spec['path'], 'Original or stage2 SAE checkpoint')
            if spec['original']:
                ae = load_dictionary(path, work.device)
                groups, allocation = {}, 'global'
            else:
                ae, groups, metadata = load_checkpoint(path, work.device)
                allocation = metadata['allocation']
                assert metadata['stage'] == 'stage2' and metadata['target_seed'] == spec['target_seed']
            ae.eval().requires_grad_(False)
            assert ae.dict_size == 8192
            counts = torch.zeros(8192, dtype=torch.int64, device=work.device)
            sequence_sse = np.empty(count, np.float64)
            reconstruction_ce = np.empty(count, np.float64)
            with torch.no_grad(), torch.autocast(device_type=work.device.type, enabled=False):
                for start in range(0, count, batch_size):
                    stop = min(start+batch_size, count)
                    ids = torch.as_tensor(np.array(tokens[start:stop], dtype=np.int64), device=work.device)
                    x = torch.as_tensor(np.array(hidden[start:stop]), device=work.device)
                    flat = x.reshape(-1, 1024)
                    z = ae.encode(flat) if spec['original'] else encode(ae, flat, groups, allocation)
                    reconstruction = ae.decode(z).reshape_as(x)
                    counts += (z > 0).sum(0)
                    sequence_sse[start:stop] = (reconstruction-x).double().square().sum((1, 2)).cpu().numpy()
                    reconstruction_ce[start:stop], _ = forward(ids, replacement=reconstruction)
                    reconstructed_states += len(flat)
                    if stop == count or (start//batch_size+1) % 8 == 0:
                        work.progress('CHECKPOINT_NATURAL_CE', checkpoint=spec['key'],
                            completed_sequences=stop, total_sequences=count,
                            completed_checkpoints=len(results), total_checkpoints=len(specs))
            norm = ae.decoder.weight.detach().norm(dim=0)
            quality = dict(states=seen, squared_error=float(sequence_sse.sum()), total_variance=total_variance,
                fve=1-float(sequence_sse.sum())/total_variance, l0=int(counts.sum())/seen,
                alive=int((counts > 0).sum()), dead=int((counts == 0).sum()),
                decoder_norm_mean=float(norm.mean()), decoder_norm_min=float(norm.min()),
                decoder_norm_max=float(norm.max()), decoder_norm_max_error=float((norm-1).abs().max()))
            reconstruction_mean = float(reconstruction_ce.mean())
            recovery = ce_recovered(clean_ce, reconstruction_mean, zero_ce)
            assert all(np.isfinite(value) for value in quality.values())
            output = work.run/(spec['key']+'_ce.npz')
            np.savez_compressed(output, sequence_ids=np.arange(count), sequence_weight=weights,
                clean_sequence_ce=clean, zero_sequence_ce=zero, reconstruction_sequence_ce=reconstruction_ce,
                reconstruction_loss_sum=reconstruction_ce*weights, sequence_squared_error=sequence_sse,
                feature_activation_counts=counts.cpu().numpy(), **quality)
            item = dict(spec, checkpoint_sha256=sha256(path), clean_ce=clean_ce, zero_ce=zero_ce,
                reconstruction_ce=reconstruction_mean, ce_recovered=recovery, quality=quality,
                sequence_result_path=str(output), allocation=allocation)
            results.append(item)
            write(work.run/'ce_results.json', dict(position_identity=identity, results=results))
            work.record(kind='incremental_natural_ce', task='natural_validation', component=spec['key'],
                row_id=0, mode='full_model_hook_reconstruction', method=spec['method'], seed=spec['target_seed'],
                arrival=spec['arrival'], operation=spec['arrival'] or 'original', clean_ce=clean_ce, zero_ce=zero_ce,
                reconstruction_ce=reconstruction_mean, ce_recovered=recovery, **quality)
            del ae, z, reconstruction
            torch.cuda.empty_cache()
        work.checks.update(all_checkpoints=len(results) == len(specs), same_natural_tokens=True,
            full_model_ce=True, no_grammar_tokens=True, all_stage2_and_original=True)
        write(work.run/'execution_cost.json', dict(full_model_sequences=work.sequence_forwards,
            full_model_tokens=work.token_forwards, reconstructed_states=reconstructed_states))
        return work.finish()
    except Exception:
        work.finish(traceback.format_exc())
        raise


if __name__ == '__main__':
    raise SystemExit(main())
