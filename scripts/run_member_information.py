from pathlib import Path
import argparse
import json
import sys
import time
import traceback

import numpy as np
from run_causalgym_multisite import MultisiteWork, write
from train_shift_dictionaries import site_module
import torch
import transformers
from ccad.intervention_transport import transport_delta


ROOT = Path(__file__).resolve().parents[1]
METHODS = ['none', 'source', 'initial', 'program', 'group_uniform',
           'source_norm_share', 'readout_initial', 'readout_trained']


def redistribute(columns, source_amplitude, groups, use_source):
    result = torch.zeros_like(columns)
    for group in groups:
        summed = columns[..., group].sum(-1, keepdim=True)
        if use_source:
            mass = source_amplitude[..., group]
            total = mass.sum(-1, keepdim=True)
            empty = total.squeeze(-1) == 0
            assert not bool((summed.squeeze(-1).abs().amax(-1)[empty] > 1e-7).any())
            weights = mass / total.clamp_min(torch.finfo(mass.dtype).tiny)
            result[..., group] = summed * weights.unsqueeze(-2)
        else:
            result[..., group] = summed / len(group)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--phase', choices=['smoke', 'development', 'confirmation'], required=True)
    parser.add_argument('--revision', default='v2')
    parser.add_argument('--reuse-run', type=Path)
    args = parser.parse_args()
    specification = json.loads(args.config.read_text(encoding='utf-8'))
    c = json.loads((ROOT / specification['base_config']).read_text(encoding='utf-8'))
    c.update(specification)
    c.update(run_id=f'MEMBER_INFORMATION_{args.phase.upper()}_{args.revision}_20260924',
             phase=args.phase, generator_script='scripts/run_member_information.py',
             seeds=[1] if args.phase != 'confirmation' else c['target_seeds'],
             audit_opened=args.phase == 'confirmation', candidate_family_frozen=True,
             evidence_level='frozen_confirmation' if args.phase == 'confirmation' else 'development',
             dataset_revision='member-information-crossed-vocabulary-20260924',
             scope='冻结program的组内信息干预；完整组作用保持；无训练更新，未建立推理加速。')
    Path(c['run_storage_root']).mkdir(parents=True, exist_ok=True)
    source_files = ['scripts/run_member_information.py', 'scripts/train_infinitive_program.py',
                    'scripts/run_causalgym_multisite.py', 'scripts/run_r011s1_raw_hook_asset.py',
                    'scripts/train_shift_dictionaries.py', 'src/ccad/intervention_transport.py',
                    'src/ccad/artifacts.py']
    w = MultisiteWork(c, args.config, source_files)
    handle, error = None, None
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        torch.cuda.set_device(c['device'])
        torch.cuda.reset_peak_memory_stats()
        w.torch, w.device = torch, torch.device(c['device'])
        w.environment = {'python': sys.executable, 'torch': torch.__version__,
                         'transformers': transformers.__version__, 'numpy': np.__version__,
                         'cpu_threads': 2, 'training_updates': 0}
        sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        w.checked(Path(c['dictionary_source_dir']) / 'dictionary_learning/trainers/top_k.py')
        source_bank = np.load(w.checked(c['source_parameters']))
        sp = {key: torch.tensor(source_bank[key], device=w.device)
              for key in ['encoder', 'encoder_bias', 'decoder', 'center']}
        panel_path = c['confirmation_panel'] if args.phase == 'confirmation' else c['development_panel']
        panel = json.loads(w.checked(panel_path).read_text(encoding='utf-8'))
        rows = panel['rows'][:8] if args.phase == 'smoke' else panel['rows']
        queries = panel['queries']
        for name in ['model.safetensors', 'config.json', 'tokenizer.json']:
            w.checked(Path(c['model_local_dir']) / name)
        model = transformers.AutoModelForCausalLM.from_pretrained(c['model_local_dir'], local_files_only=True,
                    dtype=torch.float32, attn_implementation='eager').eval().to(w.device)
        model.requires_grad_(False)
        model.config.use_cache = False
        tokenizer = transformers.AutoTokenizer.from_pretrained(c['model_local_dir'], local_files_only=True)
        answer = tokenizer.encode(panel['answer'], add_special_tokens=False)
        assert len(answer) == 1
        for row in rows:
            row['tokens'] = tokenizer.encode(row['text'], add_special_tokens=False)
        mode, query, target, initial = 'none', None, None, None
        max_group_difference, capacity_violation = 0., 0.

        def source_codes(h):
            return torch.relu((h - sp['center']) @ sp['encoder'].T + sp['encoder_bias'])

        def hook(module, inputs, output):
            nonlocal max_group_difference, capacity_violation
            h = output[0] if isinstance(output, tuple) else output
            if mode == 'source':
                h = h - (source_codes(h) * query) @ sp['decoder']
            elif mode.startswith('readout_'):
                dictionary = initial if mode == 'readout_initial' else target
                reconstructed = dictionary.decode(dictionary.encode(h))
                h = h - (source_codes(reconstructed) * query) @ sp['decoder']
            elif mode != 'none':
                dictionary = initial if mode == 'initial' else target
                delta, _, columns = transport_delta(h, dictionary, sp, query,
                                                     dictionary.encoder.weight, 8)
                if mode in ['group_uniform', 'source_norm_share']:
                    amplitude = source_codes(h) * sp['decoder'].norm(dim=-1)
                    changed = redistribute(columns, amplitude, c['groups'], mode == 'source_norm_share')
                    for group in c['groups']:
                        difference = (changed[..., group].sum(-1) - columns[..., group].sum(-1)).abs().max()
                        max_group_difference = max(max_group_difference, float(difference))
                        assert torch.allclose(changed[..., group].sum(-1), columns[..., group].sum(-1), atol=2e-6, rtol=2e-5)
                    violation = ((-changed).clamp_min(0).sum(-1) - dictionary.encode(h)).max()
                    capacity_violation = max(capacity_violation, float(violation))
                    assert float(violation) <= 2e-5
                    delta = (changed @ query) @ dictionary.decoder.weight.T
                h = h + delta
            return (h, *output[1:]) if isinstance(output, tuple) else h

        handle = site_module(model, 'resid_4').register_forward_hook(hook)

        @torch.no_grad()
        def forward(batch):
            assert time.perf_counter() - w.wall_start < c['budget_seconds']
            length = max(len(row['tokens']) for row in batch)
            ids = torch.zeros((len(batch), length), device=w.device, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for index, row in enumerate(batch):
                ids[index, :len(row['tokens'])] = torch.tensor(row['tokens'], device=w.device)
                mask[index, :len(row['tokens'])] = 1
            hidden = model.gpt_neox(ids, attention_mask=mask, use_cache=False).last_hidden_state
            last = hidden[torch.arange(len(batch), device=w.device), mask.sum(1) - 1]
            response = torch.log_softmax(model.get_output_embeddings()(last), dim=-1)[:, answer[0]]
            w.sequence_forwards += len(batch)
            w.token_forwards += int(mask.sum())
            assert torch.cuda.max_memory_allocated() <= c['peak_cuda_budget_bytes']
            return response.cpu().numpy()

        replay = {}
        for seed in c['seeds']:
            old_run = Path(c['training_run_root']) / c['target_runs'][str(seed)]
            old_config = json.loads(w.checked(old_run / 'config.resolved.json').read_text())
            dictionaries = []
            for path in [Path(old_config['target_directory']) / f'resid_4_seed{seed}.pt',
                         old_run / 'tangent_mixed/dictionary.pt']:
                state = torch.load(w.checked(path), map_location=w.device, weights_only=True)
                dictionary = AutoEncoderTopK(512, state['encoder.weight'].shape[0], int(state['k'])).to(w.device)
                dictionary.load_state_dict(state)
                dictionary.requires_grad_(False)
                dictionaries.append(dictionary)
            initial, target = dictionaries
            values = {}
            reuse_path = args.reuse_run / f'responses_t{seed}.npz' if args.reuse_run else None
            if reuse_path is not None and reuse_path.exists():
                reuse_config = json.loads(w.checked(args.reuse_run / 'config.resolved.json').read_text())
                assert reuse_config['phase'] == args.phase
                assert reuse_config['confirmation_panel'] == c['confirmation_panel']
                saved = np.load(w.checked(reuse_path, '已完成真实响应的继续执行'))
                assert set(saved.files) == set(METHODS)
                values = {key: saved[key] for key in saved.files}
                assert all(array.shape == (len(queries), len(rows)) and np.isfinite(array).all() for array in values.values())
                np.savez_compressed(w.run / f'responses_t{seed}.npz', **values)
                w.progress('REUSED_COMPLETED_TARGET', target_seed=seed, source_run=str(args.reuse_run))
            for method in METHODS:
                if method in values:
                    continue
                mode = method
                output = np.empty((len(queries), len(rows)), dtype=np.float32)
                for qi, (name, mask_values) in enumerate(queries.items()):
                    query = torch.tensor(mask_values, device=w.device, dtype=torch.float32)
                    if method == 'none' and qi:
                        output[qi] = output[0]
                        continue
                    for start in range(0, len(rows), c['batch_size']):
                        batch = rows[start:start + c['batch_size']]
                        output[qi, start:start + len(batch)] = forward(batch)
                assert np.isfinite(output).all()
                values[method] = output
                np.savez_compressed(w.run / f'responses_t{seed}.npz', **values)
                w.progress('METHOD_COMPLETE', target_seed=seed, method=method,
                           completed_methods=len(values), methods=len(METHODS), rows=len(rows), queries=len(queries))
            semantic = [i for i, name in enumerate(queries) if panel['families'][name] != 'member_subsets']
            for method in ['group_uniform', 'source_norm_share']:
                maximum = float(np.abs(values[method][semantic] - values['program'][semantic]).max())
                numeric_rms = float(np.sqrt(np.mean((values[method][semantic] - values['program'][semantic]) ** 2)))
                source_rms = float(np.sqrt(np.mean((values['source'][semantic] - values['none'][semantic]) ** 2)))
                assert source_rms > 0
                numeric_nrmse = numeric_rms / source_rms
                assert numeric_nrmse < 1e-3 and maximum < .005, (seed, method, maximum, numeric_nrmse)
                w.record(kind='semantic_identity', task='infinitive', row_id='all', component='all',
                         method=method, target_seed=seed, operation='semantic', max_abs_difference=maximum,
                         numeric_rms=numeric_rms, source_rms=source_rms, numeric_nrmse=numeric_nrmse)
            if args.phase != 'confirmation':
                reference = np.load(w.checked(old_run / 'responses.npz'))
                old_index = json.loads(w.checked(old_run / 'INDEX.json').read_text())
                assert list(queries) == old_index['queries']
                assert [row['text'] for row in rows] == [row['text'] for row in old_index['rows'][:len(rows)]]
                replay[str(seed)] = {}
                for method, old_method in [('source', 'source'), ('program', 'tangent_mixed'),
                                           ('initial', 'native_tangent_relation_8'), ('readout_initial', 'raw_reconstruction')]:
                    maximum = float(np.abs(values[method] - reference[old_method][:, :len(rows)]).max())
                    replay[str(seed)][method] = maximum
                    assert maximum < 1e-3, (method, maximum)
            for method in METHODS[2:]:
                mse = np.mean((values[method] - values['source']) ** 2, axis=1)
                energy = np.mean((values['source'] - values['none']) ** 2, axis=1)
                for name, error_mse, source_energy in zip(queries, mse, energy):
                    value = float(np.sqrt(error_mse / source_energy)) if source_energy > 0 else None
                    w.record(kind='response_error', task='infinitive', row_id=name, component='all',
                             method=method, target_seed=seed, operation=name, nrmse=value,
                             absolute_rmse=float(np.sqrt(error_mse)), source_energy=float(source_energy),
                             normalization_status='defined' if source_energy > 0 else 'zero_source_effect')
        write(w.run / 'INDEX.json', {'methods': METHODS, 'seeds': c['seeds'], 'queries': list(queries),
              'query_masks': queries, 'families': panel['families'], 'rows': rows,
              'group_sum_max_difference': max_group_difference, 'capacity_violation': capacity_violation,
              'legacy_replay_max_difference': replay})
        w.checks.update(finite_responses=True, semantic_updates_preserved=True,
                        capacity_preserved=True, frozen_checkpoints=True, training_updates_zero=True)
    except Exception:
        error = traceback.format_exc()
    finally:
        if handle is not None:
            handle.remove()
    return w.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
