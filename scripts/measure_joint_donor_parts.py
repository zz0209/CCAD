import argparse
import json
import traceback
from pathlib import Path

from composition_runtime import CompositionRun, ROOT, write, np, extract_primary_hook_tensor, HookPointContract


def summarize(logprobs, work, factor=None):
    ids = work.evaluation_ids
    small = logprobs[:, work.labels]
    probability = np.exp(small-small.max(1, keepdims=True))
    probability /= probability.sum(1, keepdims=True)
    number_probability = probability[:, [1, 3]].sum(1)
    time_probability = probability[:, [2, 3]].sum(1)
    expected_ids = ids if factor is None else work.donors[factor][ids]
    numbers = np.array([work.rows[index]['number'] for index in expected_ids])
    times = np.array([work.rows[index]['past'] for index in expected_ids])
    number_correct = (number_probability > .5) == numbers
    time_correct = (time_probability > .5) == times
    arrays = dict(row_ids=ids, expected_row_ids=expected_ids, label_logprobs=small,
                  four_word_probability=probability, vocab_probability=np.exp(small),
                  vocab_argmax=logprobs.argmax(1), four_word_argmax=small.argmax(1),
                  number_probability=number_probability, past_probability=time_probability,
                  number_correct=number_correct, past_correct=time_correct,
                  joint_success=number_correct & time_correct,
                  expected_number=numbers, expected_past=times)
    summary = dict(rows=len(ids), number_hit=float(number_correct.mean()), past_hit=float(time_correct.mean()),
                   joint_success=float(arrays['joint_success'].mean()),
                   four_word_accuracy=float(np.mean(small.argmax(1) == numbers+2*times)),
                   mean_four_word_probability=probability.mean(0).tolist(),
                   mean_vocab_probability=np.exp(small).mean(0).tolist(),
                   mean_four_word_vocab_mass=float(np.exp(small).sum(1).mean()))
    if factor in ('number', 'time'):
        desired = number_correct if factor == 'number' else time_correct
        other = time_correct if factor == 'number' else number_correct
        summary.update(target_axis_hit=float(desired.mean()), non_target_semantic_retention=float(other.mean()))
    return arrays, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    work = CompositionRun(args.config, 'scripts/measure_joint_donor_parts.py', [])
    error = None
    try:
        config = work.cfg
        assert config['sae_layer'] == 15 and isinstance(config['final_layer_cached_tail'], bool)
        seed = int(config['source_seed'])
        assert seed == 1 and config['source_budgets'] == {'number': 16, 'time': 32}
        work.load()
        assert work.n == 384
        ids = work.evaluation_ids
        for field, option in [('cue_id', 'evaluation_cues'), ('template', 'evaluation_templates'), ('distractor', 'evaluation_distractors')]:
            if option in config:
                ids = np.array([index for index in ids if work.rows[index][field] in config[option]])
        assert len(ids) > 0
        work.evaluation_ids = ids
        last = np.array([len(row['ids'])-1 for row in work.panel['token_rows']])
        for slot in (0, 1):
            np.testing.assert_array_equal(work.positions[:, slot], last)
        capture_diagnostics = {}
        if not config['final_layer_cached_tail']:
            assert not work.cached_tail_ready
            actual_hidden = np.empty((work.n, work.h.shape[-1]), dtype=np.float32)
            contract = HookPointContract('gpt_neox.layers.15', 15, 'resid_post', work.h.shape[-1])
            module = work.model.get_submodule('gpt_neox.layers.15')
            maximum_replay_error = 0.
            for start in range(0, work.n, config['batch_size']):
                capture_ids = np.arange(start, min(start+config['batch_size'], work.n))
                captured = []
                def capture(module, arguments, output):
                    hidden = extract_primary_hook_tensor(output, contract)
                    captured.append(hidden[work.torch.arange(len(capture_ids), device='cuda'),
                        work.torch.as_tensor(last[capture_ids], device='cuda')].detach().cpu().numpy())
                handle = module.register_forward_hook(capture)
                try:
                    repeated = work.forward(capture_ids, np.zeros_like(work.h[capture_ids]))
                finally:
                    handle.remove()
                assert len(captured) == 1
                actual_hidden[capture_ids] = captured[0]
                maximum_replay_error = max(maximum_replay_error, float(np.max(np.abs(repeated-work.base[capture_ids]))))
                work.progress('CURRENT_MODEL_HIDDEN_CAPTURED', rows_completed=start+len(capture_ids), rows_total=work.n)
            assert maximum_replay_error == 0.
            capture_diagnostics = dict(baseline_replay_max_error=maximum_replay_error,
                historical_hidden_max_difference=float(np.max(np.abs(actual_hidden-work.h[:, 1]))))
            work.h[:, 0] = actual_hidden
            work.h[:, 1] = actual_hidden
            np.savez_compressed(work.run/'actual_hidden.npz', hidden=actual_hidden, row_ids=np.arange(work.n), positions=last)
        for index, row in enumerate(work.rows):
            for factor, changes in [('number', (1, 0)), ('time', (0, 1)), ('joint', (1, 1))]:
                donor_index = work.donors[factor][index]
                donor = work.rows[donor_index]
                assert donor['number'] == (row['number'] ^ changes[0])
                assert donor['past'] == (row['past'] ^ changes[1])
                assert all(donor[key] == row[key] for key in ('block', 'cue_id', 'template', 'distractor'))
                assert work.donors[factor][donor_index] == index
        from sparsify import SparseCoder
        specification = next(value for value in config['sae_checkpoints'] if value['seed'] == seed)
        checkpoint = Path(specification['path'])
        work.checked(checkpoint/'sae.safetensors')
        work.checked(checkpoint/'cfg.json')
        sae = SparseCoder.load_from_disk(checkpoint, device='cuda' if not config['final_layer_cached_tail'] else 'cpu').eval()
        decoder = sae.W_dec.detach().cpu().numpy().astype(np.float64)
        with np.load(work.checked(Path(config['material_run'])/f'seed{seed}_codes.npz')) as saved:
            codes = {factor: saved[factor+'_z'].astype(np.float64) for factor in ('number', 'time')}
        if not config['final_layer_cached_tail']:
            actual_codes = np.empty_like(codes['number'], dtype=np.float32)
            with work.torch.no_grad():
                for start in range(0, work.n, config['batch_size']):
                    stop = min(start+config['batch_size'], work.n)
                    acts, indices, _ = sae.encode(work.torch.as_tensor(actual_hidden[start:stop], device='cuda'))
                    dense = work.torch.zeros((len(acts), sae.num_latents), device='cuda').scatter_(1, indices, acts)
                    actual_codes[start:stop] = dense.cpu().numpy()
            capture_diagnostics['historical_codes_max_difference'] = float(np.max(np.abs(actual_codes-codes['number'])))
            codes = {factor: actual_codes.astype(np.float64) for factor in ('number', 'time')}
            np.savez_compressed(work.run/'actual_source_codes.npz', number_z=actual_codes, time_z=actual_codes,
                                row_ids=np.arange(work.n))
        del sae
        with np.load(work.checked(Path(config['source_run'])/f'seed{seed}_source_coordinates.npz')) as saved:
            coordinates = {key: saved[key].copy() for key in saved.files}
        np.testing.assert_array_equal(codes['number'], codes['time'])
        variants = {name: np.zeros_like(work.h, dtype=np.float64) for name in ('single_factor_donor', 'joint_donor_parts')}
        supports, diagnostics = {}, {}
        for factor, slot in [('number', 1), ('time', 0)]:
            budget = config['source_budgets'][factor]
            prefix = f'{factor}_{budget}_'
            support = coordinates[prefix+'support'].astype(int)
            assert support.shape == (budget,) and len(set(support)) == budget
            assert np.all((support >= 0) & (support < len(decoder)))
            supports[prefix+'support'] = support
            z = codes[factor]
            for method, donor_factor in [('single_factor_donor', factor), ('joint_donor_parts', 'joint')]:
                variants[method][:, slot] = (z[work.donors[donor_factor]][:, support]-z[:, support]) @ decoder[support]
            original = coordinates[prefix+'coordinates'] @ coordinates[prefix+'basis'].T
            direct = np.sum((z[work.donors[factor]][:, support]-z[:, support])[:, :, None]*decoder[support][None], axis=1)
            np.testing.assert_allclose(variants['single_factor_donor'][:, slot], direct, atol=1e-9, rtol=1e-9)
            if config['final_layer_cached_tail']:
                np.testing.assert_allclose(variants['single_factor_donor'][:, slot], original, atol=1e-9, rtol=1e-9)
            difference = variants['joint_donor_parts'][:, slot]-variants['single_factor_donor'][:, slot]
            diagnostics[factor] = dict(historical_delta_max_difference=float(np.max(np.abs(variants['single_factor_donor'][:, slot]-original))),
                actual_code_sum_max_error=float(np.max(np.abs(variants['single_factor_donor'][:, slot]-direct))),
                donor_delta_difference_rms=float(np.sqrt(np.mean(difference[ids]**2))),
                donor_delta_difference_mean_norm=float(np.linalg.norm(difference[ids], axis=1).mean()))
        # 两个命名位置指向同一final token，完整SAE对照只施加一次全部成员差量。
        z = codes['number']
        full_delta = (z[work.donors['joint']]-z) @ decoder
        np.savez_compressed(work.run/'source_supports.npz', **supports)
        write(work.run/'panel.json', dict(rows=work.rows, pairs=work.panel['pairs'],
              labels=work.panel['labels'], label_ids=work.panel['label_ids'], evaluation_ids=ids.tolist(),
              position_names=work.panel['position_names'], positions=work.positions.tolist(),
              source_coordinates=str(Path(config['source_run'])/f'seed{seed}_source_coordinates.npz')))
        baseline_arrays, baseline_summary = summarize(work.base[ids], work)
        np.savez_compressed(work.run/'baseline.npz', **baseline_arrays)
        summaries = dict(baseline=baseline_summary)
        probe_ids = ids[:config['batch_size']]
        probe_delta = variants['joint_donor_parts'].copy()
        probe_delta[:, 0] = 0
        support = supports['number_16_support']
        for index in probe_ids:
            donor_index = work.donors['joint'][index]
            expected = (codes['number'][donor_index, support]-codes['number'][index, support]) @ decoder[support]
            np.testing.assert_allclose(probe_delta[index, 1], expected, rtol=1e-12, atol=1e-12)
        probe_output = work.evaluate(probe_delta, probe_ids)[:, work.labels]
        assert np.isfinite(probe_output).all()
        np.savez_compressed(work.run/'joint_donor_first_batch.npz', row_ids=probe_ids,
                            donor_row_ids=work.donors['joint'][probe_ids], label_logprobs=probe_output,
                            delta_hidden=probe_delta[probe_ids])
        work.progress('JOINT_DONOR_FIRST_BATCH_CHECKED', rows=len(probe_ids), support_members=len(support))
        for method in ('single_factor_donor', 'joint_donor_parts', 'full_sae_joint_donor'):
            for factor, slot in [('number', 1), ('time', 0), ('joint', None)]:
                if method == 'full_sae_joint_donor':
                    delta = np.zeros_like(work.h, dtype=np.float64)
                    delta[:, 1 if slot is None else slot] = full_delta
                else:
                    delta = variants[method].copy()
                    if slot is not None:
                        delta[:, 1-slot] = 0
                output = work.measure(method, factor, delta, source_seed=seed)
                arrays, summary = summarize(output, work, factor)
                name = method+'__'+factor
                summaries[name] = summary
                np.savez_compressed(work.run/(name+'.npz'), **arrays, delta_hidden=delta[ids])
                write(work.run/'JOINT_DONOR_RESULTS.json', dict(source_seed=seed, source_rows=work.n,
                      evaluation_rows=len(ids), summaries=summaries, delta_diagnostics=diagnostics,
                      current_capture_diagnostics=capture_diagnostics,
                      forward_backend='cached_final_tail' if config['final_layer_cached_tail'] else 'full_model',
                      support_sizes={factor: config['source_budgets'][factor] for factor in ('number', 'time')},
                      intervention='Frozen R4 supports; joint donor changes number and time while each requested support alone determines the edited members',
                      full_sae_control='Complete joint-donor SAE code difference applied once at final token; single-part requests test selectivity against that same complete change'))
                work.progress('JOINT_DONOR_MEASURED', method=method, factor=factor, evaluation_rows=len(ids))
        work.checks.update(frozen_original_supports=True, frozen_source_sae=True,
                           reciprocal_two_factor_donor=True, final_token_positions=True,
                           actual_code_single_donor_sum_reproduced=True,
                           complete_requested_rows=len(work.metrics) == len(ids)*9)
        work.checks['unique'] = len(work.metrics) == len({(row['method'], row['factor'], row['row_id']) for row in work.metrics})
    except Exception as exception:
        error = f'{type(exception).__name__}: {exception}'
        (work.run/'stderr.log').write_text(traceback.format_exc(), encoding='utf-8')
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
