import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

from run_causalgym_multisite import MultisiteWork, write
from ccad.activation_contract import HookPointContract, extract_primary_hook_tensor, replace_primary_hook_tensor
from ccad.artifacts import sha256
from evaluate_grammar_material_support import load_config, read_panel, OPERATIONS
from analyze_shift_member_responses import identity, read_json, write_json
from analyze_shift_vector_reuse import summarize
from analyze_grammar_material_support import statistics, REFERENCES


ARMS = ['original', 'global_supervised', 'group_supervised']


def evaluate(config_path):
    cfg = load_config(config_path)
    cfg.setdefault('generator_script', 'scripts/evaluate_grammar_functional_blocks.py')
    work = MultisiteWork(cfg, config_path, ['scripts/evaluate_grammar_functional_blocks.py',
        'scripts/evaluate_grammar_material_support.py', 'scripts/analyze_grammar_material_support.py',
        'src/ccad/functional_blocks.py', 'scripts/run_causalgym_multisite.py',
        'src/ccad/activation_contract.py', 'src/ccad/artifacts.py'])
    error = None
    tail_forwards = encoded_states = 0
    try:
        import torch
        import transformers
        sys.path.insert(0, cfg['sparsify_overlay_dir'])
        sys.path.insert(0, cfg['sparsify_source_dir'])
        from ccad.functional_blocks import load_checkpoint, encode
        work.torch, work.device = torch, torch.device(cfg['device'])
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('high')
        torch.cuda.set_device(work.device)
        torch.cuda.reset_peak_memory_stats(work.device)
        reference = Path(cfg['reference_run'])
        assert read_json(work.checked(reference/'status.json'))['status'] == 'PASS'
        reference_cfg = read_json(work.checked(reference/'config.resolved.json'))
        assert Path(reference_cfg['model_local_dir']).resolve() == Path(cfg['model_local_dir']).resolve()
        assert reference_cfg['model_revision'] == cfg['model_revision']
        reference_name = cfg.get('reference_panel', 'development')
        entry = read_json(work.checked(reference/'evaluation_index.json'))[reference_name]
        panel = read_panel(work, Path(entry['membership_path']))
        with np.load(work.checked(entry['response_path'])) as saved:
            previous = {key: saved[key].copy() for key in saved.files}
        with np.load(work.checked(reference/(reference_name+'_current_states.npz'))) as saved:
            all_hidden = saved['hidden'].copy()
            np.testing.assert_array_equal(saved['baseline_logits4'], previous['baseline_logits4'])
        selection = np.array([i for i, row in enumerate(panel['rows'])
                              if 'evaluation_subjects' not in cfg or row['subject_id'] in cfg['evaluation_subjects']], dtype=int)
        assert len(selection)
        chosen_set = set(selection.tolist())
        assert all(panel['pairs'][index][factor] in chosen_set for index in selection for factor in ['number', 'time', 'joint'])
        lookup = {int(index): local for local, index in enumerate(selection)}
        rows = [panel['rows'][index] for index in selection]
        pairs = [{key: lookup[value] for key, value in panel['pairs'][index].items()} for index in selection]
        hidden = all_hidden[selection]
        baseline = previous['baseline_logits4'][selection]
        specs = cfg['checkpoints']
        seeds = sorted({int(spec['seed']) for spec in specs})
        assert {(int(spec['seed']), spec['arm']) for spec in specs} == {(seed, arm) for seed in seeds for arm in ARMS[1:]}
        source_names = previous['method_names'].tolist()
        retained_names = REFERENCES+[f'original_s{seed}' for seed in seeds]
        retained_indices = [source_names.index(name) for name in retained_names]
        method_names = retained_names+[f'{spec["arm"]}_s{spec["seed"]}' for spec in specs]
        method_seeds = previous['method_seeds'][retained_indices].tolist()+[int(spec['seed']) for spec in specs]
        logits = np.empty((len(rows), len(method_names), len(OPERATIONS), 4), np.float32)
        norms = np.empty(logits.shape[:-1], np.float32)
        logits[:, :len(retained_names)] = previous['logits4'][selection][:, retained_indices]
        norms[:, :len(retained_names)] = previous['hidden_delta_norm'][selection][:, retained_indices]
        model_path = Path(cfg['model_local_dir'])
        for filename in ('config.json', 'tokenizer.json', 'model.safetensors'):
            work.checked(model_path/filename, 'Frozen Pythia model')
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
            dtype=torch.float32, attn_implementation='eager').to(work.device).eval()
        model.requires_grad_(False)
        model.config.use_cache = False
        assert model.config.num_hidden_layers == 16 and model.config.hidden_size == 2048
        label_ids = previous['label_ids']
        np.testing.assert_array_equal(label_ids, [tokenizer.encode(word, add_special_tokens=False)[0] for word in [' is', ' are', ' was', ' were']])
        output_weight = model.get_output_embeddings().weight[label_ids].detach()
        batch_size = int(cfg.get('batch_size', 32))
        module = model.gpt_neox.layers[15]
        contract = HookPointContract('gpt_neox.layers.15', 15, 'resid_post', 2048)
        work.environment = dict(python=sys.executable, python_version=platform.python_version(), torch=torch.__version__,
            transformers=transformers.__version__, numpy=np.__version__, dtype='float32',
            model_tail_matmul_precision='high', target_encoder_decoder_matmul_precision='highest', target_autocast=False,
            reference_current_hidden_precision='high', natural_training_cache_precision='highest')
        def tail(states):
            nonlocal tail_forwards
            assert torch.get_float32_matmul_precision() == 'high'
            result = []
            with torch.no_grad():
                for start in range(0, len(states), batch_size):
                    x = torch.as_tensor(states[start:start+batch_size], device=work.device)
                    result.append((model.gpt_neox.final_layer_norm(x)@output_weight.T).cpu().numpy())
            tail_forwards += len(states)
            return np.concatenate(result)
        def full_forward(records, delta):
            assert torch.get_float32_matmul_precision() == 'high'
            if time.perf_counter()-work.wall_start > cfg['budget_seconds']:
                raise TimeoutError('Functional block evaluation budget exhausted')
            inputs = tokenizer([row['text'] for row in records], add_special_tokens=False, padding=True, return_tensors='pt').to(work.device)
            last = inputs.attention_mask.sum(1)-1
            ix = torch.arange(len(records), device=work.device)
            def hook(module, arguments, output):
                state = extract_primary_hook_tensor(output, contract).clone()
                state[ix, last] += torch.as_tensor(delta, device=work.device)
                return replace_primary_hook_tensor(output, state, contract)
            handle = module.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    last_state = model.gpt_neox(**inputs, use_cache=False).last_hidden_state[ix, last]
                    result = (last_state@output_weight.T).cpu().numpy()
            finally:
                handle.remove()
            work.sequence_forwards += len(records)
            work.token_forwards += int(inputs.attention_mask.sum())
            return result
        current_baseline = tail(hidden)
        baseline_error = float(np.max(np.abs(current_baseline-baseline)))
        count = min(batch_size, len(rows))
        full_baseline_error = float(np.max(np.abs(full_forward(rows[:count], np.zeros_like(hidden[:count]))-baseline[:count])))
        assert max(baseline_error, full_baseline_error) <= 1e-4
        checks = dict(reference_tail_baseline_max_error=baseline_error, reference_full_baseline_max_error=full_baseline_error)
        donors = {factor: np.array([pair[factor] for pair in pairs]) for factor in ['number', 'time', 'joint']}
        checkpoint_identities, functional_diagnostics = [], {}
        for offset, spec in enumerate(specs, len(retained_names)):
            path = Path(spec['path'])
            for filename in ('cfg.json', 'sae.safetensors', 'functional_groups.json'):
                work.checked(path/filename, 'Frozen functional block checkpoint')
            torch.set_float32_matmul_precision('highest')
            try:
                sae, groups, metadata = load_checkpoint(path, str(work.device))
                sae.eval().requires_grad_(False)
                allocation = 'global' if spec['arm'] == 'global_supervised' else 'group'
                assert metadata['allocation'] == allocation
                assert [len(groups[key]) for key in ('number', 'time', 'rest')] == [256, 256, 7680]
                all_groups = torch.cat([groups[key] for key in ('number', 'time', 'rest')])
                assert torch.equal(torch.sort(all_groups).values, torch.arange(8192, device=work.device))
                codes = np.empty((len(rows), 8192), np.float32)
                with torch.no_grad(), torch.autocast(device_type=work.device.type, enabled=False):
                    for start in range(0, len(rows), batch_size):
                        x = torch.as_tensor(hidden[start:start+batch_size], device=work.device)
                        encoded = encode(sae, x, groups, allocation)
                        dense = torch.zeros((len(x), 8192), device=work.device).scatter_(1, encoded.top_indices, encoded.top_acts)
                        codes[start:start+len(x)] = dense.cpu().numpy()
                    deltas = []
                    for request, donor in OPERATIONS:
                        members = torch.cat([groups['number'], groups['time']]) if request == 'joint' else groups[request]
                        indices = members.cpu().numpy()
                        dz = codes[donors[donor]][:, indices]-codes[:, indices]
                        delta = torch.as_tensor(dz, device=work.device)@sae.W_dec[members]
                        deltas.append(delta.cpu().numpy())
                encoded_states += len(rows)
                group_activity = np.stack([(codes[:, groups[key].cpu().numpy()] > 0).sum(1)
                                           for key in ('number', 'time', 'rest')], axis=1)
                functional_diagnostics[method_names[offset]] = dict(
                    mean_active_by_group=dict(zip(('number', 'time', 'rest'), group_activity.mean(0).tolist())),
                    zero_active_fraction_by_group=dict(zip(('number', 'time', 'rest'), (group_activity == 0).mean(0).tolist())),
                    number_update_from_pure_time_mean_norm=float(np.linalg.norm(deltas[2], axis=1).mean()),
                    time_update_from_pure_number_mean_norm=float(np.linalg.norm(deltas[1], axis=1).mean()))
                checkpoint_identities.append(dict(**spec, weights_sha256=sha256(path/'sae.safetensors'),
                                                  groups_sha256=sha256(path/'functional_groups.json'), metadata=metadata))
                np.savez_compressed(work.run/(method_names[offset]+'_codes.npz'), codes=codes,
                                    number_members=groups['number'].cpu().numpy(), time_members=groups['time'].cpu().numpy(),
                                    group_activity=group_activity, group_names=np.array(['number', 'time', 'rest']))
                del sae
            finally:
                torch.set_float32_matmul_precision('high')
            for operation, delta in enumerate(deltas):
                logits[:, offset, operation] = tail(hidden+delta)
                norms[:, offset, operation] = np.linalg.norm(delta, axis=1)
            actual = full_forward(rows[:count], deltas[4][:count])
            error_value = float(np.max(np.abs(actual-logits[:count, offset, 4])))
            assert error_value <= 1e-4, (method_names[offset], error_value)
            checks[method_names[offset]] = error_value
            work.progress('FUNCTIONAL_BLOCK_EXECUTED', method=method_names[offset], rows=len(rows), operations=len(OPERATIONS))
        effect = logits-logits.mean(-1, keepdims=True)-(baseline-baseline.mean(-1, keepdims=True))[:, None, None]
        operation_names = [request+'_from_'+donor for request, donor in OPERATIONS]
        panel_name = cfg.get('panel_name', reference_name)
        output_path = work.run/(panel_name+'_responses.npz')
        np.savez_compressed(output_path, logits4=logits, centered_effect=effect, baseline_logits4=baseline,
            hidden_delta_norm=norms, expected_label=previous['expected_label'][selection], method_names=np.array(method_names),
            method_seeds=np.array(method_seeds), operation_names=np.array(operation_names), label_ids=label_ids,
            row_ids=np.arange(len(rows)), source_teacher_delta=previous['source_teacher_delta'][selection])
        write(work.run/(panel_name+'_membership.json'), dict(rows=rows, pairs=pairs))
        write(work.run/'evaluation_index.json', {panel_name: dict(response_path=str(output_path),
            membership_path=str(work.run/(panel_name+'_membership.json')), rows=len(rows))})
        write(work.run/'checkpoint_identities.json', dict(checkpoints=checkpoint_identities))
        write(work.run/'functional_diagnostics.json', functional_diagnostics)
        write(work.run/'full_model_checks.json', checks)
        for method_index, name in enumerate(method_names):
            for operation, operation_name in enumerate(operation_names):
                records = []
                for row_index, row in enumerate(rows):
                    records.append(dict(run_id=cfg['run_id'], metric_version='multisite-v1',
                        kind='functional_block', task=panel_name, component=str(row['subject_id']), row_id=row_index,
                        method=name, mode='target_group', operation=operation_name, target_seed=method_seeds[method_index],
                        correct=bool(logits[row_index, method_index, operation].argmax() == previous['expected_label'][selection[row_index], operation])))
                with (work.run/'metrics.raw.jsonl').open('a', encoding='utf-8') as stream:
                    stream.writelines(json.dumps(record)+'\n' for record in records)
                work.metrics.extend(records)
            work.progress('METHOD_RECORDS_SAVED', method=name, methods_completed=method_index+1,
                          methods_total=len(method_names), records_saved=(method_index+1)*len(operation_names)*len(rows))
        work.checks.update(frozen_groups=True, target_only_group_operation=True, same_reference=True, actual_full_model_check=True)
    except Exception:
        error = traceback.format_exc()
    write(work.run/'execution_cost.json', dict(encoded_states=encoded_states, cached_tail_sequence_forwards=tail_forwards,
          full_model_sequence_forwards=work.sequence_forwards, token_forwards=work.token_forwards, consumer_fitting_steps=0))
    return work.finish(error)


def analyze(run, panel_name, output, bootstrap):
    if output.exists():
        raise FileExistsError(output)
    # 导入原分析函数的独立实例，只设置本单元允许的训练臂；原文件与旧输出保持原状。
    import importlib.util
    source = Path(__file__).with_name('analyze_grammar_material_support.py')
    spec = importlib.util.spec_from_file_location('functional_block_statistics', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ARMS = ARMS
    data = module.load(run, panel_name, None)
    count = len(data['subjects'])
    point = statistics(data, np.arange(count), individual=True)
    samples = {key: [] for key in point}
    rng = np.random.default_rng(9232611)
    for iteration in range(bootstrap):
        values = statistics(data, rng.integers(0, count, size=count), individual=True)
        for key, value in values.items():
            samples[key].append(value)
        if (iteration+1) % 250 == 0:
            print(dict(bootstrap_completed=iteration+1, bootstrap_total=bootstrap), flush=True)
    comparisons = {}
    for first, second in [('group_supervised', 'global_supervised'), ('group_supervised', 'original'),
                          ('global_supervised', 'original'), ('group_supervised', 'raw'), ('global_supervised', 'raw')]:
        for key in [key.removeprefix(first+'/') for key in point if key.startswith(first+'/')]:
            a, b = point[first+'/'+key], point[second+'/'+key]
            differences = [x-y if x is not None and y is not None else None for x, y in zip(samples[first+'/'+key], samples[second+'/'+key])]
            comparisons[first+'_minus_'+second+'/'+key] = dict(value=a-b if a is not None and b is not None else None, ci95=summarize(differences))
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), rows=len(data['rows']), subjects=data['subjects'],
        fixed_targets=data['targets'], statistics={key: dict(value=value, ci95=summarize(samples[key])) for key, value in point.items()},
        paired_contrasts=comparisons, primary='Two mixed-donor partial requests; matching source-factor centered-four-logit response energy; equal mean normalized squared error over fixed targets and factors, then square root',
        bootstrap=dict(draws=bootstrap, seed=9232611, unit='subject lexical block, shared across methods, targets and operations'),
        identities=dict(**data['identities'], analyzer=identity(Path(__file__)), shared_statistics=identity(source)))
    output.mkdir(parents=True)
    write_json(output/'RESULTS.json', result)
    arrays = data['arrays']
    np.savez_compressed(output/'row_results.npz', **{key: value for key, value in arrays.items() if key != 'source_teacher_delta'},
                        four_word_probability=data['probability'])
    lines = ['# 功能分组的共同donor部分调用', '', '| 方法 | nRMSE | 四词正确率 | 两边际联合成功率 |', '|---|---:|---:|---:|']
    for method in REFERENCES+ARMS:
        values = [point[method+'/'+key] for key in ('primary_nrmse', 'primary_four_word_correct', 'primary_joint_success')]
        lines.append('| '+method+' | '+' | '.join('不可归一化' if value is None else f'{value:.6f}' for value in values)+' |')
    lines += ['', '统计复用既有centered响应与subject block配对定义。原对照只读取冻结响应，新模型直接执行固定功能分组。', '']
    (output/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(dict(primary={method: point[method+'/primary_nrmse'] for method in REFERENCES+ARMS}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    evaluation = subparsers.add_parser('evaluate')
    evaluation.add_argument('--config', type=Path, required=True)
    analysis = subparsers.add_parser('analyze')
    analysis.add_argument('--run', type=Path, required=True)
    analysis.add_argument('--panel', default='development')
    analysis.add_argument('--output', type=Path, required=True)
    analysis.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.command == 'evaluate':
        return evaluate(args.config)
    analyze(args.run, args.panel, args.output, args.bootstrap)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
