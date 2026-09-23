import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_shift_member_responses import identity, read_json, write_json
from analyze_shift_vector_reuse import summarize


OPERATIONS = ['number_from_number', 'time_from_number', 'number_from_time',
              'time_from_time', 'number_from_joint', 'time_from_joint', 'joint_from_joint']
MAIN = [4, 5]
FACTOR_SCALE = [4, 5, 4, 5, 4, 5, 6]
ARMS = ['original', 'natural_replay', 'correlated', 'factorial']
REFERENCES = ['source_teacher', 'source_mask', 'source_full_sae', 'raw']


def normalized(square, energy):
    return float(np.sqrt(square/energy)) if energy > 0 else None


def load(run, panel_name, target_seeds):
    assert read_json(run/'status.json')['status'] == 'PASS'
    index = read_json(run/'evaluation_index.json')[panel_name]
    response_path, membership_path = Path(index['response_path']), Path(index['membership_path'])
    panel = read_json(membership_path)
    rows = panel['rows']
    with np.load(response_path) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    assert arrays['operation_names'].tolist() == OPERATIONS
    np.testing.assert_array_equal(arrays['row_ids'], np.arange(len(rows)))
    logits = arrays['logits4'].astype(np.float64)
    effects = arrays['centered_effect'].astype(np.float64)
    baseline = arrays['baseline_logits4'].astype(np.float64)
    assert logits.shape == effects.shape == (len(rows), len(arrays['method_names']), len(OPERATIONS), 4)
    assert np.isfinite(logits).all() and np.isfinite(effects).all()
    reconstructed = arrays['logits4']-arrays['logits4'].mean(-1, keepdims=True)
    reconstructed -= (arrays['baseline_logits4']-arrays['baseline_logits4'].mean(-1, keepdims=True))[:, None, None]
    np.testing.assert_array_equal(reconstructed, arrays['centered_effect'])
    expected = arrays['expected_label']
    numbers, times = np.array([row['number'] for row in rows]), np.array([row['time'] for row in rows])
    for operation, flips in enumerate([(1, 0), (0, 0), (0, 0), (0, 1), (1, 0), (0, 1), (1, 1)]):
        np.testing.assert_array_equal(expected[:, operation], (numbers ^ flips[0])+2*(times ^ flips[1]))
    context_cells = {}
    for row in rows:
        context = (row['subject_id'], row['cue_id'], row['template'])
        context_cells.setdefault(context, []).append((row['number'], row['time'], row['attractor']))
    assert all(len(cells) == 8 and len(set(cells)) == 8 for cells in context_cells.values())
    subjects = list(dict.fromkeys(row['subject_id'] for row in rows))
    contexts_by_subject = [{(cue, template) for subject, cue, template in context_cells if subject == value} for value in subjects]
    assert all(contexts == contexts_by_subject[0] for contexts in contexts_by_subject)
    subject_ids = np.array([row['subject_id'] for row in rows])
    blocks = [np.flatnonzero(subject_ids == subject) for subject in subjects]
    method_names = arrays['method_names'].tolist()
    assert method_names[:4] == REFERENCES
    groups = {name: [method_names.index(name)] for name in REFERENCES}
    method_metadata = {}
    for method_index, name in enumerate(method_names[4:], 4):
        arm, seed = name.rsplit('_s', 1)
        seed = int(seed)
        assert arm in ARMS and seed == int(arrays['method_seeds'][method_index])
        if target_seeds is None or seed in target_seeds:
            groups.setdefault(arm, []).append(method_index)
            method_metadata[name] = dict(arm=arm, seed=seed, method_index=method_index)
    assert all(arm in groups for arm in ARMS)
    actual_targets = sorted({entry['seed'] for entry in method_metadata.values()})
    if target_seeds is not None:
        assert actual_targets == sorted(target_seeds)
    for arm in ARMS:
        assert sorted(int(arrays['method_seeds'][index]) for index in groups[arm]) == actual_targets
    probability = np.exp(logits-logits.max(-1, keepdims=True))
    probability /= probability.sum(-1, keepdims=True)
    number_probability = probability[..., [1, 3]].sum(-1)
    time_probability = probability[..., [2, 3]].sum(-1)
    number_correct = (number_probability > .5) == (expected[:, None] % 2)
    time_correct = (time_probability > .5) == (expected[:, None] // 2)
    source = effects[:, 0]
    squared_error = np.square(effects-source[:, None]).mean(-1)
    moments = dict(squared_error=squared_error, source_squared=np.square(source).mean(-1),
                   actual_squared=np.square(effects).mean(-1),
                   four_word_correct=logits.argmax(-1) == expected[:, None],
                   number_correct=number_correct, time_correct=time_correct,
                   joint_success=number_correct & time_correct,
                   number_probability=number_probability, time_probability=time_probability,
                   delta_norm=arrays['hidden_delta_norm'].astype(np.float64), probability=probability)
    block_moments = {key: np.stack([value[indices].mean(axis=0) for indices in blocks]) for key, value in moments.items()}
    teacher_delta = arrays['source_teacher_delta']
    zero_teacher = np.all(teacher_delta == 0, axis=(0, 2))
    assert zero_teacher[1] and zero_teacher[2]
    return dict(panel=panel, arrays=arrays, rows=rows, subjects=subjects, groups=groups,
                method_metadata=method_metadata, targets=actual_targets, blocks=blocks,
                block_moments=block_moments, zero_teacher=zero_teacher,
                identities=dict(responses=identity(response_path), membership=identity(membership_path),
                                index=identity(run/'evaluation_index.json'), config=identity(run/'config.resolved.json')),
                baseline=baseline, probability=probability, row_moments=moments)


def statistics(data, sampled_blocks, individual=False):
    moments = {key: value[sampled_blocks].mean(axis=0) for key, value in data['block_moments'].items()}
    source_energy = moments['source_squared']
    groups = dict(data['groups'])
    if individual:
        groups.update({name: [entry['method_index']] for name, entry in data['method_metadata'].items()})
    result = {}
    for name, method_indices in groups.items():
        mse = moments['squared_error'][method_indices].mean(axis=0)
        normalized_primary = [mse[operation]/source_energy[operation] if source_energy[operation] > 0 else None for operation in MAIN]
        result[name+'/primary_nrmse'] = float(np.sqrt(np.mean(normalized_primary))) if all(value is not None for value in normalized_primary) else None
        for key in ('four_word_correct', 'number_correct', 'time_correct', 'joint_success'):
            result[name+'/primary_'+key] = float(moments[key][method_indices][:, MAIN].mean())
        for operation, operation_name in enumerate(OPERATIONS):
            prefix = name+'/'+operation_name+'/'
            result[prefix+'rmse'] = float(np.sqrt(mse[operation]))
            result[prefix+'source_effect_rms'] = float(np.sqrt(source_energy[operation]))
            result[prefix+'actual_effect_rms'] = float(np.sqrt(moments['actual_squared'][method_indices, operation].mean()))
            result[prefix+'nrmse_factor_scale'] = normalized(mse[operation], source_energy[FACTOR_SCALE[operation]])
            result[prefix+'nrmse_own_scale'] = normalized(mse[operation], source_energy[operation]) if not data['zero_teacher'][operation] else None
            for key in ('four_word_correct', 'number_correct', 'time_correct', 'joint_success',
                        'number_probability', 'time_probability', 'delta_norm'):
                result[prefix+key] = float(moments[key][method_indices, operation].mean())
            for word, value in zip(('is', 'are', 'was', 'were'), moments['probability'][method_indices, operation].mean(axis=0)):
                result[prefix+'probability_'+word] = float(value)
    return result


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    data = load(args.run, args.panel, args.target_seeds)
    count = len(data['subjects'])
    point = statistics(data, np.arange(count), individual=True)
    samples = {key: [] for key in point}
    rng = np.random.default_rng(9232519)
    for iteration in range(args.bootstrap):
        values = statistics(data, rng.integers(0, count, size=count), individual=True)
        for key, value in values.items():
            samples[key].append(value)
        if (iteration+1) % 250 == 0:
            print(dict(bootstrap_completed=iteration+1, bootstrap_total=args.bootstrap, lexical_blocks=count), flush=True)
    contrasts = {}
    suffixes = [key.removeprefix('factorial/') for key in point if key.startswith('factorial/')]
    for other in ('correlated', 'natural_replay', 'original', 'raw', 'source_mask'):
        for suffix in suffixes:
            first, second = 'factorial/'+suffix, other+'/'+suffix
            a, b = point[first], point[second]
            differences = [x-y if x is not None and y is not None else None for x, y in zip(samples[first], samples[second])]
            contrasts['factorial_minus_'+other+'/'+suffix] = dict(value=a-b if a is not None and b is not None else None,
                                                                  ci95=summarize(differences))
    for seed in data['targets']:
        for other in ('correlated', 'natural_replay', 'original'):
            for suffix in ('primary_nrmse', 'primary_four_word_correct', 'primary_joint_success'):
                first, second = f'factorial_s{seed}/{suffix}', f'{other}_s{seed}/{suffix}'
                a, b = point[first], point[second]
                differences = [x-y if x is not None and y is not None else None for x, y in zip(samples[first], samples[second])]
                contrasts[f'factorial_minus_{other}_s{seed}/{suffix}'] = dict(value=a-b if a is not None and b is not None else None,
                                                                            ci95=summarize(differences))
    quality = None
    if args.training_run:
        source = args.training_run/'final_checkpoints.json'
        records = read_json(source)['checkpoints']
        quality = dict(identity=identity(source), records=[{key: item[key] for key in ('seed', 'arm', 'step', 'path', 'weights_sha256', 'quality')}
                                                         for item in records],
                       scope='Material reconstruction quality and functional use are separate measurements')
    arrays = data['arrays']
    baseline_prob = np.exp(data['baseline']-data['baseline'].max(-1, keepdims=True))
    baseline_prob /= baseline_prob.sum(-1, keepdims=True)
    expected_base = np.array([row['number']+2*row['time'] for row in data['rows']])
    baseline_correct = data['baseline'].argmax(-1) == expected_base
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), run=str(args.run), panel=args.panel,
                  rows=len(data['rows']), subjects=data['subjects'], fixed_target_seeds=data['targets'],
                  statistics={key: dict(value=value, ci95=summarize(samples[key])) for key, value in point.items()},
                  paired_contrasts=contrasts, quality=quality,
                  baseline=dict(four_word_accuracy=float(baseline_correct.mean()),
                                mean_four_word_probability=baseline_prob.mean(0).tolist()),
                  primary_operations=[OPERATIONS[index] for index in MAIN],
                  primary='Center each four-logit vector within its row, subtract centered baseline, divide response error mean square by the matching source factor response mean square, average equally over two factors and fixed targets, then take square root',
                  correct='Four-word argmax must match the requested factor change while preserving the other factor; marginal number/time correctness and their joint success are separate outcomes',
                  zero_teacher='Pure-donor requests for the unchanged factor have zero teacher updates; raw RMSE and matching main-factor normalized RMSE are reported, own zero-response normalization is unavailable',
                  zero_teacher_operations=[operation for operation, zero in zip(OPERATIONS, data['zero_teacher']) if zero],
                  weighting=dict(subject_counts={str(subject): len(block) for subject, block in zip(data['subjects'], data['blocks'])},
                                 subjects='Equal lexical blocks', within_subject='Complete balanced cues, templates and eight factor cells',
                                 targets='Fixed source-to-target directions receive equal weight; no seed-population inference'),
                  bootstrap=dict(draws=args.bootstrap, seed=9232519, unit='subject_id lexical block',
                                 pairing='Every draw retains all rows, cues, templates, factor cells, methods, operations and fixed targets within each selected block'),
                  identities=dict(**data['identities'], analyzer=identity(Path(__file__))))
    args.output.mkdir(parents=True)
    write_json(args.output/'RESULTS.json', result)
    np.savez_compressed(args.output/'row_results.npz', **{key: value for key, value in arrays.items() if key != 'source_teacher_delta'},
                        subject_ids=np.array([str(row['subject_id']) for row in data['rows']]),
                        four_word_probability=data['probability'],
                        squared_response_error=data['row_moments']['squared_error'],
                        number_correct=data['row_moments']['number_correct'],
                        time_correct=data['row_moments']['time_correct'],
                        joint_success=data['row_moments']['joint_success'],
                        four_word_correct=data['row_moments']['four_word_correct'])
    lines = ['# 训练材料支持与共同donor功能部分调用', '',
             f'{len(data["rows"])}条输入，{count}个subject词汇block，固定目标{data["targets"]}。', '',
             '| 方法 | 主要nRMSE | 四词正确率 | 两边际联合成功率 |', '|---|---:|---:|---:|']
    for method in REFERENCES+ARMS:
        values = [point[method+'/'+suffix] for suffix in ('primary_nrmse', 'primary_four_word_correct', 'primary_joint_success')]
        lines.append('| '+method+' | '+' | '.join('不可归一化' if value is None else f'{value:.6f}' for value in values)+' |')
    lines += ['', '| 请求 | 方法 | factor尺度nRMSE | 四词正确率 | number正确率 | time正确率 |', '|---|---|---:|---:|---:|---:|']
    for operation in OPERATIONS:
        for method in REFERENCES+ARMS:
            values = [point[method+'/'+operation+'/'+suffix] for suffix in ('nrmse_factor_scale', 'four_word_correct', 'number_correct', 'time_correct')]
            lines.append('| '+operation+' | '+method+' | '+' | '.join('不可归一化' if value is None else f'{value:.6f}' for value in values)+' |')
    lines += ['', '主要比较仅使用同时改变两个因素的donor上的两个部分请求。纯因素donor与联合请求分别报告。',
              '误差基于四词中心化响应，准确率使用实际四词选择。配对区间按subject词汇block共同重采样，固定目标与逐行观察均不充当独立seed。',
              '原逐行数组保存在row_results.npz，来源身份及各请求、目标、方法的作用幅度和配对差保存在RESULTS.json。', '']
    (args.output/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(dict(output=str(args.output), primary={method: point[method+'/primary_nrmse'] for method in REFERENCES+ARMS}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--panel', required=True)
    parser.add_argument('--target-seeds', type=int, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--training-run', type=Path)
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
