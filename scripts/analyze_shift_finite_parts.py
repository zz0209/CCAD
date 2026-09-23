import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_shift_member_responses import identity, read_json, write_json
from analyze_shift_vector_reuse import summarize


METHODS = ['source', 'joint_vector', 'independent_vector', 'joint_mixed', 'native', 'raw']
OPERATIONS = ['0', 'P', 'N', 'W', 'PN', 'PW', 'NW', 'PNW']
COMBINATIONS = [4, 5, 6, 7]
EFFECTS = [(name, index, 0) for index, name in enumerate(OPERATIONS[1:], 1)] + [
    ('P_minus_N', 1, 2), ('W_given_P', 5, 1)]


def mean_equal(values):
    return float(np.mean(values)) if all(value is not None for value in values) else None


def root_ratio(numerator, denominator):
    return float(np.sqrt(numerator / denominator)) if denominator > 0 else None


def load_inputs(runs, panel_name, panel_path, endpoint, heads_directory, original_head):
    panel = read_json(panel_path)
    row_map = {row['document_sha256']: row for row in panel['rows']}
    assert len(row_map) == len(panel['rows'])
    targets, input_files, capacities = [], [], []
    for run in runs:
        assert read_json(run / 'status.json')['status'] == 'PASS'
        path = run / 'evaluation' / f'{panel_name}.npz'
        with np.load(path) as raw:
            assert raw['method_names'].tolist() == METHODS
            assert raw['operation_names'].tolist() == OPERATIONS
            documents = raw['document_sha256'].astype(str)
            pooled = raw['pooled512'].astype(np.float64)
            baseline = raw['baseline_pooled512'].astype(np.float64)
            logits = raw['logits'].astype(np.float64)
            assert pooled.shape == (len(documents), len(METHODS), len(OPERATIONS), 512)
            assert np.isfinite(pooled).all()
            for method_index in range(len(METHODS)):
                np.testing.assert_array_equal(pooled[:, method_index, 0], baseline)
            arrays = {key: raw[key].copy() for key in ('genders', 'labels', 'professions') if key in raw}
        if targets:
            np.testing.assert_array_equal(documents, targets[0]['documents'])
        else:
            assert set(documents) == set(row_map)
        rows = [row_map[value] for value in documents]
        expected = np.array([row['gender'] for row in rows])
        np.testing.assert_array_equal(arrays['genders'], expected)
        metadata_key = 'professions' if endpoint == 'later' else 'labels'
        row_key = 'profession' if endpoint == 'later' else 'label'
        np.testing.assert_array_equal(arrays[metadata_key], [row[row_key] for row in rows])
        config = read_json(run / 'config.resolved.json')
        targets.append(dict(seed=config['target_seed'], documents=documents, pooled=pooled,
                            original_logits=logits, metadata=arrays))
        input_files.append(dict(run=run.as_posix(), arrays=identity(path),
                                config=identity(run / 'config.resolved.json'),
                                membership=identity(run / 'evaluation' / f'{panel_name}_membership.json')))
        capacities.append(dict(target_seed=config['target_seed'],
                               values=read_json(run / 'relation_summary.json'),
                               identity=identity(run / 'relation_summary.json'),
                               observed=read_json(run / 'evaluation' / f'{panel_name}_capacity.json'),
                               observed_identity=identity(run / 'evaluation' / f'{panel_name}_capacity.json')))
    assert len({target['seed'] for target in targets}) == len(targets)
    with np.load(original_head) as raw:
        original_weight = raw['weight'].astype(np.float64).reshape(512)
        original_bias = float(raw['bias'].reshape(-1)[0])
    heads = []
    if endpoint == 'later':
        for task in panel['tasks']:
            path = heads_directory / f'none__full__{task["name"]}__probe42.npz'
            with np.load(path) as raw:
                heads.append(dict(task, weight=raw['weight'].astype(np.float64).reshape(512),
                                  bias=float(raw['bias'].reshape(-1)[0]), identity=identity(path)))
        assert len(heads) == 4
    else:
        heads.append(dict(name='professor_nurse', weight=original_weight, bias=original_bias,
                          identity=identity(original_head)))
    category = targets[0]['metadata']['professions' if endpoint == 'later' else 'labels']
    genders = targets[0]['metadata']['genders']
    strata = [np.flatnonzero((category == value) & (genders == gender))
              for value in sorted(set(category)) for gender in (0, 1)]
    strata = [indices for indices in strata if len(indices)]
    for head in heads:
        mask = np.isin(category, [head['negative'], head['positive']]) if endpoint == 'later' else np.ones(len(category), bool)
        head['cells'] = [indices for indices in strata if mask[indices[0]]]
        assert len(head['cells']) == 4 if endpoint == 'later' else len(head['cells']) > 0
        head['labels'] = category == head['positive'] if endpoint == 'later' else category.astype(bool)
        head['logits'] = [target['pooled'] @ head['weight'] + head['bias'] for target in targets]
        head['cache'] = []
        for logits in head['logits']:
            response = logits - logits[:, :, :1]
            effect = np.stack([logits[:, :, first] - logits[:, :, second] for _, first, second in EFFECTS], axis=-1)
            head['cache'].append(dict(response_squared=response ** 2,
                                     error_squared=(response-response[:, :1]) ** 2,
                                     effect=effect, effect_squared=effect ** 2,
                                     effect_error_squared=(effect-effect[:, :1]) ** 2,
                                     correct=(logits > 0) == head['labels'][:, None, None]))
    projection_errors = []
    for target in targets:
        projected = target['pooled'] @ original_weight + original_bias
        projection_errors.append(float(np.max(np.abs(projected - target['original_logits']))))
        delta = target['pooled'] - target['pooled'][:, :, :1]
        error = delta - delta[:, :1]
        parallel = (error @ original_weight) ** 2 / np.dot(original_weight, original_weight)
        target['hidden_squared_error'] = np.sum(error ** 2, axis=-1)
        target['hidden_parallel_squared'] = parallel
        target['hidden_perpendicular_squared'] = target['hidden_squared_error'] - parallel
    return panel, targets, heads, strata, input_files, capacities, projection_errors


def statistics(targets, heads, strata, draws=None, details=False):
    overall_rows, per_head = {}, {}
    # 每格先求文档均值，随后各格等权；同一抽样映射供全部目标和head使用。
    samples = {int(cell[0]): cell if draws is None else draws[int(cell[0])] for cell in strata}
    for target_index, target in enumerate(targets):
        for head in heads:
            selected = [samples[int(cell[0])] for cell in head['cells']]
            def average(array):
                return np.mean([array[indices].mean(axis=0) for indices in selected], axis=0)
            cache = head['cache'][target_index]
            moments = {key: average(value) for key, value in cache.items()}
            source_energy = float(moments['response_squared'][0, COMBINATIONS].mean())
            mse = moments['error_squared'][:, COMBINATIONS].mean(axis=1)
            rms = np.sqrt(moments['response_squared'])
            correct = cache['correct']
            accuracy = moments['correct']
            worst = np.min([correct[indices].mean(axis=0) for indices in selected], axis=0)
            for method_index, method in enumerate(METHODS):
                row = dict(primary_normalized_squared=float(mse[method_index] / source_energy) if source_energy > 0 else None,
                           primary_rmse=float(np.sqrt(mse[method_index])),
                           primary_source_effect_rms=float(np.sqrt(source_energy)),
                           combination_accuracy=float(accuracy[method_index, COMBINATIONS].mean()),
                           P_minus_N_accuracy_difference=float(accuracy[method_index, 1]-accuracy[method_index, 2]),
                           P_minus_N_accuracy_difference_error=float(accuracy[method_index, 1]-accuracy[method_index, 2]-accuracy[0, 1]+accuracy[0, 2]))
                for operation_index, operation in enumerate(OPERATIONS):
                    row[operation + '/accuracy'] = float(accuracy[method_index, operation_index])
                    row[operation + '/worst_group_accuracy'] = float(worst[method_index, operation_index])
                    row[operation + '/effect_rms'] = float(rms[method_index, operation_index])
                for effect_index, (name, _, _) in enumerate(EFFECTS):
                    square = float(moments['effect_error_squared'][method_index, effect_index])
                    energy = float(moments['effect_squared'][0, effect_index])
                    row[name + '/normalized_squared'] = square / energy if energy > 0 else None
                    row[name + '/rmse'] = float(np.sqrt(square))
                    row[name + '/source_effect_rms'] = float(np.sqrt(energy))
                    row[name + '/actual_effect_rms'] = float(np.sqrt(moments['effect_squared'][method_index, effect_index]))
                    row[name + '/signed_mean'] = float(moments['effect'][method_index, effect_index])
                overall_rows.setdefault(method, []).append(row)
                if details:
                    detail = dict(row)
                    for key, value in row.items():
                        if key.endswith('normalized_squared'):
                            detail[key.replace('normalized_squared', 'nrmse')] = float(np.sqrt(value)) if value is not None else None
                    per_head[f't{target["seed"]}/{head["name"]}/{method}'] = detail
    overall = {}
    for method, rows in overall_rows.items():
        for key in rows[0]:
            value = mean_equal([row[key] for row in rows])
            if key.endswith('normalized_squared'):
                key = key.replace('normalized_squared', 'nrmse')
                value = float(np.sqrt(value)) if value is not None else None
            overall[method + '/' + key] = value
    for method_index, method in enumerate(METHODS):
        for key in ('hidden_squared_error', 'hidden_parallel_squared', 'hidden_perpendicular_squared'):
            overall[method + '/' + key] = float(np.mean([
                np.mean([target[key][samples[int(cell[0])], method_index][:, COMBINATIONS].mean()
                         for cell in strata]) for target in targets]))
    return overall, per_head


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    panel, targets, heads, strata, inputs, capacities, projection_errors = load_inputs(
        args.runs, args.panel_name, args.panel, args.endpoint, args.heads_directory, args.original_head)
    points, per_head = statistics(targets, heads, strata, details=True)
    samples = {key: [] for key in points}
    rng = np.random.default_rng(9232417)
    for iteration in range(args.bootstrap):
        draw = {int(cell[0]): rng.choice(cell, len(cell), replace=True) for cell in strata}
        values, _ = statistics(targets, heads, strata, draw)
        for key, value in values.items():
            samples[key].append(value)
        if (iteration + 1) % 100 == 0:
            print(dict(bootstrap_completed=iteration + 1, bootstrap_total=args.bootstrap), flush=True)
    contrasts = {}
    for other in METHODS[2:]:
        for suffix in [key.removeprefix('joint_vector/') for key in points if key.startswith('joint_vector/')]:
            first, second = 'joint_vector/' + suffix, other + '/' + suffix
            a, b = points[first], points[second]
            differences = [x-y if x is not None and y is not None else None
                           for x, y in zip(samples[first], samples[second])]
            contrasts['joint_vector_minus_' + other + '/' + suffix] = dict(
                value=a-b if a is not None and b is not None else None, ci95=summarize(differences))
    category_name = 'professions' if args.endpoint == 'later' else 'labels'
    metadata = targets[0]['metadata']
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), endpoint=args.endpoint,
                  panel_name=args.panel_name, documents=len(targets[0]['documents']),
                  document_exposure=args.document_exposure, fixed_targets=[target['seed'] for target in targets],
                  operations=OPERATIONS, methods=METHODS, primary_operations=[OPERATIONS[i] for i in COMBINATIONS],
                  primary='Equal mean of per-head and fixed-target combination error mean square divided by that head source response energy pooled over PN/PW/NW/PNW; square root after aggregation',
                  statistics={key: dict(value=value, ci95=summarize(samples[key])) for key, value in points.items()},
                  per_head=per_head, paired_contrasts=contrasts, relation_capacity=capacities,
                  original_head_projection_max_absolute_error_by_target=projection_errors,
                  weighting=dict(cells={f'{category_name}{metadata[category_name][cell[0]]}_gender{metadata["genders"][cell[0]]}': len(cell) for cell in strata},
                                 head='Equal weight for observed class/profession by gender cells; four cells per later head',
                                 aggregation='Fixed heads and targets receive equal weight; each contrast has its own source scale'),
                  bootstrap=dict(draws=args.bootstrap, seed=9232417,
                                 unit='Documents sampled within each observed stratum at its actual n; common indices across operations, methods, heads and targets'),
                  heads=[{key: value for key, value in head.items() if key not in ('weight', 'bias', 'cells', 'labels', 'logits', 'cache')} for head in heads],
                  identities=dict(panel=identity(args.panel), original_head=identity(args.original_head),
                                  analyzer=identity(Path(__file__)), inputs=inputs),
                  zero_source_energy='Normalized value is unavailable when source energy is zero; raw RMSE remains available')
    args.output.mkdir(parents=True)
    write_json(args.output / 'FINITE_PARTS_RESULTS.json', result)
    arrays = dict(document_sha256=targets[0]['documents'], method_names=METHODS, operation_names=OPERATIONS,
                  target_seeds=[target['seed'] for target in targets], **metadata)
    for target in targets:
        arrays[f't{target["seed"]}__pooled512'] = target['pooled'].astype(np.float32)
        arrays[f't{target["seed"]}__original_logits'] = target['original_logits']
    for head in heads:
        arrays[head['name'] + '__weight'] = head['weight']
        arrays[head['name'] + '__bias'] = head['bias']
    np.savez_compressed(args.output / 'finite_parts_arrays.npz', **arrays)
    lines = ['# 多部分功能关系的真实组合使用', '',
             f'{len(targets[0]["documents"])}篇文档，{len(heads)}个冻结head，固定目标{result["fixed_targets"]}。',
             f'文档使用身份为{args.document_exposure}。', '',
             '| 方法 | 组合nRMSE | 组合准确率 | P−N nRMSE | W在P之后nRMSE |', '|---|---:|---:|---:|---:|']
    for method in METHODS:
        values = [points[method + '/' + suffix] for suffix in ('primary_nrmse', 'combination_accuracy', 'P_minus_N/nrmse', 'W_given_P/nrmse')]
        lines.append('| ' + method + ' | ' + ' | '.join('不可归一化' if value is None else f'{value:.6f}' for value in values) + ' |')
    lines += ['', '主要组合为PN、PW、NW、PNW。每个head按source四组合共同能量归一化，随后对固定head和目标等权平均平方误差并开方。',
              '单部分与条件作用使用各自source尺度。准确率基于实际完整程序状态，文档在其实际分组内配对重采样。',
              'JSON保存所有方法的配对差、逐head实际数值、作用幅度与容量统计；NPZ保存实际完整程序状态和冻结读出。', '']
    (args.output / 'FINITE_PARTS_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(dict(output=args.output.as_posix(), primary={method: points[method+'/primary_nrmse'] for method in METHODS}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--panel-name', required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--endpoint', choices=['later', 'original'], required=True)
    parser.add_argument('--heads-directory', type=Path, default=Path('runs/IR04_shift_consumer_seed4_v1_20260916'))
    parser.add_argument('--original-head', type=Path, default=Path('runs/REFORM_R58_shift_source_development_v2_20260915/probe.npz'))
    parser.add_argument('--document-exposure', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
