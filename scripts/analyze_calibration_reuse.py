import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_shared_action_rule import read_run
from analyze_shared_calibration_structure import condition, condition_metrics, digest, rms, save_csv


ROOT = Path('D:/CCAD_Storage/runs/calibration_reuse_20260923')
OLD = Path('D:/CCAD_Storage/runs/shared_rule_20260922/SR20_GPT2_SHARED_TARGET23_DEV_20260922_SOURCE_COLUMNS')
METHODS = ['initial', 'source_columns_lr0001', 'source_columns_gain', 'program', 'readout_initial']
COMPARISONS = [('source_columns_lr0001', 'initial'), ('source_columns_gain', 'source_columns_lr0001'),
               ('source_columns_lr0001', 'program'), ('source_columns_lr0001', 'readout_initial')]


def load(path):
    config, panel, order, arrays, schedule, identity = read_run(path)
    identity.update(config_sha256=digest(path/'config.resolved.json'), panel_sha256=digest(path/'panel.json'))
    print(json.dumps(dict(verified_run=path.name, records=identity['records_verified'])), flush=True)
    return dict(path=path, config=config, panel=panel, order=order, arrays=arrays,
                schedule=schedule, identity=identity)


def series(run, target, method):
    return run['arrays'][f't{target}_{method}']


def fit_quality(truth, predicted):
    truth, predicted = truth.ravel().astype(float), predicted.ravel().astype(float)
    error = predicted-truth
    energy = float(truth@truth)
    tc, pc = truth-truth.mean(), predicted-predicted.mean()
    denominator = float(np.linalg.norm(tc)*np.linalg.norm(pc))
    return dict(n=len(truth), truth_rms=rms(truth), predicted_rms=rms(predicted), rmse=rms(error),
                r2_relative_zero=1-float(error@error)/energy if energy > 0 else None,
                pearson=float(tc@pc)/denominator if denominator > 0 else None,
                mean_error=float(error.mean()))


def primary_values(directions, draws, order):
    result = np.empty((len(directions), len(METHODS)))
    for di, (run, target) in enumerate(directions):
        query_ids = [order.index(name) for name in run['config']['primary_requests']]
        source = series(run, target, 'source')
        clean = series(run, target, 'none')
        task_scores = []
        for ids in draws.values():
            reference = source[np.ix_(query_ids, ids)]
            energy = float(np.square(reference-clean[np.ix_(query_ids, ids)]).sum())
            assert energy > 0
            task_scores.append([np.sqrt(np.square(series(run, target, method)[np.ix_(query_ids, ids)]-reference).sum()/energy)
                                for method in METHODS])
        result[di] = np.mean(task_scores, axis=0)
    return result


def source_average(values, sources):
    return np.mean([values[[i for i, source in enumerate(sources) if source == chosen]].mean(0)
                    for chosen in dict.fromkeys(sources)], axis=0)


def response_analysis(directions, bootstrap, output_dir):
    reference = directions[0][0]
    order, panel = reference['order'], reference['panel']
    tasks = reference['config']['tasks']
    groups = {task: np.array([i for i, row in enumerate(panel['rows']) if row['task'] == task]) for task in tasks}
    for run, target in directions:
        assert run['panel']['rows'] == panel['rows'] and run['order'] == order
        assert run['config']['primary_requests'] == reference['config']['primary_requests']
        assert run['config']['tasks'] == tasks
    sources = [run['config'].get('source_seed', 1) for run, target in directions]
    point = primary_values(directions, groups, order)
    rng = np.random.default_rng(9232026)
    samples = np.empty((bootstrap, len(directions), len(METHODS)))
    overall_samples = np.empty((bootstrap, len(METHODS)))
    for iteration in range(bootstrap):
        draws = {task: rng.choice(ids, len(ids), replace=True) for task, ids in groups.items()}
        samples[iteration] = primary_values(directions, draws, order)
        overall_samples[iteration] = source_average(samples[iteration], sources)
        if (iteration+1) % 500 == 0:
            print(json.dumps(dict(bootstrap_completed=iteration+1, bootstrap_total=bootstrap)), flush=True)
    details, unit_rows = {}, []
    for di, (run, target) in enumerate(directions):
        source_seed = sources[di]
        key = f's{source_seed}_t{target}'
        truth = series(run, target, 'source')
        clean = series(run, target, 'none')
        source_condition = condition(truth, order)
        by_task = {}
        for task, ids in groups.items():
            qi = [order.index(name) for name in run['config']['primary_requests']]
            energy = float(np.square((truth-clean)[np.ix_(qi, ids)]).sum())
            by_task[task] = dict(rows=len(ids), source_effect_rms=rms((truth-clean)[np.ix_(qi, ids)]), methods={})
            for method in METHODS:
                values = series(run, target, method)
                predicted_condition = condition(values, order)
                by_task[task]['methods'][method] = dict(
                    primary_nrmse=float(np.sqrt(np.square((values-truth)[np.ix_(qi, ids)]).sum()/energy)),
                    conditional=condition_metrics(source_condition[ids], predicted_condition[ids]))
                for ri in ids:
                    for query_index, query in enumerate(order):
                        unit_rows.append(dict(source_seed=source_seed, target_seed=target, task=task,
                            row_id=panel['rows'][ri]['row_id'], request=query, method=method,
                            source_effect=float(truth[query_index, ri]-clean[query_index, ri]),
                            target_effect=float(values[query_index, ri]-clean[query_index, ri]),
                            margin_error=float(values[query_index, ri]-truth[query_index, ri])))
        comparisons = {}
        for first, second in COMPARISONS:
            a, b = METHODS.index(first), METHODS.index(second)
            comparisons[first+'_minus_'+second] = dict(mean=float(point[di, a]-point[di, b]),
                ci95=np.quantile(samples[:, di, a]-samples[:, di, b], [.025, .975]).tolist())
        details[key] = dict(by_task=by_task, primary={method: dict(mean=float(point[di, mi]),
            ci95=np.quantile(samples[:, di, mi], [.025, .975]).tolist()) for mi, method in enumerate(METHODS)},
            comparisons=comparisons)
    overall = source_average(point, sources)
    comparisons = {}
    for first, second in COMPARISONS:
        a, b = METHODS.index(first), METHODS.index(second)
        comparisons[first+'_minus_'+second] = dict(mean=float(overall[a]-overall[b]),
            ci95=np.quantile(overall_samples[:, a]-overall_samples[:, b], [.025, .975]).tolist())
    save_csv(output_dir/'REUSE_RESPONSE_UNITS.csv', unit_rows)
    return dict(directions=details, source_equal_primary={method: dict(mean=float(overall[mi]),
        ci95=np.quantile(overall_samples[:, mi], [.025, .975]).tolist()) for mi, method in enumerate(METHODS)},
        source_equal_comparisons=comparisons, bootstrap=bootstrap, bootstrap_seed=9232026,
        resampling='每个语法内配对句子重采样，所有请求、方法、来源和目标共享抽样。字典保持固定，区间不外推seed总体。',
        aggregation='先等权平均每方向的三个语法，再在各source内平均方向，最后等权平均source。',
        response_rows=len(unit_rows))


def load_capture(run, target, method, identities):
    path = run['path']/f't{target}_{method}_hidden_execution.npz'
    with np.load(path) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    assert values['completed'].all()
    assert values['query_order'].tolist() == run['order']
    assert values['sentence_order'].tolist() == ['good', 'bad']
    assert values['sentence_margin_signs'].tolist() == [1, -1]
    indices = values['row_indices']
    assert values['delta'].shape[:3] == (len(run['order']), len(indices), 2)
    assert np.isfinite(values['delta']).all()
    assert values['tasks'].tolist() == [run['panel']['rows'][i]['task'] for i in indices]
    assert values['row_ids'].tolist() == [run['panel']['rows'][i]['row_id'] for i in indices]
    np.testing.assert_array_equal(values['margin'], series(run, target, method)[:, indices])
    identities.append(dict(path=path.as_posix(), sha256=digest(path), shape=list(values['delta'].shape)))
    return values


def mechanism_analysis(run, output_dir):
    identities = []
    training_targets = run['config']['training_targets']
    heldout_targets = run['config']['heldout_targets']
    targets = run['config']['target_seeds']
    source_seed = run['config']['source_seed']
    assert training_targets == [2, 3] and set(training_targets + heldout_targets) <= set(targets)
    captures = {target: {method: load_capture(run, target, method, identities)
        for method in ['source', 'initial', 'source_columns_lr0001', 'source_columns_gain']} for target in targets}
    reference = captures[2]['source']
    for entry in captures.values():
        for data in entry.values():
            np.testing.assert_array_equal(data['row_indices'], reference['row_indices'])
            assert data['delta'].shape == reference['delta'].shape
    source_gradient = reference['source_margin_gradient'].astype(np.float64)
    assert np.isfinite(source_gradient).all()
    for target in [value for value in targets if value != 2]:
        np.testing.assert_allclose(captures[target]['source']['delta'], reference['delta'], atol=2e-5, rtol=0)
        np.testing.assert_allclose(captures[target]['source']['source_margin_gradient'], source_gradient, atol=2e-5, rtol=0)
    # 所有误差向量、梯度和margin在同一个source干预点上定义。
    errors = {target: captures[target]['initial']['delta'].astype(np.float64)-captures[target]['source']['delta']
              for target in targets}
    mu = (errors[2]+errors[3])/2
    prediction = np.einsum('qrsh,qrsh->qr', source_gradient, mu)
    scalar_control = ((captures[2]['initial']['margin']-captures[2]['source']['margin'])+
                      (captures[3]['initial']['margin']-captures[3]['source']['margin']))/2
    order = run['order']
    families = dict(primary=run['config']['primary_requests'], all=order)
    tasks = run['config']['tasks']
    tasks_by_row = reference['tasks']
    rows, results = [], {}
    overall_truth, overall_predictions, overall_scalar = [], [], []
    for target in heldout_targets:
        actual = captures[target]['initial']['margin']-captures[target]['source']['margin']
        changes = {}
        for method in ['source_columns_lr0001', 'source_columns_gain']:
            correction = captures[target][method]['delta'].astype(np.float64)-captures[target]['initial']['delta']
            predicted_change = np.einsum('qrsh,qrsh->qr', source_gradient, correction)
            actual_change = captures[target][method]['margin']-captures[target]['initial']['margin']
            calibrated_error = actual+actual_change
            changes[method] = dict(predicted_change=predicted_change, actual_change=actual_change,
                calibrated_error=calibrated_error,
                actual_reduction=np.square(actual)-np.square(calibrated_error),
                predicted_reduction=np.square(actual)-np.square(actual+predicted_change))
        del correction
        results[str(target)] = {}
        for family, queries in families.items():
            qi = [order.index(name) for name in queries]
            groups = {task: np.flatnonzero(tasks_by_row == task) for task in tasks}
            groups['all_grammars'] = np.arange(len(tasks_by_row))
            results[str(target)][family] = {}
            for task, ids in groups.items():
                ix = np.ix_(qi, ids)
                truth = actual[ix]
                results[str(target)][family][task] = dict(
                    hidden_common_error_prediction=fit_quality(truth, prediction[ix]),
                    training_target_margin_mean=fit_quality(truth, scalar_control[ix]),
                    zero=fit_quality(truth, np.zeros_like(truth)), calibration={})
                for method in ['source_columns_lr0001', 'source_columns_gain']:
                    predicted_change, actual_change = changes[method]['predicted_change'], changes[method]['actual_change']
                    calibrated_error = changes[method]['calibrated_error']
                    actual_reduction, predicted_reduction = changes[method]['actual_reduction'], changes[method]['predicted_reduction']
                    results[str(target)][family][task]['calibration'][method] = dict(
                        margin_change=fit_quality(actual_change[ix], predicted_change[ix]),
                        squared_error_reduction=fit_quality(actual_reduction[ix], predicted_reduction[ix]),
                        mean_actual_squared_error_reduction=float(actual_reduction[ix].mean()),
                        mean_predicted_squared_error_reduction=float(predicted_reduction[ix].mean()),
                        initial_margin_error_rms=rms(actual[ix]), calibrated_margin_error_rms=rms(calibrated_error[ix]))
        primary_ids = [order.index(name) for name in families['primary']]
        overall_truth.append(actual[primary_ids].ravel())
        overall_predictions.append(prediction[primary_ids].ravel())
        overall_scalar.append(scalar_control[primary_ids].ravel())
        for method in ['source_columns_lr0001', 'source_columns_gain']:
            predicted_change, actual_change = changes[method]['predicted_change'], changes[method]['actual_change']
            for qi, query in enumerate(order):
                for ri, index in enumerate(reference['row_indices']):
                    row = run['panel']['rows'][index]
                    rows.append(dict(source_seed=source_seed, target_seed=target, task=row['task'], row_id=row['row_id'], request=query,
                        method=method, initial_margin_error=float(actual[qi, ri]),
                        common_hidden_prediction=float(prediction[qi, ri]),
                        training_margin_mean=float(scalar_control[qi, ri]),
                        actual_calibration_margin_change=float(actual_change[qi, ri]),
                        gradient_dot_correction=float(predicted_change[qi, ri]),
                        actual_squared_error_reduction=float(actual[qi, ri]**2-(actual[qi, ri]+actual_change[qi, ri])**2)))
    y, p, s = np.concatenate(overall_truth), np.concatenate(overall_predictions), np.concatenate(overall_scalar)
    save_csv(output_dir/f'MECHANISM_SOURCE{source_seed}_UNITS.csv', rows)
    return dict(inputs=identities, by_target=results,
        overall_primary=dict(hidden_common_error_prediction=fit_quality(y, p),
                             training_target_margin_mean=fit_quality(y, s), zero=fit_quality(y, np.zeros_like(y))),
        definition='e_t=delta_initial_t-delta_source; mu=(e_2+e_3)/2; predicted_margin_error=sum(source_margin_gradient*mu)。目标检验数据未参与mu构造。',
        calibration_definition='gradient_dot_correction使用该检验目标实际产生的校准增量，属于链式关系的事后解释。它估计margin_calibrated-margin_initial。平方误差减少为initial_error^2-calibrated_error^2。',
        source_seed=source_seed, training_targets=training_targets, heldout_targets=heldout_targets,
        scope='固定当前source，target2/3提供同句子共同误差，heldout_targets检验。每语法固定前32句对，所有query预先指定。目标检验结果未用于拟合参数。',
        overall_weighting='primary总体在本source内合并固定检验目标、三个等样本数语法与七请求的逐项误差。', rows=len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    output = args.output_dir/'CALIBRATION_REUSE.json'
    assert not output.exists(), output
    assert args.bootstrap > 0
    started, cpu = time.perf_counter(), time.process_time()
    loaded = []
    if args.development:
        run, old = load(ROOT/'CR23_GPT2_GAIN_TARGET23_20260923_SOURCE_SCALAR'), load(OLD)
        assert run['panel'] == old['panel'] and run['order'] == old['order']
        for key in run['schedule']:
            np.testing.assert_array_equal(run['schedule'][key], old['schedule'][key])
        for target in [2, 3, 4, 5]:
            for method in ['none', 'source', 'initial', 'program', 'readout_initial']:
                np.testing.assert_allclose(series(run, target, method), series(old, target, method), atol=2e-5, rtol=0)
            run['arrays'][f't{target}_source_columns_gain'] = series(run, target, 'source_scalar')
            run['arrays'][f't{target}_source_columns_lr0001'] = series(old, target, 'source_columns')
        loaded = [run, old]
        directions = [(run, target) for target in [4, 5]]
    else:
        by_source = {source: load(ROOT/f'CR23_GPT2_SOURCE{source}_CONFIRM_20260923') for source in [1, 4, 5]}
        loaded = list(by_source.values())
        directions = [(by_source[1], 4), (by_source[1], 5), (by_source[4], 5), (by_source[5], 4)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    responses = response_analysis(directions, args.bootstrap, args.output_dir)
    mechanism = None if args.development else dict(by_source={str(source): mechanism_analysis(run, args.output_dir)
        for source, run in by_source.items()}, aggregation='分别报告每个source，source内合并其heldout目标；不以方向数增加独立source样本。')
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        evidence_level='exposed_development' if args.development else 'fresh_sentence_confirmation_fixed_dictionaries',
        inputs=[run['identity'] for run in loaded], responses=responses, mechanism=mechanism,
        runtime=dict(python=sys.executable, numpy=np.__version__, wall_seconds=time.perf_counter()-started,
                     process_cpu_seconds=time.process_time()-cpu, gpu_calls=0), code_sha256=digest(Path(__file__)))
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(primary=responses['source_equal_primary'], comparisons=responses['source_equal_comparisons'],
        mechanism=None if mechanism is None else {source: value['overall_primary']
            for source, value in mechanism['by_source'].items()}, runtime=result['runtime']), indent=2), flush=True)


if __name__ == '__main__':
    main()
