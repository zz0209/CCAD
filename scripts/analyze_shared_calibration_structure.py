import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_shared_action_rule import read_run


ROOT = Path('D:/CCAD_Storage/runs/shared_rule_20260922')
BASE = 'SR20_GPT2_SHARED_TARGET23_DEV_20260922_SOURCE_COLUMNS'
TRANSFER = {4: 'SR20_GPT2_SOURCE4_TARGET5_DEV_20260922',
            5: 'SR20_GPT2_SOURCE5_TARGET4_DEV_20260922'}
ENDPOINTS = ['verb', 'number', 'gender', 'full']
RETAIN = ['number_gender', 'verb_gender', 'verb_number']
PARTS = ['verb', 'number', 'gender']
METHODS = ['initial', 'source_columns', 'program', 'readout_initial', 'response_scalar']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    config, panel, order, arrays, schedule, identity = read_run(path)
    identity.update(config_sha256=digest(path / 'config.resolved.json'),
                    panel_sha256=digest(path / 'panel.json'))
    print(json.dumps(dict(loaded_run=path.name, verified_records=identity['records_verified'])), flush=True)
    return dict(config=config, panel=panel, order=order, arrays=arrays, identity=identity)


def ratio(error, reference):
    energy = float(np.square(reference).sum())
    return float(np.sqrt(np.square(error).sum() / energy)) if energy > 0 else None


def rms(value):
    return float(np.sqrt(np.square(value).mean()))


def get_effects(run, target, scalar):
    prefix = f't{target}_'
    clean = run['arrays'][prefix + 'none']
    result = {name: run['arrays'][prefix + name] - clean
              for name in ['source', 'initial', 'program', 'readout_initial']}
    column_key = prefix + ('source_columns_lr0001' if run['config'].get('evaluation_only') else 'source_columns')
    result['source_columns'] = run['arrays'][column_key] - clean
    result['response_scalar'] = scalar * result['initial']
    return result


def condition(effects, order):
    return effects[[order.index(name) for name in RETAIN]].T - effects[order.index('full')][:, None]


def condition_metrics(source, target):
    error = target - source
    centered = error - error.mean(1, keepdims=True)
    source_centered = source - source.mean(1, keepdims=True)
    sc, tc = source.argmax(1), target.argmax(1)
    regret = source.max(1) - source[np.arange(len(source)), tc]
    energy = rms(source)
    total = float(np.square(error).sum())
    common = float(3 * np.square(error.mean(1)).sum())
    remaining = float(np.square(centered).sum())
    assert np.isclose(total, common + remaining, atol=1e-10, rtol=1e-12)
    return dict(source_effect_rms=energy, source_pairwise_difference_rms=float(np.sqrt(3) * rms(source_centered)),
                conditional_effect_nrmse=ratio(error, source),
                pairwise_difference_nrmse=ratio(centered, source_centered),
                choice_agreement=float((sc == tc).mean()),
                source_regret_mean=float(regret.mean()),
                source_regret_over_source_rms=float(regret.mean() / energy) if energy > 0 else None,
                source_tie_rows=int(((source == source.max(1, keepdims=True)).sum(1) > 1).sum()),
                target_tie_rows=int(((target == target.max(1, keepdims=True)).sum(1) > 1).sum()),
                total_sse=total, common_sse=common, centered_sse=remaining)


def analyze_direction(run, target, scalar, response_rows, condition_rows):
    config, panel, order = run['config'], run['panel'], run['order']
    source_seed = config.get('source_seed', 1)
    values = get_effects(run, target, scalar)
    primary = [order.index(name) for name in config['primary_requests']]
    groups = {task: np.array([i for i, row in enumerate(panel['rows']) if row['task'] == task])
              for task in config['tasks']}
    result = dict(source_seed=source_seed, target_seed=target, by_task={})
    source_condition = condition(values['source'], order)
    for task, ids in groups.items():
        truth = values['source'][np.ix_(primary, ids)]
        result['by_task'][task] = dict(rows=len(ids), primary_source_effect_rms=rms(truth), methods={})
        for method in METHODS:
            predicted = values[method][np.ix_(primary, ids)]
            conditional = condition(values[method], order)
            metrics = condition_metrics(source_condition[ids], conditional[ids])
            result['by_task'][task]['methods'][method] = dict(
                primary_nrmse=ratio(predicted - truth, truth), primary_error_rms=rms(predicted - truth),
                by_request={name: dict(source_effect_rms=rms(values['source'][qi, ids]),
                    error_rms=rms(values[method][qi, ids] - values['source'][qi, ids]),
                    nrmse=ratio(values[method][qi, ids] - values['source'][qi, ids], values['source'][qi, ids]))
                    for qi, name in enumerate(order)}, conditional=metrics)
            for ri in ids:
                row = panel['rows'][ri]
                for qi, query in enumerate(order):
                    response_rows.append(dict(source_seed=source_seed, target_seed=target, task=task,
                        row_id=row['row_id'], method=method, request=query,
                        source_effect=float(values['source'][qi, ri]),
                        predicted_effect=float(values[method][qi, ri]),
                        error=float(values[method][qi, ri] - values['source'][qi, ri])))
                s, t = source_condition[ri], conditional[ri]
                record = dict(source_seed=source_seed, target_seed=target, task=task, row_id=row['row_id'],
                    method=method, source_choice=PARTS[int(s.argmax())], predicted_choice=PARTS[int(t.argmax())],
                    source_regret=float(s.max() - s[t.argmax()]))
                for j, part in enumerate(PARTS):
                    record['source_' + part] = float(s[j])
                    record['predicted_' + part] = float(t[j])
                condition_rows.append(record)
    result['mean_primary_nrmse'] = {method: float(np.mean([
        entry['methods'][method]['primary_nrmse'] for entry in result['by_task'].values()])) for method in METHODS}
    return result, values


def source_comparison(base, transferred, target, scalar):
    first = get_effects(base, target, scalar)
    second = get_effects(transferred, target, scalar)
    config, panel, order = base['config'], base['panel'], base['order']
    assert panel['rows'] == transferred['panel']['rows']
    assert order == transferred['order']
    source = np.stack([first['source'], second['source']])
    families = dict(primary=config['primary_requests'], parts=PARTS,
                    full=['full'], **{name: [name] for name in config['primary_requests']})
    result = dict(source_seeds=[1, transferred['config']['source_seed']], target_seed=target, by_task={})
    for task in config['tasks']:
        ids = np.array([i for i, row in enumerate(panel['rows']) if row['task'] == task])
        result['by_task'][task] = {}
        for family, queries in families.items():
            qi = [order.index(name) for name in queries]
            truth = source[:, qi][:, :, ids]
            difference = truth[0] - truth[1]
            result['by_task'][task][family] = dict(source_effect_rms=rms(truth),
                source_difference_rms=rms(difference), methods={})
            for method in METHODS:
                predictions = np.stack([first[method], second[method]])[:, qi][:, :, ids]
                error = predictions - truth
                mean_error = error.mean(0)
                total = float(np.square(error).sum())
                common = float(2 * np.square(mean_error).sum())
                centered = float(np.square(error - mean_error[None]).sum())
                difference_error = predictions[0] - predictions[1] - difference
                assert np.isclose(total, common + centered, atol=1e-10, rtol=1e-12)
                assert np.isclose(np.square(difference_error).sum(), 2 * centered, atol=1e-10, rtol=1e-12)
                result['by_task'][task][family]['methods'][method] = dict(
                    absolute_nrmse=ratio(error, truth), source_difference_nrmse=ratio(difference_error, difference),
                    source_difference_error_rms=rms(difference_error), total_sse=total,
                    common_sse=common, centered_sse=centered)
    return result


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    output = args.output_dir / 'CALIBRATION_STRUCTURE.json'
    assert not output.exists(), output
    started, cpu = time.perf_counter(), time.process_time()
    started_at = datetime.now(timezone.utc).isoformat()
    base = load(ROOT / BASE)
    order = base['order']
    endpoints = [order.index(name) for name in ENDPOINTS]
    initial, truth = [], []
    for target in [2, 3]:
        arrays = get_effects(base, target, 1.)
        initial.append(arrays['initial'][endpoints].ravel())
        truth.append(arrays['source'][endpoints].ravel())
    x, y = np.concatenate(initial), np.concatenate(truth)
    assert float(x @ x) > 0
    scalar = float((x @ y) / (x @ x))
    loaded = [base]
    response_rows, condition_rows, directions, contrasts = [], [], {}, {}
    for target in [4, 5]:
        directions[f's1_t{target}'], _ = analyze_direction(base, target, scalar, response_rows, condition_rows)
    if not args.smoke:
        for source, name in TRANSFER.items():
            run = load(ROOT / name)
            loaded.append(run)
            target = 5 if source == 4 else 4
            assert run['panel']['rows'] == base['panel']['rows'] and run['order'] == order
            directions[f's{source}_t{target}'], _ = analyze_direction(run, target, scalar, response_rows, condition_rows)
            contrasts[f't{target}_s1_s{source}'] = source_comparison(base, run, target, scalar)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.output_dir / 'RESPONSE_UNITS.csv', response_rows)
    save_csv(args.output_dir / 'CONDITIONAL_UNITS.csv', condition_rows)
    result = dict(started_at_utc=started_at, written_at_utc=datetime.now(timezone.utc).isoformat(),
        smoke=args.smoke, evidence_level='retrospective_exposed_development_response_analysis',
        question='跨字典可复用的功能校准保存了哪些响应幅度、部分差异和来源差异',
        data_identity='三项run使用相同96个已暴露开发句对。比例拟合使用source1与target2/3，评价使用target4/5。全部文本已暴露，新来源评价保持开发身份。',
        diagnostic='response_scalar只将已保存initial响应减clean后的作用乘以一个比例。这是响应预测解释诊断，真实物理缩放干预需要另外运行模型。',
        averaging='primary先在每任务和方向内汇总平方量计算nRMSE，再等权平均三个任务。来源比较按固定target进行，两个方向共享seed。并列选择使用verb、number、gender顺序。',
        fit=dict(model='one_global_zero_intercept_response_scalar', coefficient=scalar,
                 source_seed=1, target_seeds=[2, 3], requests=ENDPOINTS,
                 observations=len(x), unique_sentence_pairs=len(base['panel']['rows']),
                 initial_nrmse=ratio(x-y, y), fitted_nrmse=ratio(scalar*x-y, y),
                 loss='sum((coefficient * initial_effect - source_effect)^2)'),
        inputs=[run['identity'] for run in loaded], directions=directions, source_comparisons=contrasts,
        output_units=dict(response_rows=len(response_rows), conditional_rows=len(condition_rows)),
        runtime=dict(python=sys.executable, numpy=np.__version__, wall_seconds=time.perf_counter()-started,
                     process_cpu_seconds=time.process_time()-cpu, gpu_calls=0),
        code_sha256=digest(Path(__file__)))
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(coefficient=scalar, directions={key: value['mean_primary_nrmse']
        for key, value in directions.items()}, runtime=result['runtime']), indent=2), flush=True)


if __name__ == '__main__':
    main()
