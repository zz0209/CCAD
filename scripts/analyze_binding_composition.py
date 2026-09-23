import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


METHODS = ('source_additive', 'source_conditional', 'raw_additive', 'raw_conditional')
OPERATIONS = ('entity', 'attribute', 'both')
CONTRASTS = (
    ('source_conditional', 'source_additive'),
    ('raw_conditional', 'raw_additive'),
    ('source_conditional', 'raw_conditional'),
    ('source_additive', 'raw_additive'),
)
FIXED_CHOICES = {
    'fixed_geometry': ('geometry', 'geometry'),
    'fixed_source_cond0': ('source_cond0', 'source_cond0'),
    'fixed_source_cond1': ('source_cond1', 'source_cond1'),
    'fixed_raw_cond0': ('raw_cond0', 'raw_cond0'),
    'fixed_raw_cond1': ('raw_cond1', 'raw_cond1'),
    'fixed_source_r57': ('source_r57_path', 'source_cond1'),
    'fixed_raw_r57': ('raw_r57_path', 'raw_cond1'),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utility(margins):
    return np.min(margins * np.array([-1.0, -1.0, 1.0]), axis=-1)


def rank_correlation(actual, predicted):
    a, p = rankdata(actual), rankdata(predicted)
    a, p = a - a.mean(), p - p.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(p)
    return float(a @ p / denominator) if denominator > 0 else None


def group_key(row):
    return tuple(row[key] for key in (
        'panel_seed', 'source_seed', 'target_seed', 'component', 'order', 'template', 'query'))


def read_runs(runs, phase):
    groups, choices, identities = defaultdict(list), {}, []
    for run in runs:
        status = json.loads((run / 'status.json').read_text(encoding='utf-8'))
        if status['status'] != 'PASS':
            raise ValueError(f'运行尚未完成 {run}')
        raw = run / 'metrics.raw.jsonl'
        config = json.loads((run / 'config.resolved.json').read_text(encoding='utf-8'))
        count = 0
        with raw.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['kind'] not in ('composition_candidate', 'composition_choice'):
                    continue
                if row['analysis_phase'] != phase:
                    raise ValueError(f'开发与确认阶段不一致 {run} {row["analysis_phase"]} {phase}')
                row.setdefault('panel_seed', config['panel_seed'])
                row.setdefault('source_seed', config['source_seed'])
                key = group_key(row)
                if row['kind'] == 'composition_candidate':
                    groups[key].append(row)
                    count += 1
                else:
                    choice_key = (key, row['method'])
                    if choice_key in choices:
                        raise ValueError(f'重复选择记录 {choice_key}')
                    choices[choice_key] = row
        identities.append(dict(run=run.as_posix(), candidate_records=count,
            raw_sha256=digest(raw), config_sha256=digest(run / 'config.resolved.json')))
    if not groups:
        raise ValueError('没有 composition_candidate 记录')
    return groups, choices, identities


def analyze_group(key, rows, choices):
    candidate_ids = [row['candidate_id'] for row in rows]
    if len(set(candidate_ids)) != len(rows):
        raise ValueError(f'候选重复 {key}')
    combinations = {(row['entity_candidate'], row['attribute_candidate']) for row in rows}
    entities = {row['entity_candidate'] for row in rows}
    attributes = {row['attribute_candidate'] for row in rows}
    if len(combinations) != len(rows) or len(rows) != len(entities) * len(attributes):
        raise ValueError(f'候选组合记录不完整 {key}')
    actual = np.array([[row[f'margin_{op}'] for op in OPERATIONS] for row in rows], dtype=float)
    if not np.isfinite(actual).all():
        raise ValueError(f'真实 margin 非有限值 {key}')
    truth = utility(actual)
    np.testing.assert_allclose(truth, [row['true_utility'] for row in rows], atol=2e-5, rtol=1e-6)
    success = actual * [-1, -1, 1] > 0
    np.testing.assert_array_equal(success, [[row[f'success_{op}'] for op in OPERATIONS] for row in rows])
    np.testing.assert_array_equal(success.all(axis=1), [row['all_success'] for row in rows])
    correct = np.array([[row[f'prediction_token_{op}'] ==
        row['original_answer_id' if op == 'both' else 'swap_answer_id']
        for op in OPERATIONS] for row in rows], dtype=bool)
    np.testing.assert_array_equal(correct, [[row[f'correct_{op}'] for op in OPERATIONS] for row in rows])
    oracle = float(truth.max())
    output, selected = {}, []
    def actual_metrics(chosen):
        values = dict(actual_utility=float(truth[chosen]), oracle_regret=float(oracle-truth[chosen]),
            oracle_utility=oracle, all_token_accuracy=float(correct[chosen].all()),
            all_margin_accuracy=float(success[chosen].all()),
            selected_oracle_percentile=float((np.sum(truth < truth[chosen]) +
                0.5 * (np.sum(truth == truth[chosen])-1)) / (len(truth)-1)) if len(truth)>1 else None)
        for oi, operation in enumerate(OPERATIONS):
            values[f'{operation}_original_margin'] = float(actual[chosen, oi])
            values[f'{operation}_token_accuracy'] = float(correct[chosen, oi])
            values[f'{operation}_margin_accuracy'] = float(success[chosen, oi])
        return values
    def save_selection(method, chosen, values, predicted_utility):
        output[method] = values
        selected.append(dict(panel_seed=key[0], source_seed=key[1], target_seed=key[2],
            component=key[3], order=key[4], template=key[5], query=key[6], method=method,
            candidate_id=candidate_ids[chosen], predicted_utility=predicted_utility, **values))
    for method in METHODS:
        predicted = np.array([[row[f'prediction_{method}_{op}'] for op in OPERATIONS] for row in rows], dtype=float)
        if not np.isfinite(predicted).all():
            raise ValueError(f'预测 margin 非有限值 {key} {method}')
        predicted_utility = utility(predicted)
        np.testing.assert_allclose(predicted_utility,
            [row[f'prediction_{method}_utility'] for row in rows], atol=2e-5, rtol=1e-6)
        chosen = int(np.argmax(predicted_utility))
        logged = choices.get((key, method))
        if logged is not None:
            if logged['candidate_id'] != candidate_ids[chosen]:
                raise ValueError(f'预测选择与执行记录不一致 {key} {method}')
            np.testing.assert_allclose(logged['actual_utility'], truth[chosen], atol=2e-5, rtol=1e-6)
            np.testing.assert_allclose(logged['oracle_regret'], oracle-truth[chosen], atol=2e-5, rtol=1e-6)
        values = actual_metrics(chosen)
        values.update(candidate_utility_mse=float(np.square(predicted_utility-truth).mean()),
            candidate_utility_spearman=rank_correlation(truth, predicted_utility))
        for oi, operation in enumerate(OPERATIONS):
            values[f'candidate_{operation}_mse'] = float(np.square(predicted[:, oi]-actual[:, oi]).mean())
            values[f'candidate_{operation}_spearman'] = rank_correlation(actual[:, oi], predicted[:, oi])
        save_selection(method, chosen, values, float(predicted_utility[chosen]))
    for method, pair in FIXED_CHOICES.items():
        matches = [i for i, row in enumerate(rows)
            if (row['entity_candidate'], row['attribute_candidate']) == pair]
        if len(matches) != 1:
            raise ValueError(f'固定比较的候选组合缺失 {key} {pair}')
        chosen = matches[0]
        values = actual_metrics(chosen)
        values.update({metric: None for metric in output[METHODS[0]] if metric.startswith('candidate_')})
        save_selection(method, chosen, values, None)
    return output, selected


def estimate(values, weights, square_root=False):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return dict(mean=None, ci95=None, valid_worlds=0), np.full(len(weights), np.nan)
    numerator = weights[:, valid] @ values[valid]
    denominator = weights[:, valid].sum(axis=1)
    samples = np.divide(numerator, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)
    point = float(values[valid].mean())
    if square_root:
        point, samples = float(np.sqrt(point)), np.sqrt(samples)
    return dict(mean=point, ci95=np.nanquantile(samples, [.025, .975]).tolist(),
        valid_worlds=int(valid.sum())), samples


def analyze(runs, phase, bootstrap, seed):
    groups, choices, identities = read_runs(runs, phase)
    cells, selected = defaultdict(list), []
    layouts = defaultdict(set)
    splits = set()
    candidate_counts = set()
    for key, rows in sorted(groups.items()):
        result, selections = analyze_group(key, rows, choices)
        selected.extend(selections)
        world = (key[0], key[3])
        layouts[world].add((key[1], key[2], key[4], key[5], key[6]))
        splits.update(row['split'] for row in rows)
        candidate_counts.add(len(rows))
        for method, metrics in result.items():
            cells[world, method].append(metrics)
    if 'fit' in splits or len(splits) != 1:
        raise ValueError(f'评价输入必须来自单一非 fit split，当前 {splits}')
    worlds = sorted(layouts)
    if any(layouts[world] != layouts[worlds[0]] for world in worlds):
        raise ValueError('world 的目标、query、order、template 组合不一致')
    if len(candidate_counts) != 1:
        raise ValueError('各评价上下文的候选数量不一致')
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(worlds), np.full(len(worlds), 1 / len(worlds)), size=bootstrap)
    metric_names = list(next(iter(cells.values()))[0])
    world_values, summaries = {}, {}
    method_names = list(METHODS) + list(FIXED_CHOICES)
    for method in method_names:
        world_values[method], summaries[method] = {}, {}
        for metric in metric_names:
            values = []
            for world in worlds:
                available = [row[metric] for row in cells[world, method] if row[metric] is not None]
                values.append(float(np.mean(available)) if available else None)
            world_values[method][metric] = values
            name = metric[:-3] + 'rmse' if metric.endswith('_mse') else metric
            summaries[method][name] = estimate(values, weights, metric.endswith('_mse'))[0]
    contrasts = {}
    comparisons = list(CONTRASTS) + [(method, fixed) for method in ('source_conditional', 'raw_conditional')
        for fixed in FIXED_CHOICES]
    for first, second in comparisons:
        contrasts[f'{first}_minus_{second}'] = {}
        for metric in metric_names:
            a = np.array(world_values[first][metric], dtype=float)
            b = np.array(world_values[second][metric], dtype=float)
            paired = np.isfinite(a) & np.isfinite(b)
            a[~paired], b[~paired] = np.nan, np.nan
            is_mse = metric.endswith('_mse')
            left, left_samples = estimate(a, weights, is_mse)
            right, right_samples = estimate(b, weights, is_mse)
            name = metric[:-3] + 'rmse' if is_mse else metric
            differences = left_samples-right_samples
            contrasts[f'{first}_minus_{second}'][name] = dict(
                mean=left['mean']-right['mean'] if paired.any() else None,
                ci95=np.nanquantile(differences, [.025, .975]).tolist() if paired.any() else None,
                valid_worlds=int(paired.sum()))
    return dict(written_at_utc=datetime.now(timezone.utc).isoformat(), phase=phase,
        inputs=identities, analyzer_sha256=digest(Path(__file__)),
        worlds=len(worlds), world_ids=[list(world) for world in worlds],
        target_seeds=sorted({key[2] for key in groups}), source_seeds=sorted({key[1] for key in groups}),
        panel_seeds=sorted({key[0] for key in groups}), raw_split=next(iter(splits)),
        candidate_count=next(iter(candidate_counts)), context_groups=len(groups), methods=summaries,
        paired_contrasts=contrasts, world_values=world_values, selected_candidates=selected,
        bootstrap=dict(draws=bootstrap, seed=seed, unit='panel_seed/world',
            scope='配对重采样整个 world，保留共享 world 的 query、order、template、全部目标和方法。目标 SAE cohort 固定，区间不外推 seed 总体。'),
        definitions=dict(margin='固定原答案 log probability 减 swapped 答案 log probability。',
            utility='min(-margin_entity, -margin_attribute, margin_both)',
            selection='最大预测 utility，精确并列使用原始候选记录中的第一项。目标响应仅用于离线评价。',
            fixed_choices=FIXED_CHOICES,
            fixed_comparison='固定候选直接从同池真实响应提取，不拟合预测器，不增加目标调用；预测误差和相关系数记为 null。',
            oracle='同一候选组合池中的最大真实 utility；regret 为 oracle utility 减所选 utility。',
            aggregation='每个 world 内等权平均目标和上下文，再等权平均 world。候选 RMSE 从整体 MSE 开平方，bootstrap 内重新开平方。',
            accuracy='token accuracy 使用全词表 argmax；margin accuracy 使用两个指定答案之间的 margin 符号。',
            rank='组内 Spearman 使用平均并列秩；真实或预测常数的组不定义，记录 null 和有效 world 数。',
            percentile='所选真实 utility 的组内秩百分位使用平均并列秩，范围为零至一。'))


def report(result):
    lines = ['# 条件成员组合预测与选择', '',
        f"阶段为 {result['phase']}，包含 {result['worlds']} 个 world、{result['context_groups']} 个上下文与目标组合，每组 {result['candidate_count']} 个候选。", '',
        result['bootstrap']['scope'], '',
        '| 方法 | utility | 三项 token 全成功 | joint token accuracy | oracle regret | 候选 utility RMSE | 组内 utility Spearman |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def show(value):
        if value['mean'] is None:
            return '未定义'
        low, high = value['ci95']
        return f"{value['mean']:.4f} [{low:.4f}, {high:.4f}]"
    keys = ('actual_utility', 'all_token_accuracy', 'both_token_accuracy', 'oracle_regret',
            'candidate_utility_rmse', 'candidate_utility_spearman')
    for method, values in result['methods'].items():
        lines.append('| ' + ' | '.join([method] + [show(values[key]) for key in keys]) + ' |')
    lines += ['', '数值区间为 world 配对 bootstrap 的 95% 区间，准确率使用零至一尺度。', '',
        '| 配对差 | utility | 三项 token 全成功 | joint token accuracy | oracle regret | 候选 utility RMSE | 组内 utility Spearman |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, values in result['paired_contrasts'].items():
        lines.append('| ' + ' | '.join([name] + [show(values[key]) for key in keys]) + ' |')
    lines += ['', '三项操作的原答案 margin、token accuracy、margin accuracy、预测 RMSE 和排序，以及每个 world 的量和所选候选均保存在同名 JSON。', '',
        result['definitions']['utility'], '', result['definitions']['selection'], '', result['definitions']['rank'], '']
    return '\n'.join(lines)


def responsibility_analysis(runs, phase, r56):
    groups, _, identities = read_runs(runs, phase)
    contexts, world_rows = [], defaultdict(list)
    for key, rows in sorted(groups.items()):
        entities = list(dict.fromkeys(row['entity_candidate'] for row in rows))
        attributes = list(dict.fromkeys(row['attribute_candidate'] for row in rows))
        by_pair = {(row['entity_candidate'], row['attribute_candidate']): row for row in rows}
        joint = np.array([[by_pair[e, a]['margin_both'] for a in attributes] for e in entities])
        entity = np.array([by_pair[e, attributes[0]]['margin_entity'] for e in entities])
        attribute = np.array([by_pair[entities[0], a]['margin_attribute'] for a in attributes])
        values = dict(entity_single_range=float(np.ptp(entity)),
            attribute_single_range=float(np.ptp(attribute)),
            joint_fixed_attribute_entity_mean_range=float(np.ptp(joint, axis=0).mean()),
            joint_fixed_entity_attribute_mean_range=float(np.ptp(joint, axis=1).mean()))
        objective_details = {}
        objectives = dict(joint_original_margin=joint,
            utility=np.minimum(np.minimum(-entity[:, None], -attribute[None, :]), joint))
        for name, matrix in objectives.items():
            chosen = np.argmax(matrix, axis=0)
            best_sets = [[entities[e] for e in np.flatnonzero(matrix[:, ai] == matrix[:, ai].max())]
                         for ai in range(len(attributes))]
            pairs = [(i, j) for i in range(len(attributes)) for j in range(i+1, len(attributes))]
            count = len(set(chosen.tolist()))
            values[name+'_distinct_best_entity_count'] = count
            values[name+'_best_entity_changes'] = float(count > 1)
            values[name+'_attribute_pair_switch_fraction'] = float(np.mean([chosen[i] != chosen[j] for i, j in pairs]))
            values[name+'_distinct_best_entity_sets'] = len({tuple(group) for group in best_sets})
            objective_details[name] = dict(best_entity_by_attribute=dict(zip(attributes,
                [entities[index] for index in chosen])), best_entity_sets_by_attribute=dict(zip(attributes, best_sets)))
        context = dict(panel_seed=key[0], source_seed=key[1], target_seed=key[2], world=key[3],
            order=key[4], template=key[5], query=key[6], metrics=values, best_entities=objective_details)
        contexts.append(context)
        world_rows[key[0], key[3]].append(values)
    names = list(contexts[0]['metrics'])
    world_values = [dict(panel_seed=world[0], world=world[1], contexts=len(rows),
        metrics={name: float(np.mean([row[name] for row in rows])) for name in names})
        for world, rows in sorted(world_rows.items())]
    overall = {name: float(np.mean([world['metrics'][name] for world in world_values])) for name in names}
    r56_config = json.loads((r56 / 'config.resolved.json').read_text(encoding='utf-8'))
    r56_rows = [json.loads(line) for line in (r56 / 'metrics.raw.jsonl').read_text(encoding='utf-8').splitlines()]
    layer_sets = {entry['name']: entry['layers'] for entry in r56_config['layer_sets']}
    r56_results = {}
    for method in ('existing_pair', 'all_except_pair'):
        selected = [row for row in r56_rows if row['kind'] == 'intervention'
                    and row['operation'] == 'attribute' and row['method'] == method]
        per_world = defaultdict(list)
        for row in selected:
            correct = row['answer_id'] == row['expected_id']
            per_world[row['component']].append(correct)
        count = sum(sum(values) for values in per_world.values())
        r56_results[method] = dict(layers=layer_sets[method], rows=len(selected), correct_rows=count,
            world_count=len(per_world), world_equal_accuracy=float(np.mean([np.mean(v) for v in per_world.values()])),
            world_counts={str(world): dict(correct_rows=sum(v), rows=len(v)) for world, v in per_world.items()})
    return dict(written_at_utc=datetime.now(timezone.utc).isoformat(), phase=phase,
        inputs=identities, analyzer_sha256=digest(Path(__file__)), worlds=len(world_values),
        contexts=len(contexts), candidate_grid=[len(entities), len(attributes)], world_equal=overall,
        world_values=world_values, context_values=contexts,
        r56=dict(run=r56.as_posix(), raw_sha256=digest(r56/'metrics.raw.jsonl'),
            config_sha256=digest(r56/'config.resolved.json'), results=r56_results),
        definitions=dict(range='候选最大原答案 margin 减最小值。单侧范围分别遍历 entity 或 attribute 候选。',
            joint_range='固定另一部分候选，改变当前部分后计算 joint 原答案 margin 范围，再平均固定候选。',
            best='分别最大化 joint 原答案 margin 和 min(-mE,-mA,mEA)。精确并列使用原始候选顺序第一项，同时保留全部并列最佳集合。',
            switch='六个 attribute 候选的十五种无序配对中，最佳 entity 不同的比例；changes 表示该上下文是否出现任何改变。',
            aggregation='各 world 内等权平均 query、order、template 和目标，再等权平均 world。这里只描述已观察的开发候选。'),
        interpretation='当前候选只改变 pre14 和 pre24 的 SAE 成员，其余层保留相同 raw role 更新。R56 对同一层分组的实际 attribute 操作提供共同背景承担属性作用的既有证据；当前 range 测量说明在此共同背景下改变所选层成员能改变多少输出。R56 与当前开发世界、操作材料及候选条件不同，不能将其正确率直接当作当前候选的反事实分解。')


def responsibility_report(result):
    metrics = result['world_equal']
    labels = dict(entity_single_range='entity 单侧候选范围', attribute_single_range='attribute 单侧候选范围',
        joint_fixed_attribute_entity_mean_range='joint 固定 attribute 改变 entity 的平均范围',
        joint_fixed_entity_attribute_mean_range='joint 固定 entity 改变 attribute 的平均范围')
    lines = ['# 成员组合的作用责任', '',
        f"使用 {result['worlds']} 个开发 world、{result['contexts']} 个上下文，每个上下文包含 8×6 个候选组合。所有统计先在 world 内平均，再对 world 等权平均。", '',
        '| 观察量 | 平均原答案 margin 范围 |', '|---|---:|']
    lines += [f'| {label} | {metrics[name]:.6f} |' for name, label in labels.items()]
    lines += ['', '| 最佳 entity 的判定目标 | 不同最佳 entity 数量 | 出现改变的上下文比例 | attribute 配对切换比例 |', '|---|---:|---:|---:|']
    for objective in ('joint_original_margin', 'utility'):
        lines.append(f"| {objective} | {metrics[objective+'_distinct_best_entity_count']:.6f} | "
            f"{metrics[objective+'_best_entity_changes']:.6f} | {metrics[objective+'_attribute_pair_switch_fraction']:.6f} |")
    lines += ['', result['definitions']['best'], '', '| R56 属性操作 | 层集合 | 正确行数 | 总行数 | world 等权准确率 |', '|---|---|---:|---:|---:|']
    for method, values in result['r56']['results'].items():
        layers = 'pre14、pre24' if method == 'existing_pair' else '其余 26 层'
        lines.append(f"| {method} | {layers} | {values['correct_rows']} | {values['rows']} | {values['world_equal_accuracy']:.6f} |")
    lines += ['', result['interpretation'], '', '逐上下文范围、全部最佳候选集合及逐 world 数值保存于同名 JSON。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--phase', choices=['development', 'confirmation'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=9232301)
    parser.add_argument('--responsibility-r56', type=Path)
    args = parser.parse_args()
    if args.bootstrap < 1:
        raise ValueError('bootstrap 必须为正数')
    if args.output.exists() or args.output.with_suffix('.md').exists():
        raise FileExistsError('分析输出已存在，请使用新的输出文件名')
    result = (responsibility_analysis(args.runs, args.phase, args.responsibility_r56)
              if args.responsibility_r56 else analyze(args.runs, args.phase, args.bootstrap, args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    rendered = responsibility_report(result) if args.responsibility_r56 else report(result)
    args.output.with_suffix('.md').write_text(rendered, encoding='utf-8')
    print(json.dumps(dict(phase=result['phase'], worlds=result['worlds'],
        output=str(args.output)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
