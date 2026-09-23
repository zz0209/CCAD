import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FAMILIES = ('source', 'native')
OPERATIONS = ('none', 'P', 'W', 'PW', 'P_restoreW', 'none_restoreW')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def effects(values):
    return dict(P=values['P']-values['none'], W=values['W']-values['none'],
        PW=values['PW']-values['none'], W_after_P=values['PW']-values['P'],
        restore=values['P_restoreW']-values['P'], restored_vs_none=values['P_restoreW']-values['none'],
        interaction=values['PW']-values['P']-values['W']+values['none'])


def ratio(a, b):
    return float(a/b) if b > 0 else None


def correlation(a, b):
    a, b = a-a.mean(), b-b.mean()
    return ratio(a@b, np.linalg.norm(a)*np.linalg.norm(b))


def rms(value):
    return float(np.sqrt(np.mean(np.square(value))))


def load(run):
    status = json.loads((run/'status.json').read_text())
    if status['status'] != 'PASS':
        raise ValueError(f'运行尚未 PASS {run}')
    rows = json.loads((run/'evaluation_membership.json').read_text())['rows']
    index = {row['document_sha256']: i for i, row in enumerate(rows)}
    data = dict(rows=rows, families={}, site_rows=defaultdict(list))
    for family in FAMILIES:
        logits, pooled = {}, {}
        for op in OPERATIONS:
            with np.load(run/f'{family}__{op}.npz') as arrays:
                logits[op], pooled[op] = arrays['logits'].astype(float), arrays['pooled'].astype(float)
        data['families'][family] = dict(logits=logits, effects=effects(logits), pooled=effects(pooled),
            identity_logit=float(np.abs(logits['none_restoreW']-logits['none']).max()),
            identity_pooled=float(np.abs(pooled['none_restoreW']-pooled['none']).max()))
    for line in (run/'metrics.raw.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['kind'] == 'site_contribution':
            data['site_rows'][row['method'], row['task'], row['operation']].append(row)
        elif row['kind'] == 'classification':
            np.testing.assert_allclose(row['logit'], data['families'][row['method']]['logits'][row['operation']][index[row['component']]], rtol=0, atol=0)
    data['sites'] = sorted({key[1] for key in data['site_rows']})
    data['contributions'] = {}
    for key, records in data['site_rows'].items():
        records = sorted(records, key=lambda row: index[row['component']])
        if [index[row['component']] for row in records] != list(range(len(rows))):
            raise ValueError(f'site 文档记录不完整 {key}')
        data['contributions'][key] = {name: np.array([row[name] for row in records]) for name in (
            'W_contribution_l2', 'baseline_W_contribution_l2', 'W_change_l2',
            'W_current_dot_baseline', 'W_code_sum', 'baseline_W_code_sum', 'restore_delta_l2')}
    return data


def statistics(data, indices):
    result = {}
    label_sign = np.array([2*row['label']-1 for row in data['rows']])[indices]
    for family, values in data['families'].items():
        e = {name: value[indices] for name, value in values['effects'].items()}
        for name, value in e.items():
            result[f'{family}/logit/{name}/mean'] = float(value.mean())
            result[f'{family}/logit/{name}/rms'] = rms(value)
            result[f'{family}/logit/{name}/label_margin_mean'] = float(np.mean(value*label_sign))
            result[f'{family}/pooled/{name}/rms'] = rms(values['pooled'][name][indices])
        p, restoration = e['P'], e['restore']
        result[f'{family}/restoration/opposes_P_fraction'] = float(np.mean(restoration*p < 0))
        result[f'{family}/restoration/zero_P_fraction'] = float(np.mean(p == 0))
        result[f'{family}/restoration/projection_against_P'] = ratio(-restoration@p, p@p)
        result[f'{family}/restoration/correlation_against_P'] = correlation(restoration, -p)
        result[f'{family}/restoration/restored_effect_rms_over_P'] = ratio(rms(e['restored_vs_none']), rms(p))
        result[f'{family}/conditional/W_after_P_rms_over_W'] = ratio(rms(e['W_after_P']), rms(e['W']))
        result[f'{family}/conditional/W_after_P_correlation_W'] = correlation(e['W_after_P'], e['W'])
        pooled_p, pooled_r = values['pooled']['P'][indices], values['pooled']['restore'][indices]
        result[f'{family}/pooled/restoration_projection_against_P'] = ratio(-np.sum(pooled_p*pooled_r), np.square(pooled_p).sum())
        energy_before, energy_after = np.zeros(len(indices)), np.zeros(len(indices))
        for site in data['sites']:
            natural = data['contributions'][family, site, 'P']
            before = natural['baseline_W_contribution_l2'][indices]**2
            after = natural['W_contribution_l2'][indices]**2
            change = natural['W_change_l2'][indices]**2
            energy_before += before
            energy_after += after
            prefix = f'{family}/site/{site}'
            result[prefix+'/baseline_W_l2_rms'] = float(np.sqrt(before.mean()))
            result[prefix+'/under_P_W_l2_rms'] = float(np.sqrt(after.mean()))
            result[prefix+'/W_retained_norm_ratio'] = ratio(np.sqrt(after.sum()), np.sqrt(before.sum()))
            result[prefix+'/W_change_norm_ratio'] = ratio(np.sqrt(change.sum()), np.sqrt(before.sum()))
            result[prefix+'/W_projection_retained'] = ratio(natural['W_current_dot_baseline'][indices].sum(), before.sum())
            result[prefix+'/W_code_sum_ratio'] = ratio(natural['W_code_sum'][indices].sum(), natural['baseline_W_code_sum'][indices].sum())
            restoring = data['contributions'][family, site, 'P_restoreW']['restore_delta_l2'][indices]
            result[prefix+'/restore_update_l2_rms'] = rms(restoring)
        result[f'{family}/W_sites/retained_norm_ratio'] = ratio(np.sqrt(energy_after.sum()), np.sqrt(energy_before.sum()))
        result[f'{family}/W_sites/energy_loss_mean'] = float(np.mean(energy_before-energy_after))
        result[f'{family}/link/W_energy_loss_correlation_compensation'] = correlation(energy_before-energy_after, -np.sign(p)*restoration)
    for name in data['families']['source']['effects']:
        source = data['families']['source']['effects'][name][indices]
        native = data['families']['native']['effects'][name][indices]
        result[f'native_source_gap/{name}/rmse'] = rms(native-source)
        result[f'native_source_gap/{name}/signed_mean'] = float(np.mean(native-source))
        result[f'native_source_gap/{name}/correlation'] = correlation(native, source)
    return result


def verify_execution(smoke):
    basis_path = smoke/'execution_basis.npz'
    batch_path = sorted((smoke/'execution_arrays').glob('batch_*.npz'))[0]
    result = dict(run=smoke.as_posix(), basis_sha256=digest(basis_path), batch_sha256=digest(batch_path), selected_sites={})
    with np.load(basis_path) as basis, np.load(batch_path) as arrays:
        for family in FAMILIES:
            candidates = sorted(key.removeprefix(family+'__').removesuffix('__w') for key in basis.files
                if key.startswith(family+'__') and key.endswith('__w') and np.any(basis[key] > 0))
            site = candidates[0]
            prefix = family+'__'+site
            p, w, decoder = [basis[prefix+'__'+key] for key in ('p', 'w', 'decoder')]
            current = arrays[f'{family}__P_restoreW__{site}__code']
            baseline = arrays[f'{family}__none__{site}__code']
            expected = -current*p + (baseline-current)*w
            coefficients = arrays[f'{family}__P_restoreW__{site}__executed_coefficients']
            delta = coefficients@decoder
            offsets = arrays['token_offsets']
            pooled = np.stack([delta[a:b].mean(axis=0) for a, b in zip(offsets[:-1], offsets[1:])])
            saved_mean = arrays[f'{family}__P_restoreW__{site}__delta_mean']
            result['selected_sites'][family] = dict(site=site, valid_tokens=len(current),
                coefficient_formula_max_error=float(np.abs(expected-coefficients).max()),
                coefficients_decoder_vs_saved_delta_mean_max_error=float(np.abs(pooled-saved_mean).max()),
                none_restore_coefficient_max=float(np.abs(arrays[f'{family}__none_restoreW__{site}__executed_coefficients']).max()))
    result['scope'] = '复算一个真实 smoke batch 中 source/native 各一个 W 非零 site 的完整 token 系数和 decoder 更新，并比较保存的逐文档实际 delta 均值；未保存完整 hidden 时不声称逐 token hidden 独立复核。'
    return result


def interval(point, values):
    finite = np.array([x for x in values if x is not None], dtype=float)
    return dict(mean=point, ci95=np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                valid_bootstrap=len(finite))


def analyze(run, smoke, bootstrap, seed):
    data = load(run)
    n = len(data['rows'])
    all_ids = np.arange(n)
    strata = {(label, gender): np.array([i for i, row in enumerate(data['rows'])
        if row['label'] == label and row['gender'] == gender]) for label in (0, 1) for gender in (0, 1)}
    point = statistics(data, all_ids)
    sampled = {key: [] for key in point}
    rng = np.random.default_rng(seed)
    for iteration in range(bootstrap):
        ids = np.concatenate([rng.choice(values, len(values), replace=True) for values in strata.values()])
        values = statistics(data, ids)
        for key, value in values.items():
            sampled[key].append(value)
        if (iteration+1) % 500 == 0:
            print(json.dumps(dict(bootstrap_completed=iteration+1, total=bootstrap)), flush=True)
    statistics_ci = {key: interval(value, sampled[key]) for key, value in point.items()}
    differences = {}
    for key in point:
        if not key.startswith('source/'):
            continue
        other = key.replace('source/', 'native/', 1)
        name = key.removeprefix('source/')
        difference = point[other]-point[key] if point[other] is not None and point[key] is not None else None
        samples = [a-b if a is not None and b is not None else None for a, b in zip(sampled[other], sampled[key])]
        differences[name] = interval(difference, samples)
    documents = []
    for i, row in enumerate(data['rows']):
        families = {}
        for family, value in data['families'].items():
            families[family] = dict(effects={key: float(x[i]) for key, x in value['effects'].items()}, sites={site:
                {key: float(x[i]) for key, x in data['contributions'][family, site, 'P'].items()} for site in data['sites']})
        documents.append(dict(document_sha256=row['document_sha256'], row_id=row['row_id'],
            label=row['label'], gender=row['gender'], families=families))
    return dict(written_at_utc=datetime.now(timezone.utc).isoformat(), run=run.as_posix(), documents=n,
        raw_sha256=digest(run/'metrics.raw.jsonl'), analyzer_sha256=digest(Path(__file__)),
        statistics=statistics_ci, native_minus_source=differences,
        strata={f'label{label}_gender{gender}': dict(documents=len(ids), statistics=statistics(data, ids)) for (label, gender), ids in strata.items()},
        document_values=documents, execution_replay=verify_execution(smoke),
        restore_identity={family: {key: value for key, value in values.items() if key.startswith('identity_')}
                          for family, values in data['families'].items()},
        bootstrap=dict(draws=bootstrap, seed=seed, unit='document',
            method='在 label×gender 四个层内重采样文档，source/native、所有操作和 site 共享抽样。'),
        definitions=dict(P='pronouns', W='associated_words', endpoint='固定原 source profession head 的有符号 logit，正值指向 label 1。',
            P_effect='P-none', W_effect='W-none', W_after_P='PW-P', restore='P_restoreW-P',
            restored_vs_none='P_restoreW-none', interaction='(PW-P)-(W-none)',
            contribution='P 执行过程中各 hook 接收 upstream P 后、执行本地修改前的 D(w*z_current)，与同文档未修改 baseline 比较。',
            norm_ratio='先在文档和有效 token 上汇总平方量，再开平方计算比值；不平均逐文档小分母比值。跨 site 合计表示各 site 能量之和。',
            cancellation='projection_against_P = -sum(restore*P_effect)/sum(P_effect^2)。正值表示沿原 P 效果反向恢复；一表示该投影完全抵消，不能据此认定其他方向也恢复。',
            link='按文档计算 W 各 site 自然贡献能量下降，与 -sign(P_effect)*restore 的 Pearson 相关；该联系为机制观察。',
            scope='已暴露的开发文档、原 source 与原 target2、冻结权重和固定 head。恢复属于多 site 条件操作，不确定唯一中介或新选择能力。'))


def report(result):
    stats = result['statistics']
    def show(key):
        value = stats[key]
        if value['mean'] is None:
            return '未定义'
        lo, hi = value['ci95']
        return f"{value['mean']:.6f} [{lo:.6f}, {hi:.6f}]"
    lines = ['# 自然贡献变化与条件恢复', '',
        f"使用 {result['documents']} 个已有开发文档及固定原 source profession head。W_after_P=PW-P，restore=P_restoreW-P，所有变化保留原 logit 符号。", '',
        '| family | W 自然贡献保留范数比 | 恢复对原 P 的反向投影比例 | 恢复后剩余效果 RMS / P RMS | W 能量下降与补偿输出的相关 |',
        '|---|---:|---:|---:|---:|']
    for family in FAMILIES:
        keys = ('W_sites/retained_norm_ratio', 'restoration/projection_against_P',
                'restoration/restored_effect_rms_over_P', 'link/W_energy_loss_correlation_compensation')
        lines.append('| '+family+' | '+' | '.join(show(f'{family}/{key}') for key in keys)+' |')
    lines += ['', '范数比低于一表示自然贡献的合计能量减少。反向投影比例为正表示恢复沿原 P 输出效果的相反方向变化；剩余效果 RMS 比较衡量恢复后与 none 的距离。', '',
        '| 连续效果 | source 有符号均值 | source RMS | native 有符号均值 | native RMS | native/source 逐文档差异 RMSE |',
        '|---|---:|---:|---:|---:|---:|']
    for effect in ('P', 'W', 'PW', 'W_after_P', 'restore', 'restored_vs_none', 'interaction'):
        keys = [f'{family}/logit/{effect}/{metric}' for family in FAMILIES for metric in ('mean', 'rms')]
        keys += [f'native_source_gap/{effect}/rmse']
        lines.append('| '+effect+' | '+' | '.join(show(key) for key in keys)+' |')
    lines += ['', '| family / site | baseline W L2 RMS | P 后 W L2 RMS | 保留范数比 | W 变化范数比 |', '|---|---:|---:|---:|---:|']
    for family in FAMILIES:
        sites = sorted({key.split('/')[2] for key in stats if key.startswith(f'{family}/site/')})
        for site in sites:
            keys = ('baseline_W_l2_rms', 'under_P_W_l2_rms', 'W_retained_norm_ratio', 'W_change_norm_ratio')
            lines.append('| '+family+'/'+site+' | '+' | '.join(show(f'{family}/site/{site}/{key}') for key in keys)+' |')
    lines += ['', result['bootstrap']['method']+' 区间使用 95% percentile bootstrap。', '',
        result['definitions']['contribution'], '', result['definitions']['scope'], '',
        '同名 JSON 保存逐文档量、label/gender 分层统计、native-source 配对差区间、pooled 机制辅助量及实际数组复算结果。', '']
    return '\n'.join(lines)


def replication(runs, bootstrap, seed):
    datasets = [load(run) for run in runs]
    configs = [json.loads((run/'config.resolved.json').read_text()) for run in runs]
    targets = [config['target_seed'] for config in configs]
    if len(set(targets)) != len(targets):
        raise ValueError('目标 seed 重复')
    reference = datasets[0]
    membership = [(row['document_sha256'], row['label'], row['gender']) for row in reference['rows']]
    for data in datasets[1:]:
        if [(row['document_sha256'], row['label'], row['gender']) for row in data['rows']] != membership:
            raise ValueError('跨目标文档或分层身份不一致')
        for operation in OPERATIONS:
            np.testing.assert_allclose(data['families']['source']['logits'][operation],
                reference['families']['source']['logits'][operation], rtol=0, atol=0)
    n = len(membership)
    combined = dict(rows=[row for data in datasets for row in data['rows']],
        sites=reference['sites'], families={}, contributions={})
    for family in FAMILIES:
        combined['families'][family] = {name: {key: np.concatenate([data['families'][family][name][key]
            for data in datasets]) for key in reference['families'][family][name]} for name in ('effects', 'pooled')}
    for key in reference['contributions']:
        combined['contributions'][key] = {metric: np.concatenate([data['contributions'][key][metric]
            for data in datasets]) for metric in reference['contributions'][key]}
    units = {f'target{target}': data for target, data in zip(targets, datasets)}
    units['fixed_target_cohort'] = combined
    points = {name: statistics(data, np.arange(len(data['rows']))) for name, data in units.items()}
    sampled = {name: {key: [] for key in values} for name, values in points.items()}
    strata = {(label, gender): np.array([i for i, row in enumerate(reference['rows'])
        if row['label'] == label and row['gender'] == gender]) for label in (0, 1) for gender in (0, 1)}
    rng = np.random.default_rng(seed)
    for iteration in range(bootstrap):
        selected = np.concatenate([rng.choice(values, len(values), replace=True) for values in strata.values()])
        for name, data in units.items():
            ids = np.concatenate([selected+i*n for i in range(len(targets))]) if name == 'fixed_target_cohort' else selected
            values = statistics(data, ids)
            for key, value in values.items():
                sampled[name][key].append(value)
        if (iteration+1) % 500 == 0:
            print(json.dumps(dict(replication_bootstrap_completed=iteration+1, total=bootstrap)), flush=True)
    summaries = {name: {key: interval(value, sampled[name][key]) for key, value in values.items()}
                 for name, values in points.items()}
    differences = {}
    for name in units:
        differences[name] = {}
        for key in points[name]:
            if not key.startswith('source/'):
                continue
            native = key.replace('source/', 'native/', 1)
            point = points[name][native]-points[name][key] if points[name][native] is not None and points[name][key] is not None else None
            values = [a-b if a is not None and b is not None else None
                      for a, b in zip(sampled[name][native], sampled[name][key])]
            differences[name][key.removeprefix('source/')] = interval(point, values)
    target_differences = {}
    comparison_keys = ('W_sites/retained_norm_ratio', 'restoration/projection_against_P',
        'restoration/opposes_P_fraction', 'restoration/restored_effect_rms_over_P',
        'conditional/W_after_P_rms_over_W')
    for i, first in enumerate(targets):
        for second in targets[i+1:]:
            name = f'target{first}_minus_target{second}'
            target_differences[name] = {}
            for metric in comparison_keys:
                key = 'native/'+metric
                point = points[f'target{first}'][key]-points[f'target{second}'][key]
                values = np.array(sampled[f'target{first}'][key])-np.array(sampled[f'target{second}'][key])
                target_differences[name][metric] = interval(point, values)
    document_values = []
    for i, row in enumerate(reference['rows']):
        values = {}
        for target, data in zip(targets, datasets):
            effect = data['families']['native']['effects']
            before = sum(data['contributions']['native', site, 'P']['baseline_W_contribution_l2'][i]**2 for site in data['sites'])
            after = sum(data['contributions']['native', site, 'P']['W_contribution_l2'][i]**2 for site in data['sites'])
            values[str(target)] = dict(P=float(effect['P'][i]), W=float(effect['W'][i]),
                W_after_P=float(effect['W_after_P'][i]), restore=float(effect['restore'][i]),
                restored_vs_none=float(effect['restored_vs_none'][i]),
                restore_opposes_P=bool(effect['P'][i]*effect['restore'][i] < 0),
                W_baseline_energy=float(before), W_under_P_energy=float(after))
        document_values.append(dict(document_sha256=row['document_sha256'], label=row['label'], gender=row['gender'], targets=values))
    return dict(written_at_utc=datetime.now(timezone.utc).isoformat(), targets=targets, documents=n,
        scope='同一批128个已暴露开发文档、冻结原权重及各自 native relation 的跨字典复现；文档和 source 共享，固定目标 cohort，不作独立文本确认或 seed 总体推断。',
        inputs=[dict(run=run.as_posix(), target_seed=target, raw_sha256=digest(run/'metrics.raw.jsonl'),
            config_sha256=digest(run/'config.resolved.json')) for run, target in zip(runs, targets)],
        analyzer_sha256=digest(Path(__file__)), statistics=summaries, native_minus_source=differences,
        paired_target_differences=target_differences,
        document_values=document_values,
        restore_identity={str(target): {family: {key: value for key, value in data['families'][family].items()
            if key.startswith('identity_')} for family in FAMILIES} for target, data in zip(targets, datasets)},
        bootstrap=dict(draws=bootstrap, seed=seed, unit='document',
            method='在label×gender层内重采样文档；所有目标、source、site及操作使用同一组文档索引。',
            aggregation='固定目标等权，各文档的目标分量共同进入总量，范数比与投影比均由合计量计算；512个文档×目标记录保留128个文档的依赖。'),
        definitions=dict(W_after_P='PW-P', restore='P_restoreW-P',
            contribution='P运行中hook接收upstream P之后、执行本地修改前的W自然贡献，相对同文档none。',
            projection='-sum(restore*(P-none))/sum((P-none)^2)',
            direction='restore*(P-none)<0；零值单列，不合并为反向。',
            scope='条件恢复提供功能关系的操作证据，不定义唯一中介或新选择能力。'))


def replication_report(result):
    columns = ('W_sites/retained_norm_ratio', 'restoration/projection_against_P',
        'restoration/opposes_P_fraction', 'restoration/restored_effect_rms_over_P',
        'conditional/W_after_P_rms_over_W')
    def show(value):
        if value['mean'] is None:
            return '未定义'
        low, high = value['ci95']
        return f"{value['mean']:.6f} [{low:.6f}, {high:.6f}]"
    lines = ['# 条件恢复的跨字典复现', '', result['scope'], '',
        '| 目标及family | W贡献保留范数比 | 恢复反向投影比例 | 恢复逆向P的文档比例 | 恢复后剩余RMS/P | W_after_P RMS/W RMS |',
        '|---|---:|---:|---:|---:|---:|']
    first = f"target{result['targets'][0]}"
    source = result['statistics'][first]
    lines.append('| 共同source | '+' | '.join(show(source['source/'+key]) for key in columns)+' |')
    for name, values in result['statistics'].items():
        lines.append('| '+name+' native | '+' | '.join(show(values['native/'+key]) for key in columns)+' |')
    lines += ['', '| native减source | W贡献保留范数比 | 恢复反向投影比例 | 恢复逆向P的文档比例 | 恢复后剩余RMS/P | W_after_P RMS/W RMS |',
        '|---|---:|---:|---:|---:|---:|']
    for name, values in result['native_minus_source'].items():
        lines.append('| '+name+' | '+' | '.join(show(values[key]) for key in columns)+' |')
    lines += ['', result['bootstrap']['method'], '', result['bootstrap']['aggregation'], '',
        'W_after_P=PW-P，restore=P_restoreW-P。正的反向投影表示恢复沿原P效果的相反方向变化。', '',
        '同名JSON保存全部连续logit与pooled指标、各site自然贡献、逐目标配对区间及每文档方向。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path)
    parser.add_argument('--smoke', type=Path)
    parser.add_argument('--replication-runs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=9232302)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.md').exists():
        raise FileExistsError(args.output)
    if args.replication_runs:
        result = replication(args.replication_runs, args.bootstrap, args.seed)
        rendered = replication_report(result)
    else:
        if args.run is None or args.smoke is None:
            parser.error('单run分析需要 --run 和 --smoke')
        result = analyze(args.run, args.smoke, args.bootstrap, args.seed)
        rendered = report(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    args.output.with_suffix('.md').write_text(rendered, encoding='utf-8')
    print(json.dumps(dict(documents=result['documents'], output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
