import argparse
import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np


METHODS = ('native_support', 'native_natural', 'native_operation', 'source_raw_path')
CONDITIONS = ('clean', 'conditional')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(run):
    status = json.loads((run/'status.json').read_text())
    if status['status'] != 'PASS':
        raise ValueError(f'运行尚未 PASS {run}')
    membership = json.loads((run/'selection_membership.json').read_text())['rows']
    rows = [row for row in membership if row['selection_split'] == 'evaluation']
    fit = [row for row in membership if row['selection_split'] == 'fit']
    ids = {row['document_sha256']: i for i, row in enumerate(rows)}
    if set(ids) & {row['document_sha256'] for row in fit}:
        raise ValueError('拟合与评价文档重复')
    details = json.loads((run/'selected_members.json').read_text())
    config = json.loads((run/'config.resolved.json').read_text())
    selected = {}
    with np.load(run/'candidate_basis.npz') as basis, np.load(run/'selection_parameters.npz') as parameters:
        for method in METHODS:
            selected[method] = {}
            for site, entry in details[method].items():
                candidates = basis[site+'__candidates']
                binary = parameters[method+'__'+site+'__binary']
                continuous = parameters[method+'__'+site+'__continuous']
                np.testing.assert_array_equal(candidates, entry['candidate_ids'])
                np.testing.assert_array_equal(binary, entry['binary_mask'])
                np.testing.assert_array_equal(continuous, entry['continuous_fit'])
                if not np.isin(binary, [0, 1]).all():
                    raise ValueError(f'实际删除 mask 非二值 {method} {site}')
                chosen = candidates[binary == 1].tolist()
                if set(chosen) != set(entry['selected_ids']) or len(chosen) != config['members_by_site'][site]:
                    raise ValueError(f'feature IDs 与执行 mask 不一致 {method} {site}')
                selected[method][site] = dict(candidate_count=len(candidates), selected_ids=entry['selected_ids'],
                    members=len(chosen), binary_key=method+'__'+site+'__binary',
                    continuous_key=method+'__'+site+'__continuous',
                    parameters_path=(run/'selection_parameters.npz').as_posix(),
                    membership_path=(run/'selected_members.json').as_posix())
    n = len(rows)
    arrays = {method: {key: np.full((2, n), np.nan) for key in ('logit', 'actual_effect', 'prediction')}
              for method in ('source',)+METHODS}
    baseline = np.full((2, n), np.nan)
    source_effect = np.full((2, n), np.nan)
    records = {}
    with (run/'metrics.raw.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] != 'conditional_member_effect':
                continue
            ci, di, method = row['condition'], ids[row['component']], row['method']
            key = (method, ci, di)
            if key in records or row['split'] != 'evaluation':
                raise ValueError(f'逐文档响应身份不一致 {key}')
            reference = rows[di]
            if (row['row_id'], row['label'], row['gender']) != (reference['row_id'], reference['label'], reference['gender']):
                raise ValueError(f'逐文档标签不一致 {key}')
            records[key] = row
            for field in arrays[method]:
                arrays[method][field][ci, di] = row[field]
            if method == 'source':
                baseline[ci, di] = row['baseline_logit']
                source_effect[ci, di] = row['source_effect']
    seen = set()
    for path in sorted((run/'evaluation_arrays').glob('batch_*.npz')):
        with np.load(path) as batch:
            for local, global_index in enumerate(batch['row_indices']):
                document = str(batch['document_sha256'][local])
                row = membership[int(global_index)]
                if row['document_sha256'] != document or row['row_id'] != int(batch['row_ids'][local]):
                    raise ValueError(f'NPZ文档身份不一致 {path}')
                if document in seen:
                    raise ValueError(f'NPZ文档重复 {document}')
                seen.add(document)
                di = ids[document]
                for ci in (0, 1):
                    np.testing.assert_equal(baseline[ci, di], float(batch[f'c{ci}__baseline'][local]))
                    for method in arrays:
                        raw = records[method, ci, di]
                        actual_logit = batch[f'c{ci}__{method}__logits'][local]
                        np.testing.assert_equal(arrays[method]['logit'][ci, di], float(actual_logit))
                        np.testing.assert_equal(raw['baseline_logit'], baseline[ci, di])
                        np.testing.assert_equal(raw['source_logit'], float(batch[f'c{ci}__source'][local]))
                        np.testing.assert_equal(raw['source_effect'], source_effect[ci, di])
                        np.testing.assert_equal(raw['actual_effect'], float(actual_logit-batch[f'c{ci}__baseline'][local]))
                        if raw['prediction'] != int(actual_logit > 0) or raw['correct'] != (raw['prediction'] == row['label']):
                            raise ValueError(f'prediction 与实际 logit 不一致 {method} {document}')
    if seen != set(ids) or any(not np.isfinite(value).all() for data in arrays.values() for value in data.values()):
        raise ValueError('评价响应不完整')
    source_scale = float(np.sqrt(np.mean(source_effect**2)))
    saved = json.loads((run/'MEMBER_RESULTS.json').read_text())
    np.testing.assert_allclose(source_scale, saved['source_rms_scale'], rtol=1e-12, atol=0)
    return dict(rows=rows, fit_documents=len(fit), arrays=arrays, source_effect=source_effect,
                baseline=baseline, selections=selected, records=records, source_scale=source_scale)


def statistics(data, indices):
    labels = np.array([row['label'] for row in data['rows']])[indices]
    genders = np.array([row['gender'] for row in data['rows']])[indices]
    reference = data['source_effect'][:, indices]
    scale = float(np.sqrt(np.mean(reference**2)))
    values = dict(source_rms_scale=scale)
    group_masks = [(labels == label) & (genders == gender) for label in (0, 1) for gender in (0, 1)]
    for method, arrays in data['arrays'].items():
        effects = arrays['actual_effect'][:, indices]
        logits = arrays['logit'][:, indices]
        for name, conditions in (('joint', [0, 1]), ('clean', [0]), ('conditional', [1])):
            actual, source = effects[conditions], reference[conditions]
            error = actual-source
            prefix = method+'/'+name
            rmse = float(np.sqrt(np.mean(error**2)))
            values[prefix+'/rmse'] = rmse
            values[prefix+'/nrmse_common_source_scale'] = rmse/scale
            values[prefix+'/actual_effect_mean'] = float(actual.mean())
            values[prefix+'/actual_effect_rms'] = float(np.sqrt(np.mean(actual**2)))
            values[prefix+'/source_effect_mean'] = float(source.mean())
            values[prefix+'/source_effect_rms'] = float(np.sqrt(np.mean(source**2)))
            correct = (logits[conditions] > 0) == labels[None]
            values[prefix+'/profession_accuracy'] = float(correct.mean())
            values[prefix+'/worst_group_accuracy'] = min(float(correct[:, mask].mean()) for mask in group_masks)
            for label in (0, 1):
                for gender in (0, 1):
                    mask = (labels == label) & (genders == gender)
                    values[prefix+f'/label{label}_gender{gender}_accuracy'] = float(correct[:, mask].mean())
        clean_rms = np.sqrt(np.mean(effects[0]**2))
        values[method+'/conditional_to_clean_effect_rms'] = float(np.sqrt(np.mean(effects[1]**2))/clean_rms) if clean_rms > 0 else None
    for ci, condition in enumerate(CONDITIONS):
        correct = (data['baseline'][ci, indices] > 0) == labels
        values['baseline/'+condition+'/profession_accuracy'] = float(correct.mean())
        values['baseline/'+condition+'/worst_group_accuracy'] = min(float(correct[mask].mean()) for mask in group_masks)
    return values


def interval(point, samples):
    finite = np.array([value for value in samples if value is not None], dtype=float)
    return dict(mean=point, ci95=np.quantile(finite, [.025, .975]).tolist() if len(finite) else None)


def analyze(run, bootstrap, seed):
    data = read(run)
    n = len(data['rows'])
    strata = [np.array([i for i, row in enumerate(data['rows']) if row['label'] == label and row['gender'] == gender])
              for label in (0, 1) for gender in (0, 1)]
    points = statistics(data, np.arange(n))
    sampled = {key: [] for key in points}
    rng = np.random.default_rng(seed)
    for iteration in range(bootstrap):
        ids = np.concatenate([rng.choice(group, len(group), replace=True) for group in strata])
        for key, value in statistics(data, ids).items():
            sampled[key].append(value)
        if (iteration+1) % 500 == 0:
            print(json.dumps(dict(bootstrap_completed=iteration+1, total=bootstrap)), flush=True)
    contrasts = {}
    suffixes = [key.removeprefix(METHODS[0]+'/') for key in points if key.startswith(METHODS[0]+'/')]
    for first, second in combinations(METHODS, 2):
        contrasts[first+'_minus_'+second] = {}
        for suffix in suffixes:
            a, b = first+'/'+suffix, second+'/'+suffix
            delta = points[a]-points[b] if points[a] is not None and points[b] is not None else None
            samples = [x-y if x is not None and y is not None else None for x, y in zip(sampled[a], sampled[b])]
            contrasts[first+'_minus_'+second][suffix] = interval(delta, samples)
    document_values = []
    for (method, condition, index), row in sorted(data['records'].items()):
        document_values.append(dict(document_sha256=row['component'], row_id=row['row_id'],
            label=row['label'], gender=row['gender'], method=method, condition=CONDITIONS[condition],
            baseline_logit=row['baseline_logit'], source_logit=row['source_logit'], logit=row['logit'],
            prediction=row['prediction'], actual_effect=row['actual_effect'], source_effect=row['source_effect'],
            profession_correct=row['correct'], source_correct=row['source_correct'],
            selection_reference=method if method in METHODS else 'published_source_W'))
    return dict(written_at_utc=datetime.now(timezone.utc).isoformat(), run=run.as_posix(),
        fit_documents=data['fit_documents'], evaluation_documents=n,
        source_rms_scale=data['source_scale'], statistics={key: interval(value, sampled[key]) for key, value in points.items()},
        paired_contrasts=contrasts, selections=data['selections'], document_values=document_values,
        identities={name: dict(path=(run/name).as_posix(), sha256=digest(run/name)) for name in (
            'metrics.raw.jsonl', 'selected_members.json', 'selection_parameters.npz', 'candidate_basis.npz',
            'selection_membership.json', 'config.resolved.json', 'selection_protocol.json')},
        analyzer_sha256=digest(Path(__file__)),
        bootstrap=dict(draws=bootstrap, seed=seed, unit='evaluation document',
            method='在label×gender层内配对重采样评价文档；所有方法及两个条件共用抽样。每次重新计算共同source尺度及worst-group。'),
        definitions=dict(clean='正常条件下目标W删除logit减none；源参考为source W-none。',
            conditional='source P背景下追加目标W删除logit减同一source P baseline；源参考为source PW-P。',
            primary='sqrt(sum_{doc,condition}(actual_effect-source_effect)^2 / sum_{doc,condition}source_effect^2)',
            scale='两个条件和全部评价文档的source effect共同RMS；单条件nRMSE也用此共同尺度。单条件RMSE和实际effect另列。',
            selection='各处固定额度的完整自然成员删除，连续权重只用于支持选择。每条prediction通过method关联保存的feature IDs与实际binary mask。',
            accuracy='使用固定原source职业head的全体与label×gender组准确率；worst-group为四组最小值，未用于选择或调参。',
            scope='历史已暴露开发文档按hash分开的fit/evaluation；仅评价部分进入推断，不视为独立文本确认。'))


def report(result):
    stats = result['statistics']
    def show(key):
        value = stats[key]
        if value['mean'] is None:
            return '未定义'
        low, high = value['ci95']
        return f"{value['mean']:.6f} [{low:.6f}, {high:.6f}]"
    lines = ['# 条件功能关系下的完整自然成员选择', '',
        f"拟合文档 {result['fit_documents']} 篇，评价文档 {result['evaluation_documents']} 篇。共同source响应尺度为 {result['source_rms_scale']:.6f}。", '',
        '| 方法 | 两条件共同尺度nRMSE | clean RMSE | conditional RMSE | clean实际effect RMS | conditional实际effect RMS |',
        '|---|---:|---:|---:|---:|---:|']
    keys = ('joint/nrmse_common_source_scale', 'clean/rmse', 'conditional/rmse',
            'clean/actual_effect_rms', 'conditional/actual_effect_rms')
    for method in ('source',)+METHODS:
        lines.append('| '+method+' | '+' | '.join(show(method+'/'+key) for key in keys)+' |')
    lines += ['', '| 方法 | clean有符号effect均值 | conditional有符号effect均值 |', '|---|---:|---:|']
    for method in ('source',)+METHODS:
        lines.append('| '+method+' | '+show(method+'/clean/actual_effect_mean')+' | '+show(method+'/conditional/actual_effect_mean')+' |')
    lines += ['', '| 方法 | clean职业准确率 | conditional职业准确率 | clean worst-group | conditional worst-group |',
        '|---|---:|---:|---:|---:|']
    for method in ('baseline', 'source')+METHODS:
        keys = ('clean/profession_accuracy', 'conditional/profession_accuracy',
                'clean/worst_group_accuracy', 'conditional/worst_group_accuracy')
        lines.append('| '+method+' | '+' | '.join(show(method+'/'+key) for key in keys)+' |')
    lines += ['', '| 配对差 | 两条件nRMSE | clean RMSE | conditional RMSE |', '|---|---:|---:|---:|']
    for name, values in result['paired_contrasts'].items():
        cells = []
        for key in ('joint/nrmse_common_source_scale', 'clean/rmse', 'conditional/rmse'):
            value = values[key]
            low, high = value['ci95']
            cells.append(f"{value['mean']:.6f} [{low:.6f}, {high:.6f}]")
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines += ['', result['definitions']['clean'], '', result['definitions']['conditional'], '',
        result['bootstrap']['method']+' 区间为95% percentile bootstrap。', '',
        result['definitions']['selection'], '', result['definitions']['scope'], '',
        '同名JSON保存逐文档prediction、实际响应、方法与支持集合关联、各组准确率及全部配对差。', '']
    return '\n'.join(lines)


def costs(runs):
    records = []
    for run in runs:
        summary = json.loads((run/'metrics.summary.json').read_text())
        status = json.loads((run/'status.json').read_text())
        member = json.loads((run/'MEMBER_RESULTS.json').read_text())
        records.append(dict(run=run.as_posix(), status=status, driver_seconds=summary['wall_seconds'],
            process_cpu_seconds=summary['process_cpu_seconds'], sequence_forwards=summary['sequence_forwards'],
            token_forwards=summary['token_forwards'], backward_sequences=member['backward_sequences'],
            peak_cuda_allocated_bytes=summary['peak_allocated_bytes'],
            directory_bytes=sum(path.stat().st_size for path in run.rglob('*') if path.is_file())))
    total = {key: sum(row[key] for row in records) for key in ('driver_seconds', 'process_cpu_seconds',
        'sequence_forwards', 'token_forwards', 'backward_sequences', 'directory_bytes')}
    total['peak_cuda_allocated_bytes'] = max(row['peak_cuda_allocated_bytes'] for row in records)
    return dict(runs=records, total=total,
        scope='逐run真实summary/status及目录文件大小；driver包含加载、拟合、执行和保存，非纯GPU时间。')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=9232303)
    parser.add_argument('--cost-runs', nargs='+', type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.md').exists():
        raise FileExistsError(args.output)
    result = analyze(args.run, args.bootstrap, args.seed)
    if args.cost_runs:
        result['cost'] = costs(args.cost_runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    args.output.with_suffix('.md').write_text(report(result), encoding='utf-8')
    print(json.dumps(dict(evaluation_documents=result['evaluation_documents'], output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
