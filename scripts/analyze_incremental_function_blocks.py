import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


PARTS = ['verb', 'number', 'gender']
TASKS = ['regular_plural_subject_verb_agreement_1', 'anaphor_number_agreement', 'anaphor_gender_agreement']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def estimate(values):
    if not np.isfinite(values[0]):
        return dict(value=None, ci95=None)
    finite = values[1:][np.isfinite(values[1:])]
    return dict(value=float(values[0]), ci95=np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                bootstrap_defined=int(len(finite)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--query-order', nargs=7, help='Explicit audited axis order for historical panels without query_order')
    args = parser.parse_args()
    assert not args.output.exists()
    assert read(args.run/'status.json')['status'] == 'PASS'
    panel, methods = read(args.run/'panel.json'), read(args.run/'methods.json')
    methods = {key: dict(value, stage=int(str(value['stage']).removeprefix('stage'))) for key, value in methods.items()}
    rows, queries = panel['rows'], panel['queries']
    names = panel.get('query_order')
    if names is None:
        assert args.query_order is not None, 'Historical panel requires explicit --query-order verified from named raw records'
        names = args.query_order
    elif args.query_order is not None:
        assert names == args.query_order
    assert set(queries) == set(names)
    requests = np.array([queries[name] for name in names])
    assert len(names) == 7 and requests.shape == (7, 3)
    assert set(row['task'] for row in rows) == set(TASKS)
    tasks = {task: np.array([i for i, row in enumerate(rows) if row['task'] == task]) for task in TASKS}
    arrays = dict(np.load(args.run/'responses.npz', allow_pickle=False))
    if 'query_order' in arrays:
        assert arrays['query_order'].tolist() == names
    clean, source = arrays['clean'], arrays['source']
    assert clean.shape == (len(rows),) and source.shape == (7, len(rows))
    assert np.isfinite(source).all() and np.isfinite(clean).all()
    assert set(methods) <= set(arrays)
    targets = sorted({item['target_seed'] for item in methods.values() if item['target_seed'] is not None})
    arrivals = sorted({item['arrival'] for item in methods.values() if item['arrival'] is not None})
    cells, observed, references, base = [], [], [], []
    for key, metadata in methods.items():
        values = arrays[key]
        valid = np.asarray(metadata['valid_queries'], dtype=bool)
        assert values.shape == source.shape and valid.shape == (7,)
        assert np.isfinite(values[valid]).all() and np.isnan(values[~valid]).all()
        if metadata['stage'] == 1:
            assert np.array_equal(valid, requests[:, PARTS.index(metadata['arrival'])] == 0)
        for task, ids in tasks.items():
            for qi in np.flatnonzero(valid):
                cells.append(dict(key=key, method=metadata['method'], stage=metadata['stage'],
                                  target_seed=metadata['target_seed'], arrival=metadata['arrival'],
                                  task=task, query=names[qi], query_index=int(qi)))
                observed.append(values[qi])
                references.append(source[qi])
                base.append(clean)
    y, reference, baseline = map(np.array, (observed, references, base))
    task_indices = {task: np.array([i for i, cell in enumerate(cells) if cell['task'] == task]) for task in TASKS}
    moments = dict(squared_error=(y-reference)**2, source_energy=(reference-baseline)**2,
                   actual_energy=(y-baseline)**2, accuracy=(y > 0).astype(float),
                   source_accuracy=(reference > 0).astype(float), binary_agreement=((y > 0) == (reference > 0)).astype(float),
                   clean_correct=((baseline > 0)).astype(float),
                   clean_correct_retained=((baseline > 0) & (y > 0)).astype(float),
                   baseline_flip=((y > 0) != (baseline > 0)).astype(float))
    parent_keys = {}
    drift_squared = np.full_like(y, np.nan)
    drift_binary = np.full_like(y, np.nan)
    for key, item in methods.items():
        if item['stage'] != 2 or item['method'] not in ['global_frozen', 'group_frozen', 'global_replay']:
            continue
        prior_method = 'group' if item['method'] == 'group_frozen' else 'global'
        prior = [name for name, metadata in methods.items() if metadata['stage'] == 1 and
                 metadata['method'] == prior_method and metadata['target_seed'] == item['target_seed'] and metadata['arrival'] == item['arrival']]
        assert len(prior) == 1, (key, prior)
        parent_keys[key] = prior[0]
    for i, cell in enumerate(cells):
        if cell['key'] in parent_keys:
            parent = parent_keys[cell['key']]
            qi = cell['query_index']
            if methods[parent]['valid_queries'][qi]:
                drift_squared[i] = (arrays[cell['key']][qi]-arrays[parent][qi])**2
                drift_binary[i] = ((arrays[cell['key']][qi] > 0) == (arrays[parent][qi] > 0)).astype(float)
    moments.update(stage_change_squared=drift_squared, stage_binary_agreement=drift_binary)
    groups = {}
    for i, cell in enumerate(cells):
        groups[f"individual/{cell['key']}/{cell['task']}/{cell['query']}"] = [i]
    # 按固定目标、接收顺序、任务及请求等权；共享参照不计为额外目标。
    for method in sorted({item['method'] for item in methods.values()}):
        for stage in sorted({item['stage'] for item in methods.values() if item['method'] == method}):
            candidate = [i for i, cell in enumerate(cells) if cell['method'] == method and cell['stage'] == stage]
            for endpoint in ['old_own', 'new_own', 'new_combinations', 'all_valid']:
                chosen = []
                for i in candidate:
                    cell = cells[i]
                    arrival = cell['arrival']
                    query = requests[cell['query_index']]
                    task_part = PARTS[TASKS.index(cell['task'])]
                    if endpoint == 'all_valid':
                        chosen.append(i)
                    elif arrival is not None:
                        if endpoint == 'new_own' and cell['query'] == arrival and task_part == arrival:
                            chosen.append(i)
                        if endpoint == 'old_own' and cell['query'] == task_part and task_part != arrival:
                            chosen.append(i)
                        if endpoint == 'new_combinations' and query[PARTS.index(arrival)] and query.sum() > 1:
                            chosen.append(i)
                    else:
                        # source/native按本次实际接收顺序复用，保持同一请求集合及权重。
                        for shared_arrival in arrivals:
                            if endpoint == 'new_own' and cell['query'] == task_part == shared_arrival:
                                chosen.append(i)
                            if endpoint == 'old_own' and cell['query'] == task_part and task_part != shared_arrival:
                                chosen.append(i)
                            if endpoint == 'new_combinations' and query.sum() > 1 and query[PARTS.index(shared_arrival)]:
                                chosen.append(i)
                if chosen:
                    prefix = f'{method}/stage{stage}/{endpoint}'
                    groups[prefix] = chosen
                    for arrival in arrivals:
                        subset = [i for i in chosen if cells[i]['arrival'] == arrival]
                        if subset:
                            groups[prefix+'/arrival_'+arrival] = subset
                    for target in targets:
                        subset = [i for i in chosen if cells[i]['target_seed'] == target]
                        if subset:
                            groups[prefix+'/target_'+str(target)] = subset
    groups = {key: np.asarray(ids) for key, ids in groups.items()}
    rng = np.random.default_rng(9232711)
    outputs = {}
    for iteration in range(args.bootstrap+1):
        means = {key: np.empty(len(cells)) for key in moments}
        for task, ids in tasks.items():
            selected = ids if iteration == 0 else rng.choice(ids, len(ids), replace=True)
            ci = task_indices[task]
            for key, value in moments.items():
                means[key][ci] = value[np.ix_(ci, selected)].mean(axis=1)
        for group, indices in groups.items():
            mse, energy = means['squared_error'][indices], means['source_energy'][indices]
            values = {key: float(means[key][indices].mean()) for key in ('accuracy', 'source_accuracy', 'binary_agreement', 'baseline_flip')}
            values.update(rmse=float(np.sqrt(mse.mean())), source_effect_rms=float(np.sqrt(energy.mean())),
                          actual_effect_rms=float(np.sqrt(means['actual_energy'][indices].mean())),
                          nrmse=float(np.sqrt(np.mean(mse/energy))) if np.all(energy > 0) else np.nan)
            denominator = means['clean_correct'][indices]
            values['clean_correct_retention'] = float(np.mean(means['clean_correct_retained'][indices]/denominator)) if np.all(denominator > 0) else np.nan
            if np.isfinite(means['stage_change_squared'][indices]).all():
                values['stage_change_rmse'] = float(np.sqrt(means['stage_change_squared'][indices].mean()))
                values['stage_change_nrmse'] = float(np.sqrt(np.mean(means['stage_change_squared'][indices]/energy))) if np.all(energy > 0) else np.nan
                values['stage_binary_agreement'] = float(means['stage_binary_agreement'][indices].mean())
            for metric, value in values.items():
                outputs.setdefault(group+'/'+metric, np.empty(args.bootstrap+1))[iteration] = value
        if iteration and iteration % 250 == 0:
            print(dict(bootstrap_completed=iteration, bootstrap_total=args.bootstrap), flush=True)
    contrasts = {}
    focal_methods = ['global_frozen', 'group_frozen', 'global_replay']
    for first in focal_methods:
        for second in focal_methods+['raw', 'native']:
            if first == second:
                continue
            second_stage = 0 if second == 'native' else 2
            for key in list(outputs):
                prefix = first+'/stage2/'
                if key.startswith(prefix):
                    suffix = key[len(prefix):]
                    other = second+f'/stage{second_stage}/'+suffix
                    if other in outputs:
                        contrasts[first+'_minus_'+second+'/'+suffix] = estimate(outputs[key]-outputs[other])
    drift = {}
    for key, item in methods.items():
        if item['stage'] != 2 or item['method'] not in focal_methods:
            continue
        old_key = parent_keys[key]
        valid = np.asarray(methods[old_key]['valid_queries'], bool)
        current_vector = np.load(item['vectors'])['delta']
        old_vector = np.load(methods[old_key]['vectors'])['delta']
        drift[key] = dict(parent=old_key, requests={})
        for qi in np.flatnonzero(valid):
            delta = arrays[key][qi]-arrays[old_key][qi]
            for task, ids in tasks.items():
                drift[key]['requests'][task+'/'+names[qi]] = dict(margin_change_rmse=estimate(outputs[f'individual/{key}/{task}/{names[qi]}/stage_change_rmse']),
                    binary_agreement=float(np.mean((arrays[key][qi, ids] > 0) == (arrays[old_key][qi, ids] > 0))))
        old_parts = [i for i, part in enumerate(PARTS) if part != item['arrival']]
        drift[key]['old_vector_change_rms'] = float(np.sqrt(np.mean((current_vector[old_parts]-old_vector[old_parts])**2)))
    args.output.mkdir(parents=True)
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), rows=len(rows),
                  task_counts={task: len(ids) for task, ids in tasks.items()}, fixed_targets=targets,
                  fixed_arrivals=arrivals, methods=methods, queries=queries, query_order=names, statistics={key: estimate(value) for key, value in outputs.items()},
                  paired_contrasts=contrasts, stage_drift=drift,
                  normalization='Source intervention response energy within each task and query; equal normalized squared errors across the fixed selected cells, then square root; zero-source cells retain unnormalized errors',
                  bootstrap=dict(draws=args.bootstrap, seed=9232711, unit='Sentence pair within task; shared indices across methods, requests, fixed targets and arrival orders'),
                  identities={name: identity(args.run/name) for name in ('responses.npz', 'methods.json', 'panel.json', 'config.resolved.json')},
                  analyzer=identity(Path(__file__)))
    (args.output/'RESULTS.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    np.savez_compressed(args.output/'row_responses.npz', **dict(arrays, query_order=np.array(names)))
    lines = ['# 逐项接收功能解释的实际调用', '', '| 方法 | 新功能 nRMSE | 旧功能 nRMSE | 新组合 nRMSE | 新组合一致率 |', '|---|---:|---:|---:|---:|']
    for method in focal_methods+['raw', 'native']:
        stage = 0 if method == 'native' else 2
        keys = ['new_own/nrmse', 'old_own/nrmse', 'new_combinations/nrmse', 'new_combinations/binary_agreement']
        values = [result['statistics'].get(method+f'/stage{stage}/'+key, {}).get('value') for key in keys]
        lines.append('| '+method+' | '+' | '.join('未定义' if value is None else f'{value:.6f}' for value in values)+' |')
    lines += ['', 'margin为完整句good−bad。二值一致率比较目标与source干预后的正负；语法正确率和clean正确样本保持率分别保存。', '']
    (args.output/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
