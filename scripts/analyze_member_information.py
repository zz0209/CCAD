from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def identity(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    status = json.loads((args.run / 'status.json').read_text())
    assert status['status'] == 'PASS', status
    config = json.loads((args.run / 'config.resolved.json').read_text())
    index = json.loads((args.run / 'INDEX.json').read_text())
    methods = [method for method in index['methods'] if method not in ['none', 'source']]
    seeds, rows, queries = index['seeds'], index['rows'], index['queries']
    sources, cleans, outputs, identities = [], [], [], []
    for seed in seeds:
        path = args.run / f'responses_t{seed}.npz'
        raw = np.load(path)
        sources.append(raw['source'].astype(np.float64))
        cleans.append(raw['none'].astype(np.float64))
        outputs.append(np.stack([raw[method].astype(np.float64) for method in methods]))
        identities.append(identity(path))
    source, clean = np.stack(sources), np.stack(cleans)
    values = np.stack(outputs, axis=1)
    squared_error = np.square(values - source[None])
    squared_effect = np.square(source - clean)
    families = {'member_subsets': [i for i, q in enumerate(queries) if index['families'][q] == 'member_subsets'],
                'participation': [i for i, q in enumerate(queries) if index['families'][q] in ['interior', 'boundary']],
                'endpoints': [i for i, q in enumerate(queries) if index['families'][q] == 'endpoints']}
    assert len(families['member_subsets']) == 12
    verbs = sorted({row['verb'] for row in rows})
    nouns = sorted({row['noun'] for row in rows})
    vi = np.array([verbs.index(row['verb']) for row in rows])
    ni = np.array([nouns.index(row['noun']) for row in rows])
    query_energy = squared_effect.mean(-1)
    query_mse = squared_error.mean(-1)
    point_by_query = np.sqrt(np.divide(query_mse, query_energy[None],
                                      out=np.full_like(query_mse, np.nan), where=query_energy[None] > 0))
    by_target = np.stack([np.sqrt(squared_error[:, :, positions].sum((2, 3)) /
                                  squared_effect[:, positions].sum((1, 2))[None])
                         for positions in families.values()], axis=-1)
    assert np.isfinite(by_target).all()
    point = by_target.mean(1)
    random = np.random.default_rng(config['bootstrap_seed'])
    bootstrap = np.empty((config['bootstrap'], len(methods), len(families)))
    for draw in range(config['bootstrap']):
        verb_count = random.multinomial(len(verbs), np.ones(len(verbs)) / len(verbs))
        noun_count = random.multinomial(len(nouns), np.ones(len(nouns)) / len(nouns))
        weight = verb_count[vi] * noun_count[ni]
        assert weight.sum() > 0
        target_draw = random.integers(len(seeds), size=len(seeds))
        numerator = (squared_error * weight).sum(-1)
        denominator = (squared_effect * weight).sum(-1)
        family_errors = []
        for positions in families.values():
            energy = denominator[:, positions].sum(-1)
            assert (energy > 0).all()
            error = np.sqrt(numerator[:, :, positions].sum(-1) / energy[None])
            family_errors.append(error[:, target_draw].mean(1))
        bootstrap[draw] = np.stack(family_errors, axis=1)
        if (draw + 1) % 500 == 0:
            print(json.dumps({'bootstrap_completed': draw + 1, 'bootstrap_total': config['bootstrap']}), flush=True)
    summary = {method: {family: {'mean': float(point[mi, fi]),
                                'ci95': np.quantile(bootstrap[:, mi, fi], [.025, .975]).tolist(),
                                'by_target': {str(seed): float(by_target[mi, ti, fi])
                                              for ti, seed in enumerate(seeds)}}
                        for fi, family in enumerate(families)} for mi, method in enumerate(methods)}
    original = methods.index('program')
    differences = {method: {family: {'mean': float(point[mi, fi] - point[original, fi]),
                                    'ci95': np.quantile(bootstrap[:, mi, fi] - bootstrap[:, original, fi], [.025, .975]).tolist()}
                            for fi, family in enumerate(families)}
                   for mi, method in enumerate(methods) if method != 'program'}
    role_effects = {}
    for name in ['source'] + methods:
        array = source if name == 'source' else values[methods.index(name)]
        role_effects[name] = {role: {query: float((clean[:, queries.index(query), selected] -
                                                  array[:, queries.index(query), selected]).mean())
                                    for query in ['predicate', 'object', 'full']}
                             for role in ['predicate', 'object']
                             for selected in [np.array([row['role'] == role for row in rows])]}
    costs = []
    cost_directories = {args.run, *args.run.parent.glob('MEMBER_INFORMATION_*_20260924')}
    for run in sorted(cost_directories):
        path = run / 'metrics.summary.json'
        if path.exists():
            metric = json.loads(path.read_text())
            costs.append({'run': str(run), 'status': metric['status'],
                          **{key: metric.get(key) for key in ['wall_seconds', 'process_cpu_seconds',
                              'sequence_forwards', 'token_forwards', 'peak_allocated_bytes']},
                          'asset_bytes': sum(path.stat().st_size for path in run.rglob('*') if path.is_file())})
    total = {key: sum(row[key] for row in costs if row[key] is not None)
             for key in ['wall_seconds', 'process_cpu_seconds', 'sequence_forwards', 'token_forwards', 'asset_bytes']}
    total['peak_allocated_bytes'] = max(row['peak_allocated_bytes'] or 0 for row in costs)
    result = {'written_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS',
              'run': str(args.run), 'script': identity(Path(__file__)), 'inputs': identities,
              'source_index': identity(args.run / 'INDEX.json'), 'methods': methods, 'target_seeds': seeds,
              'families': families, 'rows': len(rows), 'verbs': verbs, 'nouns': nouns,
              'summary': summary, 'difference_from_program': differences, 'role_effects': role_effects,
              'normalization': '每个target内对固定请求族的全部请求与文本汇总误差能量/source效应能量后开方，再对target等权平均。真实零作用请求保留其全部误差；逐请求nRMSE在source能量为零时未定义。',
              'absolute_errors': {method: {family: {'rmse_by_target': np.sqrt(squared_error[mi, :, positions].mean((0, 2))).tolist()}
                         for family, positions in families.items()} for mi, method in enumerate(methods)},
              'source_effect': {family: {'rms_by_target': np.sqrt(squared_effect[:, positions].mean((1, 2))).tolist(),
                           'zero_response_fraction_by_target': (squared_effect[:, positions] == 0).mean((1, 2)).tolist(),
                           'zero_energy_requests_by_target': {str(seed): [queries[qi] for qi in positions if query_energy[ti, qi] == 0]
                                                            for ti, seed in enumerate(seeds)}}
                         for family, positions in families.items()},
              'per_request': {query: {'source_energy': query_energy[:, qi].tolist(),
                    'methods': {method: {'absolute_rmse': np.sqrt(query_mse[mi, :, qi]).tolist(),
                        'nrmse': [float(value) if np.isfinite(value) else None for value in point_by_query[mi, :, qi]]}
                        for mi, method in enumerate(methods)}} for qi, query in enumerate(queries)},
              'inference': '共同抽取四个固定target及交叉verb/noun；四个句式、全部12成员请求及所有方法保持配对。请求集合固定，条件于公开source解释和本次词汇总体。',
              'bootstrap': config['bootstrap'], 'bootstrap_seed': config['bootstrap_seed'],
              'group_sum_max_difference': index['group_sum_max_difference'],
              'capacity_violation': index['capacity_violation'], 'cost_runs': costs, 'cost_total': total,
              'interpretation_boundary': '冻结执行中的成员信息干预；未增加训练模块，也未检验推理加速或唯一语义识别。'}
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'ARRAYS.npz', responses=values, source=source, clean=clean,
                        point_by_query=point_by_query, bootstrap=bootstrap)
    (args.output / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# 冻结程序中的组内成员信息', '',
             f'确认输入{len(rows)}条，target为{seeds}；所有参数固定。', '',
             '| 执行 | 成员子集nRMSE | 参与请求nRMSE | 完整语义请求nRMSE |', '|---|---:|---:|---:|']
    for method in methods:
        lines.append('| ' + method + ' | ' + ' | '.join(f'{summary[method][family]["mean"]:.6f}' for family in families) + ' |')
    lines.extend(['', '主要差值为份额方法减原program，正值说明原成员列误差更低。'])
    for method in ['group_uniform', 'source_norm_share']:
        contrast = differences[method]['member_subsets']
        lines.append(f'- {method}: {contrast["mean"]:.6f}, 95% CI {contrast["ci95"]}.')
    lines.extend(['', result['normalization'], '', result['inference'], '', result['interpretation_boundary'], '',
                  f'实际运行时间{total["wall_seconds"]:.3f}秒；新增训练更新为零；峰值CUDA {total["peak_allocated_bytes"]} bytes。'])
    (args.output / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    latex = ['\\begin{tabular}{lrrr}', '\\toprule',
             'Execution & Member subsets & Participation & Endpoints \\\\', '\\midrule']
    labels = {'initial': 'Initial program', 'program': 'Trained member columns', 'group_uniform': 'Uniform within groups',
              'source_norm_share': 'Source-contribution shares', 'readout_initial': 'Source-direction readout',
              'readout_trained': 'Trained-dictionary readout'}
    for method in methods:
        latex.append(labels[method] + ' & ' + ' & '.join(f'{summary[method][family]["mean"]:.3f}' for family in families) + ' \\\\')
    latex.extend(['\\bottomrule', '\\end{tabular}'])
    (args.output / 'member_information_table.tex').write_text('\n'.join(latex) + '\n', encoding='utf-8')
    print(json.dumps({'summary': summary, 'difference_from_program': differences,
                      'cost_total': total}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
