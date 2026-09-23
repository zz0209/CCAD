import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def identity(path):
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def calculate(margins, clean, source_index, indices, request_ids):
    selected = margins[:, :, request_ids][:, :, :, indices]
    baseline = clean[:, indices]
    source = selected[:, source_index]
    error = selected-source[:, None]
    source_effect = source-baseline[:, None]
    source_ms = np.mean(source_effect**2)
    assert source_ms > 0
    mse = np.mean(error**2, axis=(0, 2, 3))
    return dict(nrmse=np.sqrt(mse/source_ms), rmse=np.sqrt(mse),
                source_effect_rms=np.repeat(np.sqrt(source_ms), selected.shape[1]),
                actual_effect_rms=np.sqrt(np.mean((selected-baseline[:, None, None])**2, axis=(0, 2, 3))),
                accuracy=np.mean(selected > 0, axis=(0, 2, 3)),
                source_decision_agreement=np.mean((selected > 0) == (source[:, None] > 0), axis=(0, 2, 3)))


def analyze(paths, panel_path, output, replicates):
    panel = json.loads(panel_path.read_text(encoding='utf-8'))
    rows = panel['rows']
    with np.load(paths[0], allow_pickle=False) as saved:
        methods = saved['method_names'].astype(str).tolist()
        requests = saved['request_names'].astype(str).tolist()
    source_index = methods.index('source')
    values, clean = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            assert saved['method_names'].astype(str).tolist() == methods
            assert saved['request_names'].astype(str).tolist() == requests
            values.append(saved['margins'].astype(np.float64))
            clean.append(saved['clean'].astype(np.float64))
    values, clean = np.stack(values), np.stack(clean)
    assert values.shape == (len(paths), len(methods), len(requests), len(rows))
    assert clean.shape == (len(paths), len(rows))
    assert np.isfinite(values).all() and np.isfinite(clean).all()
    request_metadata = panel['requests']
    if isinstance(request_metadata, list):
        request_metadata = {item['name']: item for item in request_metadata}
    families = {request_metadata[name]['family'] for name in requests}
    tasks = sorted({row['task'] for row in rows})
    strata = [np.array([i for i, row in enumerate(rows) if row['task'] == task]) for task in tasks]
    assert len({len(ids) for ids in strata}) == 1
    # 语法、固定目标与请求在各请求族内等权，所有方法复用同一句对抽样。
    rng = np.random.default_rng(2026092372)
    bootstrap_indices = [np.concatenate([rng.choice(ids, len(ids), replace=True) for ids in strata])
                         for _ in range(replicates)]
    family_results = {}
    comparisons = [('code', name) for name in ('physical', 'readout_initial', 'program', 'readout_program')
                   if name in methods]
    for family in sorted(families):
        ids = [i for i, name in enumerate(requests) if request_metadata[name]['family'] == family]
        estimate = calculate(values, clean, source_index, np.arange(len(rows)), ids)
        bootstrap = [calculate(values, clean, source_index, sample, ids) for sample in bootstrap_indices]
        report = dict(requests=[requests[i] for i in ids], methods={}, comparisons={})
        for m, name in enumerate(methods):
            report['methods'][name] = {key: float(value[m]) for key, value in estimate.items()}
        for first, second in comparisons:
            a, b = methods.index(first), methods.index(second)
            report['comparisons'][first+' minus '+second] = {
                key: dict(estimate=float(value[a]-value[b]), ci95=np.quantile(
                    [sample[key][a]-sample[key][b] for sample in bootstrap], [.025, .975]).tolist())
                for key, value in estimate.items() if key != 'source_effect_rms'}
        report['by_target'] = [{methods[m]: {key: float(value[m]) for key, value in calculate(
            values[t:t+1], clean[t:t+1], source_index, np.arange(len(rows)), ids).items()}
            for m in range(len(methods))} for t in range(len(paths))]
        report['by_grammar'] = {task: {methods[m]: {key: float(value[m]) for key, value in calculate(
            values, clean, source_index, row_ids, ids).items()} for m in range(len(methods))}
            for task, row_ids in zip(tasks, strata)}
        family_results[family] = report
    summary = dict(inputs=[identity(path) for path in paths], panel=identity(panel_path),
        analysis=identity(__file__), rows=len(rows), fixed_targets=len(paths), methods=methods,
        bootstrap_replicates=replicates, bootstrap_seed=2026092372,
        statistical_scope='Common sentence-pair resampling within each fixed grammar; fixed targets remain together. Conditional on these trained dictionaries and grammar generators.',
        clean_accuracy=float((clean > 0).mean()),
        source_max_between_target_difference=float(np.max(np.abs(values[:, source_index]-values[0, source_index]))),
        clean_max_between_target_difference=float(np.max(np.abs(clean-clean[0]))),
        families=family_results)
    output.mkdir(parents=True, exist_ok=True)
    result = output/'RESULTS.json'
    assert not result.exists(), result
    result.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['# 冻结请求的完整模型比较', '',
             '区间按语法内共同句对抽样，目标字典保持共同参与。nRMSE以同请求族真实源作用RMS归一。', '',
             '| 请求族 | 方法 | nRMSE | 原量纲RMSE | 语法正确率 | 源判断保持 |',
             '|---|---|---:|---:|---:|---:|']
    for family, report in family_results.items():
        for name, metric in report['methods'].items():
            lines.append(f"| {family} | {name} | {metric['nrmse']:.6f} | {metric['rmse']:.6f} | {metric['accuracy']:.6f} | {metric['source_decision_agreement']:.6f} |")
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(result), nrmse={family: {name: values['nrmse']
        for name, values in report['methods'].items()} for family, report in family_results.items()}),
        allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--responses', type=Path, nargs='+', required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    analyze(args.responses, args.panel, args.output, args.bootstrap)


if __name__ == '__main__':
    main()
