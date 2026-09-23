from pathlib import Path
import argparse
import hashlib
import json
import os
import time

os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
import numpy as np


LABELS = [' himself', ' herself', ' themselves']
METHODS = ['region6', 'region2', 'region4', 'union6', 'union_minus_region6', 'cached64_matched', 'raw']
METRICS = ['controller', 'distractor', 'selectivity', 'donor_correct', 'distractor_correct',
           'clean_correct', 'clean_three_candidate_correct', 'donor_three_candidate_correct',
           'distractor_three_candidate_correct', 'controller_flip_to_donor',
           'controller_flip_away_donor']
COMPARISONS = [('region6', 'cached64_matched'), ('union6', 'union_minus_region6')]


def summarize(rows, baseline, values):
    contrast = np.array([[LABELS.index(r['label0']), LABELS.index(r['label1'])] for r in rows])
    ii = np.arange(len(rows))
    base_margin = baseline[ii, contrast[:, 1]] - baseline[ii, contrast[:, 0]]
    edited_margin = values[ii, :, contrast[:, 1]] - values[ii, :, contrast[:, 0]]
    change = edited_margin - base_margin[:, None]
    controller = np.array([r['controller'] for r in rows])
    state = np.array([r['states'][r['controller']] for r in rows])
    desired = (1-2*state) * change[ii, controller]
    collateral = np.abs(change[ii, 1-controller])
    donor_wins = (1-2*state) * edited_margin[ii, controller] > 0
    clean_correct = (2*state-1)*base_margin > 0
    retained = (2*state-1)*edited_margin[ii, 1-controller] > 0
    clean_labels = contrast[ii, state]
    donor_labels = contrast[ii, 1-state]
    clean_oriented_margin = (2*state-1)*base_margin
    donor_oriented_margin = (1-2*state)*edited_margin[ii, controller]
    return dict(controller=desired, distractor=collateral, selectivity=desired-collateral,
                donor_correct=donor_wins, clean_correct=clean_correct, distractor_correct=retained,
                clean_three_candidate_correct=baseline.argmax(1) == clean_labels,
                donor_three_candidate_correct=values[ii, controller].argmax(1) == donor_labels,
                distractor_three_candidate_correct=values[ii, 1-controller].argmax(1) == clean_labels,
                controller_flip_to_donor=(clean_oriented_margin > 0) & (donor_oriented_margin > 0),
                controller_flip_away_donor=(clean_oriented_margin < 0) & (donor_oriented_margin < 0))


def identity(path):
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load_runs(runs, panel_path):
    panel = json.loads(panel_path.read_text(encoding='utf-8'))
    canonical = {row['id']: row for row in panel['rows']}
    cells, inputs, row_ids = {}, [identity(panel_path)], None
    for run in runs:
        status = json.loads((run/'status.json').read_text(encoding='utf-8'))
        assert status['status'] == 'PASS', str(run)
        index_path = run/'response_index.json'
        index = json.loads(index_path.read_text(encoding='utf-8'))
        inputs.append(identity(index_path))
        assert index['identity']['panel_sha256'] == inputs[0]['sha256']
        with np.load(index['baseline']['path']) as saved:
            baseline_rows = saved['row_indices'].astype(int).tolist()
            baseline = saved['logprobs'].astype(np.float64)
        lookup = {value: pos for pos, value in enumerate(baseline_rows)}
        for block in index['blocks']:
            key = (block['objective'], int(block['target_seed']), block['method'])
            assert key not in cells, key
            path = Path(block['path'])
            with np.load(path) as saved:
                ids = saved['row_indices'].astype(int).tolist()
                donor_ids = saved['donor_indices'].astype(int)
                values = saved['logprobs'].astype(np.float64)
                assert values.shape == (len(ids), 2, 3)
                np.testing.assert_array_equal(donor_ids, [[canonical[i]['donor_first'],
                                                         canonical[i]['donor_second']] for i in ids])
                member_ids = saved['member_ids'].astype(int).tolist()
                delta_norm = saved['delta_norm'].astype(np.float64)
            if row_ids is None:
                row_ids = ids
            assert ids == row_ids
            base = baseline[[lookup[i] for i in ids]]
            assert np.isfinite(base).all() and np.isfinite(values).all()
            cells[key] = dict(values=values, baseline=base, member_ids=member_ids,
                              delta_norm=delta_norm, path=str(path), sha256=block['sha256'])
    rows = [canonical[i] for i in row_ids]
    assert len({row['split'] for row in rows}) == 1
    objectives = sorted({key[0] for key in cells})
    targets = sorted({key[1] for key in cells})
    assert len(cells) == len(objectives)*len(targets)*len(METHODS)
    values = np.empty((len(objectives), len(targets), len(METHODS), len(rows), len(METRICS)))
    raw = np.empty((*values.shape[:-1], 2, 3))
    baselines = np.empty((len(objectives), len(targets), len(rows), 3))
    metadata = []
    for oi, objective in enumerate(objectives):
        for ti, target in enumerate(targets):
            for mi, method in enumerate(METHODS):
                item = cells[objective, target, method]
                summary = summarize(rows, item['baseline'], item['values'])
                values[oi, ti, mi] = np.stack([summary[name] for name in METRICS], axis=-1)
                raw[oi, ti, mi] = item['values']
                if mi == 0:
                    baselines[oi, ti] = item['baseline']
                else:
                    np.testing.assert_array_equal(item['baseline'], baselines[oi, ti])
                metadata.append(dict(objective=objective, target_seed=target, method=method,
                    member_ids=item['member_ids'], member_count=len(item['member_ids']),
                    delta_norm_mean=item['delta_norm'].mean(0).tolist(), path=item['path'], sha256=item['sha256']))
    return rows, objectives, targets, values, raw, baselines, metadata, inputs


def noun_bootstrap(rows, draws, seed):
    nouns = sorted({family for row in rows for family in row['noun_families']})
    lookup = {value: i for i, value in enumerate(nouns)}
    ends = np.asarray([[lookup[value] for value in row['noun_families']] for row in rows])
    assert np.all(ends[:, 0] != ends[:, 1])
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(nouns), np.full(len(nouns), 1/len(nouns)), size=draws)
    # 相同名词族在两位置共享出现次数，语法四状态与所有目标共享权重。
    weights = counts[:, ends[:, 0]]*counts[:, ends[:, 1]]
    return nouns, counts, weights.astype(np.float64)


def metric_report(estimate, samples):
    interval = np.quantile(samples, [.025, .975], axis=0) if len(samples) else None
    return {name: dict(estimate=float(estimate[i]),
                      interval95=None if interval is None else interval[:, i].tolist())
            for i, name in enumerate(METRICS)}


def comparison_report(estimate, samples):
    return {first+' minus '+second: metric_report(estimate[METHODS.index(first)]-
                estimate[METHODS.index(second)], samples[:, METHODS.index(first)]-
                samples[:, METHODS.index(second)]) for first, second in COMPARISONS}


def analyze(runs, panel_path, output, draws=1000, seed=20260924):
    started, cpu_started = time.perf_counter(), time.process_time()
    assert not output.exists(), output
    rows, objectives, targets, values, raw, baselines, metadata, inputs = load_runs(runs, panel_path)
    nouns, counts, weights = noun_bootstrap(rows, draws, seed)
    factors, syntaxes = sorted({r['factor'] for r in rows}), sorted({r['syntax'] for r in rows})
    selections = {factor+'__'+syntax: np.asarray([i for i, row in enumerate(rows)
                  if row['factor'] == factor and row['syntax'] == syntax])
                  for factor in factors for syntax in syntaxes}
    assert all(len(ids) for ids in selections.values())
    estimates, samples, valid = {}, {}, {}
    for name, ids in selections.items():
        estimate = values[..., ids, :].mean(-2)
        totals = weights[:, ids].sum(1)
        valid[name] = totals > 0
        resampled = np.full((draws, *estimate.shape), np.nan)
        resampled[valid[name]] = np.einsum('bn,otmnk->botmk', weights[valid[name]][:, ids],
            values[..., ids, :], optimize=True)/totals[valid[name], None, None, None, None]
        estimates[name], samples[name] = estimate, resampled
    common_valid = np.logical_and.reduce(list(valid.values()))
    estimates['equal_factor_syntax'] = np.mean(list(estimates.values()), axis=0)
    samples['equal_factor_syntax'] = np.mean(list(samples.values()), axis=0)
    valid['equal_factor_syntax'] = common_valid
    result = dict(inputs=inputs, analysis=identity(__file__), rows=len(rows), split=rows[0]['split'],
        objectives=objectives, fixed_targets=targets, methods=METHODS, labels=LABELS,
        metrics=dict(controller='Signed donor-directed change in the predefined two-label log odds at the controller.',
            distractor='Absolute change in the same two-label log odds after changing the other noun.',
            selectivity='Controller-directed change minus absolute distractor change.',
            clean_correct='Two-label competence on every clean input, with exact log-odds ties counted incorrect.',
            controller_flip_to_donor='Fraction of all inputs with strictly correct clean two-label ordering and strictly donor-correct edited ordering.',
            controller_flip_away_donor='Fraction of all inputs with strictly donor-correct clean ordering and strictly original-correct edited ordering. Exact ties at either endpoint count as neither flip.',
            three_candidate_correct='Argmax among the three saved reflexive candidates; this is not full-vocabulary correctness.',
            logprobs='Saved candidate log probabilities use the original full-vocabulary softmax denominator.'),
        bootstrap=dict(method='Common noun-family multinomial multiplicities; dyadic row weight m[first]*m[second].',
            requested_replicates=draws, seed=seed, noun_families=nouns, noun_family_count=len(nouns),
            valid_replicates={key: int(mask.sum()) for key, mask in valid.items()},
            invalid_replicates={key: int((~mask).sum()) for key, mask in valid.items()},
            scope='Pigeonhole resampling intervals describe sensitivity to this small observed noun-family set. Grammar templates, corpus, model, source bank, SAE objectives and target seeds are fixed. Targets are not independent sampling units.',
            limitation='Four development or eight confirmation noun families limit interval precision; per-target estimates and ranges are retained. Zero-weight observed-edge draws are undefined and excluded without replacement.'),
        baseline_maximum_between_cells_difference=float(np.abs(baselines-baselines[0, 0]).max()),
        member_and_input_metadata=metadata, results={})
    for oi, objective in enumerate(objectives):
        reports = {}
        for name, estimate in estimates.items():
            boot = samples[name][valid[name], oi]
            target_points = estimate[oi]
            pooled_point, pooled_boot = target_points.mean(0), boot.mean(1)
            reports[name] = dict(
                pooled_methods={method: metric_report(pooled_point[mi], pooled_boot[:, mi])
                                for mi, method in enumerate(METHODS)},
                pooled_comparisons=comparison_report(pooled_point, pooled_boot),
                target_ranges={method: {metric: [float(target_points[:, mi, ki].min()),
                    float(target_points[:, mi, ki].max())] for ki, metric in enumerate(METRICS)}
                    for mi, method in enumerate(METHODS)},
                by_target={str(target): dict(methods={method: metric_report(target_points[ti, mi], boot[:, ti, mi])
                    for mi, method in enumerate(METHODS)}, comparisons=comparison_report(target_points[ti], boot[:, ti]))
                    for ti, target in enumerate(targets)})
        result['results'][objective] = reports
    result['analysis_cost'] = dict(wall_seconds=time.perf_counter()-started,
                                   process_cpu_seconds=time.process_time()-cpu_started)
    output.mkdir(parents=True)
    np.savez_compressed(output/'ROW_METRICS.npz', values=values, metric_names=METRICS,
        method_names=METHODS, objectives=objectives, target_seeds=targets, row_ids=[r['id'] for r in rows],
        logprobs=raw, baseline_logprobs=baselines, noun_bootstrap_counts=counts,
        valid_all_cells=common_valid, noun_family_ids=nouns)
    (output/'RESULTS.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['# 冻结共同组成的内容传递', '',
        '保留全部clean输入。区间使用共同名词族的重复次数，两位置共享权重；两个机制分别报告，目标seed保持固定。',
        f'共同有效重采样 {int(common_valid.sum())}/{draws}，名词族 {len(nouns)} 个。有限词汇下区间精度有限，逐目标结果完整保存。', '',
        '| SAE机制 | Factor与句法 | 方法 | Controller定向变化 | Distractor绝对变化 | 两者差值 | Clean两候选 | Clean三候选 |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for objective, reports in result['results'].items():
        for name, report in reports.items():
            for method, summary in report['pooled_methods'].items():
                fields = [summary[key]['estimate'] for key in ('controller', 'distractor', 'selectivity',
                                                               'clean_correct', 'clean_three_candidate_correct')]
                lines.append('| '+objective+' | '+name+' | '+method+' | '+' | '.join(f'{v:.6f}' for v in fields)+' |')
    lines += ['', 'Clean两候选对应预定factor的答案区别；Clean三候选对应保存的三个反身代词。完整词表正确率未作为这些字段的含义。',
              '共同region减cached、union减去除共同region的逐目标与共同配对区间见RESULTS.json。']
    lines += ['', '| SAE机制 | 方法 | 转向donor的比例 | 远离donor的比例 |',
              '|---|---|---:|---:|']
    for objective, reports in result['results'].items():
        for method, summary in reports['equal_factor_syntax']['pooled_methods'].items():
            toward = summary['controller_flip_to_donor']['estimate']
            away = summary['controller_flip_away_donor']['estimate']
            lines.append(f'| {objective} | {method} | {toward:.6f} | {away:.6f} |')
    lines += ['', '翻转比例的分母包括全部输入；两个端点均须具有严格的两候选顺序，任一端点并列均不计翻转。']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(output), rows=len(rows), valid_bootstrap=int(common_valid.sum()),
        equal_factor_syntax={obj: result['results'][obj]['equal_factor_syntax']['pooled_methods']['region6']
                             for obj in objectives})), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, nargs='+', required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20260924)
    args = parser.parse_args()
    analyze(args.run, args.panel, args.output, args.bootstrap, args.seed)


if __name__ == '__main__':
    main()
