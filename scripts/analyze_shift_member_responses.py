import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def identity(path):
    path = Path(path)
    return dict(path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def matrix_rows(documents):
    return (np.asarray(documents)[:, None]*2+np.arange(2)[None]).reshape(-1)


def fixed_documents(rows, available, number):
    assert number % 4 == 0
    result = []
    for label in (0, 1):
        for gender in (0, 1):
            group = sorted([int(i) for i in available if rows[i]['label'] == label and rows[i]['gender'] == gender],
                           key=lambda i: rows[i]['document_sha256'])
            assert len(group) >= number//4
            result.extend(group[:number//4])
    return np.array(result, dtype=int)


def source_basis(values, training_rows, rank):
    mean = values[training_rows].mean(0)
    _, singular, right = np.linalg.svd(values[training_rows]-mean, full_matrices=False)
    assert rank <= len(singular)
    projection = right[:rank].T
    scores = (values-mean)@projection
    scale = np.sqrt(np.mean(scores[training_rows]**2, axis=0))
    scale[scale == 0] = 1.
    scores /= scale
    return np.column_stack([np.ones(len(values)), scores]), dict(mean=mean, projection=projection,
        score_scale=scale, singular_values=singular)


def choose_rank(source, rows, fit, ranks, budgets):
    folds = [[] for _ in range(4)]
    for label in (0, 1):
        for gender in (0, 1):
            group = sorted([int(i) for i in fit if rows[i]['label'] == label and rows[i]['gender'] == gender],
                           key=lambda i: rows[i]['document_sha256'])
            for offset, i in enumerate(group):
                folds[offset % 4].append(i)
    records = []
    for fold, validation in enumerate(folds):
        training = np.array([i for i in fit if i not in validation], dtype=int)
        train_rows, valid_rows = matrix_rows(training), matrix_rows(validation)
        for held_member in range(source.shape[1]):
            observed = np.delete(source, held_member, axis=1)
            for rank in ranks:
                design, _ = source_basis(observed, train_rows, rank)
                for budget in budgets:
                    calibration = fixed_documents(rows, training, budget)
                    cal_rows = matrix_rows(calibration)
                    coefficients = np.linalg.lstsq(design[cal_rows], source[cal_rows, held_member], rcond=None)[0]
                    predicted = design[valid_rows]@coefficients
                    truth = source[valid_rows, held_member]
                    records.append(dict(fold=fold, held_source_member=held_member, rank=rank, budget=budget,
                        squared_error=float(np.sum((predicted-truth)**2)), count=len(truth),
                        calibration_document_indices=calibration.tolist(), validation_document_indices=validation))
    errors = {rank: sum(r['squared_error'] for r in records if r['rank'] == rank)/
              sum(r['count'] for r in records if r['rank'] == rank) for rank in ranks}
    chosen = min(ranks, key=lambda rank: (errors[rank], rank))
    return dict(rank=chosen, source_only_cv_mse=errors, folds=records,
        rule='Four stratified document folds crossed with leave-one-source-member-out; source-only calibration of 8/16 documents; minimum held response MSE, smaller rank for exact ties',
        target_responses_used=False)


def constrained_selection(predicted, desired, sites, member_ids, quotas):
    q = predicted.T@predicted/len(predicted)
    rhs = predicted.T@desired/len(predicted)
    binary = np.zeros(len(member_ids), dtype=np.float64)
    groups = {site: np.flatnonzero(sites == site) for site in quotas}
    for site, ids in groups.items():
        order = sorted(ids, key=lambda i: (q[i, i]-2*rhs[i], int(member_ids[i])))
        binary[order[:quotas[site]]] = 1.
    objective = float(binary@q@binary-2*rhs@binary)
    initial = objective
    history = []
    while True:
        gradient = q@binary-rhs
        best = None
        for site, ids in groups.items():
            outgoing = sorted(ids[binary[ids] == 1], key=lambda i: int(member_ids[i]))
            incoming = sorted(ids[binary[ids] == 0], key=lambda i: int(member_ids[i]))
            if not incoming:
                continue
            outgoing, incoming = np.array(outgoing), np.array(incoming)
            change = 2*(gradient[incoming][None]-gradient[outgoing][:, None])+q.diagonal()[outgoing][:, None]+q.diagonal()[incoming][None]-2*q[np.ix_(outgoing, incoming)]
            oi, ji = np.unravel_index(np.argmin(change), change.shape)
            candidate = (float(change[oi, ji]), site, int(member_ids[outgoing[oi]]), int(member_ids[incoming[ji]]), int(outgoing[oi]), int(incoming[ji]))
            if best is None or candidate[:4] < best[:4]:
                best = candidate
        precision = 64*np.finfo(float).eps*max(1., abs(objective), float(np.max(np.abs(q))), float(np.max(np.abs(rhs))))
        if best is None or best[0] >= -precision:
            break
        change, site, removed, added, i, j = best
        binary[i], binary[j] = 0., 1.
        updated = float(binary@q@binary-2*rhs@binary)
        assert updated < objective and abs(updated-objective-change) <= precision*len(binary)
        history.append(dict(site=site, removed=removed, added=added, before=objective, after=updated))
        objective = updated
    selected = {site: member_ids[ids[binary[ids] == 1]].astype(int).tolist() for site, ids in groups.items()}
    assert all(len(selected[site]) == quotas[site] for site in quotas)
    return binary, selected, dict(initial_objective=initial, final_objective=objective,
        exchanges=history, best_remaining_change=None if best is None else best[0],
        stopping='No improving same-site single exchange at floating-point precision', global_optimum_claim=False)


def read_bank(paths, reference):
    arrays = {key: [] for key in ('effects', 'code_sum', 'member_ids')}
    sites = []
    for path in paths:
        with np.load(path) as block:
            assert str(block['status']) == 'PASS'
            for key in ('row_ids', 'document_sha256', 'splits'):
                np.testing.assert_array_equal(block[key], reference[key])
            site = str(block['site'].item())
            n = len(block['member_ids'])
            sites.extend([site]*n)
            for key in arrays:
                arrays[key].append(block[key].copy())
    result = {key: np.concatenate(parts, axis=-1) for key, parts in arrays.items()}
    result['sites'] = np.array(sites)
    assert len(set(zip(sites, result['member_ids'].tolist()))) == len(sites)
    assert result['effects'].shape == result['code_sum'].shape
    assert result['effects'].shape[:2] == reference['baseline_logits'].shape
    assert np.isfinite(result['effects']).all() and np.isfinite(result['code_sum']).all()
    return result


def response_metrics(actual, predicted, documents, bootstrap, rng):
    truth = actual[documents]
    estimate = predicted[documents]
    error_doc = np.mean((estimate-truth)**2, axis=(1, 2))
    truth_doc = np.mean(truth**2, axis=(1, 2))
    prediction_doc = np.mean(estimate**2, axis=(1, 2))
    def values(indices):
        rmse = np.sqrt(error_doc[indices].mean())
        scale = np.sqrt(truth_doc[indices].mean())
        assert scale > 0
        return [rmse, rmse/scale, np.sqrt(prediction_doc[indices].mean()), scale]
    point = values(np.arange(len(documents)))
    samples = np.array([values(rng.choice(len(documents), len(documents), replace=True)) for _ in range(bootstrap)])
    keys = ('rmse', 'nrmse', 'predicted_effect_rms', 'actual_effect_rms')
    return {key: dict(value=float(point[i]), ci95=np.quantile(samples[:, i], [.025, .975]).tolist()) for i, key in enumerate(keys)}


def source_structure(index, output):
    with np.load(index['source_reference']) as raw:
        reference = {key: raw[key].copy() for key in raw.files}
    source = read_bank(index['source_blocks'], reference)
    additive = source['effects'].sum(-1)
    whole = reference['whole_W_effects']
    result = {}
    for split in ('fit', 'evaluation'):
        documents = np.flatnonzero(reference['splits'] == split)
        scale = float(np.sqrt(np.mean(whole[documents]**2)))
        result[split] = {}
        for condition, ids in [('joint', [0, 1]), ('clean', [0]), ('conditional', [1])]:
            a, b = additive[documents][:, ids], whole[documents][:, ids]
            rmse = float(np.sqrt(np.mean((a-b)**2)))
            result[split][condition] = dict(rmse=rmse, nrmse_common_whole_W_scale=rmse/scale,
                singleton_sum_rms=float(np.sqrt(np.mean(a**2))), actual_whole_W_rms=float(np.sqrt(np.mean(b**2))),
                mean_singleton_sum=float(a.mean()), mean_actual_whole_W=float(b.mean()))
    write_json(output, dict(source_singleton_additivity=result,
        definition='Sum of 11 independently deleted source members compared against actual simultaneous multilevel source W deletion',
        source_reference=identity(index['source_reference'])))
    return result


def group_statistics(effect, singleton_sum, predicted_sum, reference, baseline, labels, genders, documents):
    result = {}
    scale = float(np.sqrt(np.mean(reference[documents]**2)))
    labels, genders = labels[documents], genders[documents]
    groups = [(labels == label) & (genders == gender) for label in (0, 1) for gender in (0, 1)]
    for name, conditions in [('joint', [0, 1]), ('clean', [0]), ('conditional', [1])]:
        actual, truth = effect[documents][:, conditions], reference[documents][:, conditions]
        additive = None if singleton_sum is None else singleton_sum[documents][:, conditions]
        predicted = None if predicted_sum is None else predicted_sum[documents][:, conditions]
        current_baseline = baseline[documents][:, conditions]
        values = dict(rmse=float(np.sqrt(np.mean((actual-truth)**2))),
            actual_group_rms=float(np.sqrt(np.mean(actual**2))), source_whole_W_rms=float(np.sqrt(np.mean(truth**2))))
        if additive is not None:
            values.update(singleton_sum_rms=float(np.sqrt(np.mean(additive**2))),
                singleton_sum_to_group_rmse=float(np.sqrt(np.mean((additive-actual)**2))),
                singleton_sum_to_source_rmse=float(np.sqrt(np.mean((additive-truth)**2))))
        values['nrmse_common_source_scale'] = values['rmse']/scale
        for key, logits in [('profession', current_baseline+actual), ('source_profession', current_baseline+truth),
                            ('baseline_profession', current_baseline)]:
            correct = (logits > 0) == labels[:, None]
            values[key+'_accuracy'] = float(correct.mean())
            values[key+'_worst_group'] = min(float(correct[group].mean()) for group in groups)
        if predicted is not None:
            values.update(predicted_sum_rms=float(np.sqrt(np.mean(predicted**2))),
                prediction_to_group_rmse=float(np.sqrt(np.mean((predicted-actual)**2))),
                prediction_to_singleton_sum_rmse=float(np.sqrt(np.mean((predicted-additive)**2))),
                prediction_to_source_rmse=float(np.sqrt(np.mean((predicted-truth)**2))))
        result.update({name+'/'+key: value for key, value in values.items()})
    return result


def analyze_groups(run, group_run, selection_directory, output, bootstrap):
    if output.exists():
        raise FileExistsError(output)
    assert read_json(group_run/'status.json')['status'] == 'PASS'
    index = read_json(run/'response_index.json')
    group_index = read_json(group_run/'response_index.json')
    selected = read_json(selection_directory/'selected_members.json')
    with np.load(index['source_reference']) as raw:
        reference = {key: raw[key].copy() for key in raw.files}
    target = read_bank(index['target_blocks'], reference)
    predictions = np.load(selection_directory/'response_predictions.npz')
    np.testing.assert_array_equal(predictions['document_sha256'], reference['document_sha256'])
    evaluation = np.flatnonzero(reference['splits'] == 'evaluation')
    labels, genders = reference['labels'], reference['genders']
    strata = [evaluation[(labels[evaluation] == label) & (genders[evaluation] == gender)] for label in (0, 1) for gender in (0, 1)]
    assert all(len(group) for group in strata)
    rng = np.random.default_rng(9232321)
    samples = [np.concatenate([rng.choice(group, len(group), replace=True) for group in strata]) for _ in range(bootstrap)]
    stats, sampled, arrays, rows = {}, {}, {}, []
    for path in group_index['group_blocks']:
        with np.load(path) as block:
            assert str(block['status']) == 'PASS'
            np.testing.assert_array_equal(block['document_sha256'], reference['document_sha256'])
            np.testing.assert_array_equal(block['row_ids'], reference['row_ids'])
            method = str(block['method'])
            assert method not in stats
            actual_selection = json.loads(str(block['selected_members']))
            assert {site: sorted(ids) for site, ids in actual_selection.items()} == {site: sorted(ids) for site, ids in selected[method].items()}
            active = np.array([int(member) in selected[method][site] for site, member in zip(target['sites'], target['member_ids'])])
            assert active.sum() == sum(len(ids) for ids in selected[method].values()) == 22
            additive = target['effects'][:, :, active].sum(-1)
            predicted = predictions[method][:, :, active].sum(-1) if method in predictions.files else None
            np.testing.assert_array_equal(block['effects'], block['logits']-block['baseline_logits'])
            effect, baseline, truth = [block[key].astype(np.float64) for key in ('effects', 'baseline_logits', 'whole_W_effects')]
            arguments = (effect, additive, predicted, truth, baseline, labels, genders)
            points = group_statistics(*arguments, evaluation)
            sampled[method] = {key: [] for key in points}
            for documents in samples:
                for key, value in group_statistics(*arguments, documents).items():
                    sampled[method][key].append(value)
            stats[method] = {key: dict(value=value, ci95=np.quantile(sampled[method][key], [.025, .975]).tolist()) for key, value in points.items()}
            arrays[method+'__actual_group'] = effect
            arrays[method+'__actual_singleton_sum'] = additive
            if predicted is not None:
                arrays[method+'__predicted_singleton_sum'] = predicted
            for i in evaluation:
                for condition in (0, 1):
                    rows.append(dict(document_sha256=str(reference['document_sha256'][i]), row_id=int(reference['row_ids'][i]),
                        method=method, condition=condition, label=int(labels[i]), gender=int(genders[i]),
                        actual_group=float(effect[i, condition]), actual_singleton_sum=float(additive[i, condition]),
                        predicted_singleton_sum=None if predicted is None else float(predicted[i, condition]),
                        source_effect=float(truth[i, condition]), baseline_logit=float(baseline[i, condition])))
    assert set(stats) == set(selected)
    contrasts = {}
    for first in stats:
        if not first.startswith('finite_response_'):
            continue
        suffix = first.removeprefix('finite_response_')
        for second in [name for name in stats if name.endswith('_'+suffix) and name != first]:
            contrasts[first+'_minus_'+second] = {}
            for key in ('joint/nrmse_common_source_scale', 'clean/rmse', 'conditional/rmse',
                        'clean/profession_accuracy', 'conditional/profession_accuracy'):
                differences = np.asarray(sampled[first][key])-np.asarray(sampled[second][key])
                contrasts[first+'_minus_'+second][key] = dict(value=stats[first][key]['value']-stats[second][key]['value'],
                    ci95=np.quantile(differences, [.025, .975]).tolist())
    output.mkdir(parents=True)
    write_json(output/'GROUP_RESULTS.json', dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        run=run.as_posix(), group_run=group_run.as_posix(), statistics=stats, paired_contrasts=contrasts,
        document_values=rows, identities=dict(selection=identity(selection_directory/'selected_members.json'),
            response_index=identity(run/'response_index.json'), group_index=identity(group_run/'response_index.json'), analyzer=identity(Path(__file__))),
        bootstrap=dict(draws=bootstrap, seed=9232321, unit='Evaluation document stratified by label and gender; all methods, conditions and members paired'),
        direct_calibration_prediction='Direct calibration selects on observed calibration responses; no held-document singleton predictor is defined',
        definitions=dict(actual_group='Actual simultaneous multilevel complete-member removal',
            actual_singleton_sum='Sum of independently measured selected-member deletion effects',
            predicted_singleton_sum='Sum of frozen predicted selected-member effects; used for selection only')))
    np.savez_compressed(output/'group_comparison_arrays.npz', **arrays, row_ids=reference['row_ids'],
        document_sha256=reference['document_sha256'], splits=reference['splits'])
    lines = ['# 完整成员组合的实际使用', '',
        '| 方法 | 两条件nRMSE | clean RMSE | conditional RMSE | clean职业准确率 | conditional职业准确率 |',
        '|---|---:|---:|---:|---:|---:|']
    keys = ('joint/nrmse_common_source_scale', 'clean/rmse', 'conditional/rmse', 'clean/profession_accuracy', 'conditional/profession_accuracy')
    for method, values in stats.items():
        lines.append('| '+method+' | '+' | '.join(f"{values[key]['value']:.6f}" for key in keys)+' |')
    lines += ['', '| 方法 | 单成员实际作用之和到真实组合RMSE | 预测作用之和到真实组合RMSE | 预测作用之和到单成员实际作用之和RMSE |', '|---|---:|---:|---:|']
    keys = ('joint/singleton_sum_to_group_rmse', 'joint/prediction_to_group_rmse', 'joint/prediction_to_singleton_sum_rmse')
    for method, values in stats.items():
        lines.append('| '+method+' | '+' | '.join(f"{values[key]['value']:.6f}" if key in values else '不适用' for key in keys)+' |')
    lines += ['', '全部选择在真实组合执行之前保存。组成用途由真实组合结果评价，单成员的实际作用与预测作用之和分别说明响应预测误差和共同执行产生的变化。', '',
        'JSON保存按label与gender分层的文档配对bootstrap区间、方法差、实际作用幅度、最差组职业准确率及逐文档结果。', '']
    (output/'GROUP_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(output=output.as_posix(), groups=len(stats), documents=len(evaluation))), flush=True)


def aggregate_groups(directories, output, bootstrap):
    if output.exists():
        raise FileExistsError(output)
    reports = [read_json(path/'GROUP_RESULTS.json') for path in directories]
    methods = list(reports[0]['statistics'])
    documents = sorted({row['document_sha256'] for row in reports[0]['document_values']})
    document_map = {value: i for i, value in enumerate(documents)}
    n = len(documents)
    labels, genders = np.full(n, -1), np.full(n, -1)
    observations = []
    for report in reports:
        assert set(report['statistics']) == set(methods)
        target = {}
        for method in methods:
            rows = [row for row in report['document_values'] if row['method'] == method]
            assert len(rows) == n*2
            has_singletons = all(row['actual_singleton_sum'] is not None for row in rows)
            fields = ['actual_group', 'source_effect', 'baseline_logit']+(['actual_singleton_sum'] if has_singletons else [])
            arrays = {key: np.full((n, 2), np.nan) for key in fields}
            has_prediction = all(row['predicted_singleton_sum'] is not None for row in rows)
            predicted = np.full((n, 2), np.nan) if has_prediction else None
            for row in rows:
                i, c = document_map[row['document_sha256']], row['condition']
                if labels[i] != -1:
                    assert (labels[i], genders[i]) == (row['label'], row['gender'])
                labels[i], genders[i] = row['label'], row['gender']
                for key in arrays:
                    assert np.isnan(arrays[key][i, c])
                    arrays[key][i, c] = row[key]
                if has_prediction:
                    predicted[i, c] = row['predicted_singleton_sum']
            assert all(np.isfinite(value).all() for value in arrays.values())
            target[method] = (arrays['actual_group'], arrays['actual_singleton_sum'] if has_singletons else None, predicted,
                arrays['source_effect'], arrays['baseline_logit'], labels, genders)
        observations.append(target)
    strata = [np.flatnonzero((labels == label) & (genders == gender)) for label in (0, 1) for gender in (0, 1)]
    rng = np.random.default_rng(9232321)
    samples = [np.concatenate([rng.choice(group, len(group), replace=True) for group in strata]) for _ in range(bootstrap)]
    stats, draws = {}, {}
    for method in methods:
        target_points = [group_statistics(*target[method], np.arange(n)) for target in observations]
        points = {key: float(np.mean([value[key] for value in target_points])) for key in target_points[0]}
        draws[method] = {key: [] for key in points}
        for documents_sample in samples:
            target_sample = [group_statistics(*target[method], documents_sample) for target in observations]
            for key in points:
                draws[method][key].append(float(np.mean([value[key] for value in target_sample])))
        stats[method] = {key: dict(value=value, ci95=np.quantile(draws[method][key], [.025, .975]).tolist()) for key, value in points.items()}
    contrasts = {}
    for comparator in ('direct_calibration_n8', 'activation_amplitude_n8'):
        for method in methods:
            if method == comparator:
                continue
            contrasts[method+'_minus_'+comparator] = {}
            for key in ('joint/nrmse_common_source_scale', 'clean/rmse', 'conditional/rmse', 'clean/profession_accuracy', 'conditional/profession_accuracy'):
                values = np.array(draws[method][key])-np.array(draws[comparator][key])
                contrasts[method+'_minus_'+comparator][key] = dict(value=stats[method][key]['value']-stats[comparator][key]['value'],
                    ci95=np.quantile(values, [.025, .975]).tolist())
    output.mkdir(parents=True)
    write_json(output/'COHORT_GROUP_RESULTS.json', dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        statistics=stats, paired_contrasts=contrasts, target_count=len(reports), evaluation_documents=n,
        target_reports=[identity(path/'GROUP_RESULTS.json') for path in directories],
        estimand='Equal mean of target-specific statistics in a fixed dictionary cohort',
        bootstrap=dict(unit='Document resampled within label/gender strata; same sampled documents across targets, methods and conditions',
            draws=bootstrap, seed=9232321, target_resampling=False)))
    lines = ['# 固定目标字典共同使用结果', '',
        f'{len(reports)}个目标等权，{n}篇评价文档共同抽样。区间条件于这些固定字典。', '',
        '| 方法 | 两条件nRMSE | clean实际作用RMS | conditional实际作用RMS | clean职业准确率 | conditional职业准确率 |',
        '|---|---:|---:|---:|---:|---:|']
    keys = ('joint/nrmse_common_source_scale', 'clean/actual_group_rms', 'conditional/actual_group_rms', 'clean/profession_accuracy', 'conditional/profession_accuracy')
    for method, values in stats.items():
        lines.append('| '+method+' | '+' | '.join(f"{values[key]['value']:.6f}" for key in keys)+' |')
    (output/'COHORT_GROUP_RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=output.as_posix(), targets=len(reports), documents=n)), flush=True)


def analyze_frozen_groups(group_run, output, bootstrap):
    if output.exists():
        raise FileExistsError(output)
    assert read_json(group_run/'status.json')['status'] == 'PASS'
    index = read_json(group_run/'response_index.json')
    config = read_json(group_run/'config.resolved.json')
    selected = read_json(config['selected_groups_file'])
    with np.load(index['source_reference']) as raw:
        reference = {key: raw[key].copy() for key in raw.files}
    evaluation = np.flatnonzero(reference['splits'] == 'evaluation')
    labels, genders = reference['labels'], reference['genders']
    strata = [evaluation[(labels[evaluation] == label) & (genders[evaluation] == gender)] for label in (0, 1) for gender in (0, 1)]
    assert all(len(group) for group in strata)
    rng = np.random.default_rng(9232321)
    samples = [np.concatenate([rng.choice(group, len(group), replace=True) for group in strata]) for _ in range(bootstrap)]
    statistics, draws, rows = {}, {}, []
    for path in index['group_blocks']:
        with np.load(path) as block:
            assert str(block['status']) == 'PASS'
            for key in ('document_sha256', 'row_ids', 'splits', 'labels', 'genders'):
                np.testing.assert_array_equal(block[key], reference[key])
            method = str(block['method'])
            assert method not in statistics
            assert json.loads(str(block['selected_members'])) == selected[method]
            np.testing.assert_array_equal(block['effects'], block['logits']-block['baseline_logits'])
            effect, baseline, truth = [block[key].astype(np.float64) for key in ('effects', 'baseline_logits', 'whole_W_effects')]
            arguments = effect, None, None, truth, baseline, labels, genders
            points = group_statistics(*arguments, evaluation)
            draws[method] = {key: [] for key in points}
            for documents in samples:
                for key, value in group_statistics(*arguments, documents).items():
                    draws[method][key].append(value)
            statistics[method] = {key: dict(value=value, ci95=np.quantile(draws[method][key], [.025, .975]).tolist()) for key, value in points.items()}
            for i in evaluation:
                for condition in (0, 1):
                    rows.append(dict(document_sha256=str(reference['document_sha256'][i]), row_id=int(reference['row_ids'][i]),
                        method=method, condition=condition, label=int(labels[i]), gender=int(genders[i]),
                        actual_group=float(effect[i, condition]), actual_singleton_sum=None, predicted_singleton_sum=None,
                        source_effect=float(truth[i, condition]), baseline_logit=float(baseline[i, condition])))
    assert set(statistics) == set(selected)
    contrasts = {}
    for first in statistics:
        for second in statistics:
            if first >= second:
                continue
            contrasts[first+'_minus_'+second] = {}
            for key in points:
                difference = np.array(draws[first][key])-np.array(draws[second][key])
                contrasts[first+'_minus_'+second][key] = dict(value=statistics[first][key]['value']-statistics[second][key]['value'],
                    ci95=np.quantile(difference, [.025, .975]).tolist())
    output.mkdir(parents=True)
    write_json(output/'GROUP_RESULTS.json', dict(written_at_utc=datetime.now(timezone.utc).isoformat(), group_run=group_run.as_posix(),
        statistics=statistics, paired_contrasts=contrasts, document_values=rows,
        identities=dict(group_index=identity(group_run/'response_index.json'), selection=identity(config['selected_groups_file']), analyzer=identity(Path(__file__))),
        bootstrap=dict(draws=bootstrap, seed=9232321, unit='Evaluation document stratified by label and gender; all frozen methods and conditions paired'),
        definitions=dict(actual_group='Actual frozen multilevel complete-member removal on current evaluation documents',
            singleton_values='Not observed for these documents; no historical singleton or prediction arrays used')))
    lines = ['# 冻结成员组成的文档确认', '', '| 方法 | 两条件nRMSE | clean作用RMS | conditional作用RMS | clean职业准确率 | conditional职业准确率 |', '|---|---:|---:|---:|---:|---:|']
    keys = ('joint/nrmse_common_source_scale', 'clean/actual_group_rms', 'conditional/actual_group_rms', 'clean/profession_accuracy', 'conditional/profession_accuracy')
    for method, values in statistics.items():
        lines.append('| '+method+' | '+' | '.join(f"{values[key]['value']:.6f}" for key in keys)+' |')
    lines += ['', '使用保存的固定成员，在当前文档执行正常与source-P背景的完整删除。当前文档的单成员响应未采集，未使用历史单成员数组代替。', '']
    (output/'GROUP_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(output=output.as_posix(), methods=len(statistics), documents=len(evaluation))), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--rank-selection', type=Path)
    parser.add_argument('--budgets', nargs='+', type=int, default=[8, 16])
    parser.add_argument('--ranks', nargs='+', type=int, default=[1, 2, 4, 8])
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--group-run', type=Path)
    parser.add_argument('--selection-directory', type=Path)
    parser.add_argument('--cohort-groups', nargs='+', type=Path)
    parser.add_argument('--frozen-groups', action='store_true')
    args = parser.parse_args()
    if args.cohort_groups:
        aggregate_groups(args.cohort_groups, args.output, args.bootstrap)
        return
    if args.frozen_groups:
        assert args.group_run is not None
        analyze_frozen_groups(args.group_run, args.output, args.bootstrap)
        return
    assert args.run is not None
    if args.group_run:
        assert args.selection_directory is not None
        analyze_groups(args.run, args.group_run, args.selection_directory, args.output, args.bootstrap)
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    index = read_json(args.run/'response_index.json')
    if read_json(args.run/'status.json')['status'] != 'PASS':
        raise ValueError('成员响应采集尚未完成')
    config = read_json(args.run/'config.resolved.json')
    rows = read_json(args.run/'response_membership.json')['rows']
    with np.load(index['source_reference']) as reference_npz:
        reference = {key: reference_npz[key].copy() for key in reference_npz.files}
    np.testing.assert_array_equal(reference['document_sha256'], [r['document_sha256'] for r in rows])
    fit = np.flatnonzero(reference['splits'] == 'fit')
    evaluation = np.flatnonzero(reference['splits'] == 'evaluation')
    assert len(set(reference['document_sha256'])) == len(rows) and len(fit) and len(evaluation)
    source = read_bank(index['source_blocks'], reference)
    target = read_bank(index['target_blocks'], reference)
    assert source['effects'].shape[-1] == 11
    with np.load(Path(config['relation_run'])/'relation.npz') as relation:
        for site in config['members_by_site']:
            np.testing.assert_array_equal(np.sort(target['member_ids'][target['sites'] == site]),
                np.sort(relation[site+'__candidates']))
    n, _, members = target['effects'].shape
    s = source['effects'].reshape(n*2, -1).astype(np.float64)
    y = target['effects'].reshape(n*2, members).astype(np.float64)
    natural = (source['code_sum']/reference['token_counts'][:, None, None]).reshape(n*2, -1).astype(np.float64)
    whole = reference['whole_W_effects'].reshape(-1).astype(np.float64)
    fit_rows = matrix_rows(fit)
    rank_selection = read_json(args.rank_selection) if args.rank_selection else choose_rank(s, rows, fit, args.ranks, args.budgets)
    rank = rank_selection['rank']
    designs, basis_parameters = {}, {}
    for method, matrix in [('finite_response', s), ('natural_activation', natural)]:
        designs[method], values = source_basis(matrix, fit_rows, rank)
        basis_parameters.update({method+'__'+key: value for key, value in values.items()})
    with np.load(index['target_path_projection']) as path_bank:
        np.testing.assert_array_equal(path_bank['document_sha256'], reference['document_sha256'])
        np.testing.assert_array_equal(path_bank['member_ids'], target['member_ids'])
        np.testing.assert_array_equal(path_bank['sites'], target['sites'])
        path_projection = path_bank['path_projection'].copy()
    assert path_projection.shape == target['effects'].shape and np.isfinite(path_projection).all()
    amplitude = target['code_sum']/reference['token_counts'][:, None, None]
    selections, details, predictions, metrics = {}, {}, {}, {}
    for budget in args.budgets:
        calibration = fixed_documents(rows, fit, budget)
        cal_rows = matrix_rows(calibration)
        # 拟合函数只接收规定的目标校准响应。
        observed = y[cal_rows].copy()
        for base, design in designs.items():
            method = base+f'_n{budget}'
            coefficients = np.linalg.lstsq(design[cal_rows], observed, rcond=None)[0]
            predicted = design@coefficients
            binary, selected, diagnostics = constrained_selection(predicted[fit_rows], whole[fit_rows],
                target['sites'], target['member_ids'], config['members_by_site'])
            predictions[method] = predicted.reshape(n, 2, members)
            selections[method] = selected
            details[method] = dict(calibration_document_indices=calibration.tolist(), selection_document_indices=fit.tolist(),
                target_calibration_member_condition_calls=budget*2*members, diagnostics=diagnostics)
            basis_parameters[method+'__coefficients'] = coefficients
            basis_parameters[method+'__binary'] = binary
        for base, proxy in [('path_calibration', path_projection), ('activation_amplitude', amplitude)]:
            method = base+f'_n{budget}'
            predicted = np.zeros_like(target['effects'], dtype=np.float64)
            coefficients = np.empty((2, members, 2))
            for condition in (0, 1):
                for member in range(members):
                    design = np.column_stack([np.ones(n), proxy[:, condition, member]])
                    coefficients[condition, member] = np.linalg.lstsq(design[calibration],
                        observed.reshape(budget, 2, members)[:, condition, member], rcond=None)[0]
                    predicted[:, condition, member] = design@coefficients[condition, member]
            binary, selected, diagnostics = constrained_selection(predicted.reshape(n*2, members)[fit_rows],
                whole[fit_rows], target['sites'], target['member_ids'], config['members_by_site'])
            predictions[method] = predicted
            selections[method] = selected
            details[method] = dict(calibration_document_indices=calibration.tolist(), selection_document_indices=fit.tolist(),
                target_calibration_member_condition_calls=budget*2*members, diagnostics=diagnostics,
                calibration='Independent signed affine fit per member and condition; standard least-squares minimum norm')
            basis_parameters[method+'__coefficients'] = coefficients
            basis_parameters[method+'__binary'] = binary
        method = 'direct_calibration'+f'_n{budget}'
        binary, selected, diagnostics = constrained_selection(observed, whole[cal_rows], target['sites'],
            target['member_ids'], config['members_by_site'])
        selections[method] = selected
        details[method] = dict(calibration_document_indices=calibration.tolist(), selection_document_indices=calibration.tolist(),
            target_calibration_member_condition_calls=budget*2*members, diagnostics=diagnostics)
        basis_parameters[method+'__binary'] = binary
    for method, predicted in predictions.items():
        metrics[method] = {}
        for condition, condition_ids in [('joint', [0, 1]), ('clean', [0]), ('conditional', [1])]:
            metrics[method][condition] = response_metrics(target['effects'][:, condition_ids], predicted[:, condition_ids],
                evaluation, args.bootstrap, np.random.default_rng(9232317))
    args.output.mkdir(parents=True)
    source_structure(index, args.output/'SOURCE_STRUCTURE.json')
    write_json(args.output/'selected_members.json', selections)
    write_json(args.output/'rank_selection.json', rank_selection)
    write_json(args.output/'selection_meta.json', dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        run=args.run.as_posix(), rank=rank, methods=details, candidate_ids=target['member_ids'].tolist(),
        candidate_sites=target['sites'].tolist(), source_member_ids=source['member_ids'].tolist(),
        source_member_sites=source['sites'].tolist(), members_by_site=config['members_by_site'],
        fitting='Two conditions share document splits; intercept retained; source basis fitted on fit documents only',
        selection='Sum of predicted singleton effects used only to select; actual group response requires separate model execution',
        rank_selection_source=identity(args.rank_selection) if args.rank_selection else 'source-only cross-validation',
        source_index=identity(args.run/'response_index.json'), analyzer=identity(Path(__file__))))
    contrasts = {}
    truth = target['effects'][evaluation]
    rng = np.random.default_rng(9232317)
    bootstrap_indices = [rng.choice(len(evaluation), len(evaluation), replace=True) for _ in range(args.bootstrap)]
    scale_per_doc = np.mean(truth**2, axis=(1, 2))
    for budget in args.budgets:
        first = 'finite_response'+f'_n{budget}'
        first_errors = np.mean((predictions[first][evaluation]-truth)**2, axis=(1, 2))
        for base in ('natural_activation', 'path_calibration', 'activation_amplitude'):
            second = base+f'_n{budget}'
            second_errors = np.mean((predictions[second][evaluation]-truth)**2, axis=(1, 2))
            def difference(indices):
                return float((np.sqrt(first_errors[indices].mean())-np.sqrt(second_errors[indices].mean()))/
                             np.sqrt(scale_per_doc[indices].mean()))
            contrasts[first+'_minus_'+second] = dict(nrmse_difference=difference(np.arange(len(evaluation))),
                ci95=np.quantile([difference(ids) for ids in bootstrap_indices], [.025, .975]).tolist())
    write_json(args.output/'RESPONSE_RESULTS.json', dict(rank=rank, metrics=metrics, paired_contrasts=contrasts,
        inference_unit='Evaluation document; paired resampling includes every member and both conditions together',
        scope='Fixed exposed development cohort and target dictionary',
        singleton_additivity_assumed_for_selection_only=True))
    lines = ['# 有限成员响应预测', '', f'采用source-only交叉验证选择的rank {rank}。评价文档为{len(evaluation)}篇，目标候选为{members}个。', '',
        '| 方法 | 两条件nRMSE | clean RMSE | conditional RMSE | 实际响应RMS | 预测响应RMS |', '|---|---:|---:|---:|---:|---:|']
    for method, values in metrics.items():
        values_to_show = [values['joint']['nrmse']['value'], values['clean']['rmse']['value'],
            values['conditional']['rmse']['value'], values['joint']['actual_effect_rms']['value'], values['joint']['predicted_effect_rms']['value']]
        lines.append('| '+method+' | '+' | '.join(f'{value:.6f}' for value in values_to_show)+' |')
    lines += ['', '拟合仅使用规定的8或16篇目标校准文档。source响应和自然activation基使用同一rank，路径与激活幅度对照按成员、条件分别拟合截距和有符号系数。', '',
        '单成员响应相加用于组成选择。所选完整成员的真实组合响应由独立模型执行记录承担。当前结果条件于历史开发文档及固定字典。', '']
    (args.output/'RESPONSE_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    np.savez_compressed(args.output/'fit_parameters.npz', **basis_parameters)
    np.savez_compressed(args.output/'response_predictions.npz', **predictions,
        row_ids=reference['row_ids'], document_sha256=reference['document_sha256'],
        member_ids=target['member_ids'], sites=target['sites'])
    print(json.dumps(dict(output=args.output.as_posix(), rank=rank, members=members, methods=list(selections))), flush=True)


if __name__ == '__main__':
    main()
