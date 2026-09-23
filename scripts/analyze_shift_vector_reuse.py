import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_shift_member_responses import aggregate_groups, analyze_frozen_groups, constrained_selection, identity, read_json, write_json


def load_reference(index):
    with np.load(index['source_reference']) as raw:
        return {key: raw[key].copy() for key in raw.files}


def native_support(config, scalar_index):
    source = read_json(config['source_manifest'])
    source_ids = {}
    for path in scalar_index['source_blocks']:
        with np.load(path) as block:
            source_ids.setdefault(str(block['site']), []).extend(block['member_ids'].tolist())
    selected = {}
    with np.load(Path(config['relation_run'])/'relation.npz') as relation:
        for site, quota in config['members_by_site'].items():
            candidates = relation[site+'__candidates']
            mask = np.array([member in source_ids[site] for member in source['members'][site]])
            scores = relation[site+'__native'].astype(np.float64)[:, mask].sum(1)[candidates]
            selected[site] = candidates[np.argsort(-scores, kind='stable')[:quota]].astype(int).tolist()
    return selected


def select(run, scalar_run, baseline_selection, calibration_meta, output):
    assert read_json(run/'status.json')['status'] == 'PASS'
    if output.exists():
        raise FileExistsError(output)
    config = read_json(run/'config.resolved.json')
    assert config['target_seed'] == read_json(scalar_run/'config.resolved.json')['target_seed']
    index = read_json(run/'response_index.json')
    scalar_index = read_json(scalar_run/'response_index.json')
    reference, scalar_reference = load_reference(index), load_reference(scalar_index)
    scalar_positions = {str(value): i for i, value in enumerate(scalar_reference['document_sha256'])}
    documents = reference['document_sha256'].tolist()
    positions = np.array([scalar_positions[document] for document in documents])
    previous_meta = read_json(calibration_meta)
    assert Path(previous_meta['run']).resolve() == scalar_run.resolve()
    expected = scalar_reference['document_sha256'][previous_meta['methods']['direct_calibration_n8']['calibration_document_indices']]
    assert len(documents) == 8 and set(documents) == set(expected)
    with np.load(Path(config['frozen_source_run'])/'probe.npz') as head:
        weight = head['weight'].astype(np.float64).ravel()
    source_delta = (reference['whole_W_pooled512'].astype(np.float64)-reference['baseline_pooled512']).astype(np.float64)
    assert source_delta.shape == (8, 2, 512)
    blocks, ids, sites, scalar_effects = [], [], [], []
    for path in index['target_blocks']:
        with np.load(path) as block:
            assert str(block['status']) == 'PASS'
            np.testing.assert_array_equal(block['document_sha256'], reference['document_sha256'])
            blocks.append(block['delta_pooled512'].astype(np.float64))
            ids.extend(block['member_ids'].tolist())
            sites.extend([str(block['site'])]*len(block['member_ids']))
            scalar_effects.append(block['effects'].copy())
    vectors = np.concatenate(blocks, axis=2)
    scalar_effects = np.concatenate(scalar_effects, axis=2)
    ids, sites = np.array(ids), np.array(sites)
    assert vectors.shape == (8, 2, len(ids), 512) and np.isfinite(vectors).all()
    old_effects = {}
    for path in scalar_index['target_blocks']:
        with np.load(path) as block:
            for j, member in enumerate(block['member_ids']):
                old_effects[str(block['site']), int(member)] = block['effects'][positions, :, j]
    assert set(zip(sites.tolist(), ids.tolist())) == set(old_effects)
    old_scalar = np.stack([old_effects[site, int(member)] for site, member in zip(sites, ids)], axis=2)
    design = vectors.transpose(0, 1, 3, 2).reshape(-1, len(ids))
    response = source_delta.reshape(-1)
    binary, selected, diagnostics = constrained_selection(design, response, sites, ids, config['members_by_site'])
    original = read_json(baseline_selection)
    choices = {method: original[method] for method in ('direct_calibration_n8', 'activation_amplitude_n8', 'path_calibration_n8')}
    choices['native_support'] = original['native_support'] if 'native_support' in original else native_support(config, scalar_index)
    choices['finite_vector_n8'] = selected
    assert all(sum(map(len, values.values())) == 22 for values in choices.values())
    for values in choices.values():
        for site, selected_ids in values.items():
            assert len(selected_ids) == config['members_by_site'][site]
            assert set(selected_ids) <= set(ids[sites == site])
    output.mkdir(parents=True)
    write_json(output/'selected_members.json', choices)
    write_json(output/'vector_only_selected_members.json', dict(finite_vector_n8=selected))
    write_json(output/'selection_meta.json', dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        calibration_run=run.as_posix(), scalar_run=scalar_run.as_posix(), calibration_documents=documents,
        target_seed=config['target_seed'], target_candidate_count=len(ids), target_intervention_calls=len(ids)*8*2,
        scalar_observation_dimensions=1, vector_observation_dimensions=512, source_whole_W_dimensions=512,
        coordinate_selection=False, whitening=False, objectives='Unweighted finite pooled-state squared error over 8 documents, both conditions and all 512 coordinates',
        diagnostics=diagnostics, candidate_ids=ids.tolist(), candidate_sites=sites.tolist(),
        projection_checks=dict(vector_to_saved_scalar_max_abs=float(np.max(np.abs(vectors@weight-scalar_effects))),
            source_vector_to_saved_scalar_max_abs=float(np.max(np.abs(source_delta@weight-reference['whole_W_effects']))),
            vector_to_original_scalar_max_abs=float(np.max(np.abs(vectors@weight-old_scalar)))),
        identities=dict(index=identity(run/'response_index.json'), scalar_index=identity(scalar_run/'response_index.json'),
            baseline_selection=identity(baseline_selection), original_calibration=identity(calibration_meta), analyzer=identity(Path(__file__)))))
    np.savez_compressed(output/'vector_selection_parameters.npz', Q=design.T@design/len(design),
        rhs=design.T@response/len(design), binary=binary, member_ids=ids, sites=sites,
        source_delta_pooled512=source_delta, document_sha256=reference['document_sha256'])
    print({'output': output.as_posix(), 'candidates': len(ids), 'target_intervention_calls': len(ids)*16,
        'selected_members': int(binary.sum()), 'exchanges': len(diagnostics['exchanges'])}, flush=True)


def summarize(samples):
    defined = [float(value) for value in samples if value is not None]
    return np.quantile(defined, [.025, .975]).tolist() if defined else None


def read_groups(run, original_weight):
    assert read_json(run/'status.json')['status'] == 'PASS'
    index = read_json(run/'response_index.json')
    reference = load_reference(index)
    baseline = reference['baseline_pooled512'].astype(np.float64)
    source = reference['whole_W_pooled512'].astype(np.float64)-baseline
    result = {}
    for path in index['group_blocks']:
        with np.load(path) as group:
            np.testing.assert_array_equal(group['document_sha256'], reference['document_sha256'])
            method = str(group['method'])
            assert method not in result and str(group['status']) == 'PASS'
            actual = group['pooled512'].astype(np.float64)-baseline
            error = actual-source
            parallel = (error@original_weight)[..., None]*original_weight/float(original_weight@original_weight)
            perpendicular = error-parallel
            result[method] = dict(actual=actual, error=error, parallel=parallel, perpendicular=perpendicular,
                error_squared_norm=np.sum(error**2, axis=-1), parallel_squared_norm=np.sum(parallel**2, axis=-1),
                perpendicular_squared_norm=np.sum(perpendicular**2, axis=-1))
            np.testing.assert_allclose(result[method]['error_squared_norm'], result[method]['parallel_squared_norm']+
                result[method]['perpendicular_squared_norm'], atol=1e-10, rtol=1e-10)
    return reference, baseline, source, result


def cell_mean(values, masks, stratum_balanced):
    if stratum_balanced:
        return float(np.mean([values[mask].mean() for mask in masks]))
    return float(values.mean())


def evaluation_statistics(targets, heads, professions, genders, documents, stratum_balanced=False):
    per_head, normalized, overall_task = {}, {}, {}
    for target_index, (_, baseline, source, methods) in enumerate(targets):
        for head in heads:
            indices = documents[np.isin(professions[documents], [head['negative'], head['positive']])]
            assert len(indices)
            labels = professions[indices] == head['positive']
            group_masks = [(labels == value) & (genders[indices] == gender) for value in (0, 1) for gender in (0, 1)]
            cache = head['cache'][target_index]
            reference_effect = cache['source'][indices]
            source_scale_squared = cell_mean(reference_effect**2, group_masks, stratum_balanced)
            base_logits = cache['baseline'][indices]
            for method, arrays in methods.items():
                prefix = f't{target_index}/{head["name"]}/{method}'
                actual, error, a, b = [cache[method][key][indices] for key in ('actual', 'error', 'parallel', 'perpendicular')]
                square = cell_mean(error**2, group_masks, stratum_balanced)
                head_normalized = square/source_scale_squared if source_scale_squared > 0 else None
                normalized.setdefault(method, []).append(head_normalized)
                for name, condition_ids in [('joint', [0, 1]), ('clean', [0]), ('conditional', [1])]:
                    err, src, act = error[:, condition_ids], reference_effect[:, condition_ids], actual[:, condition_ids]
                    av, bv = a[:, condition_ids], b[:, condition_ids]
                    values = dict(rmse=float(np.sqrt(cell_mean(err**2, group_masks, stratum_balanced))),
                        source_effect_rms=float(np.sqrt(cell_mean(src**2, group_masks, stratum_balanced))),
                        actual_effect_rms=float(np.sqrt(cell_mean(act**2, group_masks, stratum_balanced))),
                        parallel_squared=cell_mean(av**2, group_masks, stratum_balanced),
                        perpendicular_squared=cell_mean(bv**2, group_masks, stratum_balanced),
                        cross_term=cell_mean(2*av*bv, group_masks, stratum_balanced),
                        total_squared=cell_mean(err**2, group_masks, stratum_balanced))
                    values['nrmse_head_common_scale'] = values['rmse']/np.sqrt(source_scale_squared) if source_scale_squared > 0 else None
                    for operation, logits in [('actual', base_logits[:, condition_ids]+act),
                                              ('source', base_logits[:, condition_ids]+src), ('baseline', base_logits[:, condition_ids])]:
                        correct = (logits > 0) == labels[:, None]
                        values[operation+'_accuracy'] = cell_mean(correct, group_masks, stratum_balanced)
                        values[operation+'_worst_group'] = min(float(correct[group].mean()) for group in group_masks)
                    per_head.update({prefix+'/'+name+'/'+key: value for key, value in values.items()})
                    overall_task.setdefault((method, name), []).append(values)
    overall = {}
    for method, values in normalized.items():
        overall[method+'/primary_nrmse'] = float(np.sqrt(np.mean(values))) if all(value is not None for value in values) else None
    for (method, condition), rows in overall_task.items():
        for key in ('actual_accuracy', 'source_accuracy', 'baseline_accuracy', 'actual_worst_group', 'source_worst_group',
                    'parallel_squared', 'perpendicular_squared', 'cross_term', 'total_squared'):
            overall[method+'/'+condition+'/'+key] = float(np.mean([row[key] for row in rows]))
    for method in targets[0][3]:
        all_masks = [(professions[documents] == profession) & (genders[documents] == gender)
            for profession in sorted(set(professions)) for gender in (0, 1)]
        for condition, ids in [('joint', [0, 1]), ('clean', [0]), ('conditional', [1])]:
            for key in ('error_squared_norm', 'parallel_squared_norm', 'perpendicular_squared_norm'):
                overall[method+'/hidden/'+condition+'/'+key] = float(np.mean([
                    cell_mean(target[3][method][key][documents][:, ids], all_masks, stratum_balanced) for target in targets]))
    return overall, per_head


def evaluate(runs, panel_path, heads_directory, original_head, output, bootstrap, stratum_balanced=False):
    if output.exists():
        raise FileExistsError(output)
    panel = read_json(panel_path)
    with np.load(original_head) as head:
        original_weight = head['weight'].astype(np.float64).ravel()
    targets = [read_groups(run, original_weight) for run in runs]
    reference = targets[0][0]
    for target in targets[1:]:
        np.testing.assert_array_equal(target[0]['document_sha256'], reference['document_sha256'])
        assert set(target[3]) == set(targets[0][3])
    row_map = {row['document_sha256']: row for row in panel['rows']}
    rows = [row_map[str(value)] for value in reference['document_sha256']]
    assert len(row_map) == len(rows)
    professions = np.array([row['profession'] for row in rows])
    genders = np.array([row['gender'] for row in rows])
    heads = []
    for task in panel['tasks']:
        path = heads_directory/f'none__full__{task["name"]}__probe42.npz'
        with np.load(path) as raw:
            assert raw['weight'].shape == (1, 512) and raw['bias'].shape == (1,)
            heads.append(dict(task, weight=raw['weight'].astype(np.float64).ravel(), bias=float(raw['bias'][0]), identity=identity(path)))
    assert len(heads) == 4
    for head in heads:
        head['cache'] = []
        for _, baseline, source, methods in targets:
            cache = dict(source=source@head['weight'], baseline=baseline@head['weight']+head['bias'])
            for method, arrays in methods.items():
                cache[method] = {key: arrays[key]@head['weight'] for key in ('actual', 'error', 'parallel', 'perpendicular')}
                values = cache[method]
                np.testing.assert_allclose(values['error'], values['actual']-cache['source'], atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(values['error'], values['parallel']+values['perpendicular'], atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(values['error']**2, values['parallel']**2+values['perpendicular']**2+
                    2*values['parallel']*values['perpendicular'], atol=1e-10, rtol=1e-10)
            head['cache'].append(cache)
    points, head_points = evaluation_statistics(targets, heads, professions, genders, np.arange(len(rows)), stratum_balanced)
    draws = {key: [] for key in points}
    head_draws = {key: [] for key in head_points}
    strata = [np.flatnonzero((professions == profession) & (genders == gender)) for profession in sorted(set(professions)) for gender in (0, 1)]
    assert len(strata) == 8 and all(len(indices) for indices in strata)
    rng = np.random.default_rng(9232341)
    for iteration in range(bootstrap):
        sampled = np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in strata])
        values, head_values = evaluation_statistics(targets, heads, professions, genders, sampled, stratum_balanced)
        for key, value in values.items():
            draws[key].append(value)
        for key, value in head_values.items():
            head_draws[key].append(value)
        if (iteration+1) % 250 == 0:
            print({'bootstrap_completed': iteration+1, 'bootstrap_total': bootstrap}, flush=True)
    comparisons = {}
    for method in targets[0][3]:
        if method == 'finite_vector_n8':
            continue
        differences = []
        for first, second in zip(draws['finite_vector_n8/primary_nrmse'], draws[method+'/primary_nrmse']):
            differences.append(first-second if first is not None and second is not None else None)
        a, b = points['finite_vector_n8/primary_nrmse'], points[method+'/primary_nrmse']
        comparisons['finite_vector_n8_minus_'+method] = dict(value=a-b if a is not None and b is not None else None,
            ci95=summarize(differences))
    output.mkdir(parents=True)
    write_json(output/'VECTOR_REUSE_RESULTS.json', dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        statistics={key: dict(value=value, ci95=summarize(draws[key])) for key, value in points.items()},
        per_head={key: dict(value=value, ci95=summarize(head_draws[key])) for key, value in head_points.items()},
        paired_contrasts=comparisons, documents=len(rows), fixed_targets=[read_json(run/'config.resolved.json')['target_seed'] for run in runs],
        heads=[{key: value for key, value in head.items() if key not in ('weight', 'bias', 'cache')} for head in heads],
        primary='Square root of equal mean over fixed heads and targets of head-specific MSE divided by the same head source effect mean square; both conditions included',
        bootstrap=dict(draws=bootstrap, seed=9232341, unit='Biography document within profession/gender strata; shared heads, conditions and targets are paired'),
        task_aggregation='Equal mean of per-head and per-target accuracy, and of each head/target profession-by-gender minimum group accuracy',
        weighting=dict(stratum_balanced=stratum_balanced,
            cell_counts={f'profession{profession}_gender{gender}': int(np.sum((professions == profession) & (genders == gender)))
                for profession in sorted(set(professions)) for gender in (0, 1)},
            within_head='Each of the four profession/gender cells has weight 1/4 for squared errors, source scales, accuracy and mechanism moments' if stratum_balanced else 'Each document within a head has equal weight',
            hidden='Each of all eight profession/gender cells has weight 1/8' if stratum_balanced else 'Each document has equal weight',
            bootstrap='Each cell draws its observed document count with replacement; duplicate occurrences retain their stratum and all methods, heads and targets share the draw'),
        zero_source_effect='Head normalization is undefined when its source effect mean square is zero; raw RMSE retained, no denominator added',
        mechanism='Actual group error projected parallel and orthogonal to original head; later-head squared terms and signed cross term retained',
        identities=dict(panel=identity(panel_path), original_head=identity(original_head), analyzer=identity(Path(__file__)),
            groups=[identity(run/'response_index.json') for run in runs])))
    arrays = dict(document_sha256=reference['document_sha256'], professions=professions, genders=genders)
    for target_index, (_, baseline, source, methods) in enumerate(targets):
        arrays[f't{target_index}__baseline'] = baseline
        arrays[f't{target_index}__source_effect'] = source
        for method, values in methods.items():
            arrays.update({f't{target_index}__{method}__{key}': value for key, value in values.items()})
    np.savez_compressed(output/'vector_reuse_arrays.npz', **arrays)
    lines = ['# 固定后来读出的功能解释复用', '', '| 方法 | 主要nRMSE | clean职业准确率 | conditional职业准确率 |', '|---|---:|---:|---:|']
    for method in targets[0][3]:
        keys = ('primary_nrmse', 'clean/actual_accuracy', 'conditional/actual_accuracy')
        values = [points[method+'/'+key] for key in keys]
        lines.append('| '+method+' | '+' | '.join('不可归一化' if value is None else f'{value:.6f}' for value in values)+' |')
    lines += ['', '四个冻结head及固定目标共同使用文档配对统计。每个head先按自身source两背景作用归一化，再等权汇总平方误差。', '',
        'JSON保存逐head真实误差、作用幅度、职业准确率与最差组，以及原head平行项、正交项和有符号交叉项。NPZ保存真实完整组作用向量及投影。', '']
    (output/'VECTOR_REUSE_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print({'output': output.as_posix(), 'documents': len(rows), 'heads': len(heads), 'targets': len(runs)}, flush=True)


def original_endpoint(vector_runs, scalar_reports, output, bootstrap):
    if output.exists():
        raise FileExistsError(output)
    assert len(vector_runs) == len(scalar_reports)
    output.mkdir(parents=True)
    merged_paths = []
    for run, scalar_report in zip(vector_runs, scalar_reports):
        seed = read_json(run/'config.resolved.json')['target_seed']
        vector_directory = output/f't{seed}_vector'
        analyze_frozen_groups(run, vector_directory, bootstrap)
        vector = read_json(vector_directory/'GROUP_RESULTS.json')
        scalar = read_json(scalar_report)
        keep = ('direct_calibration_n8', 'activation_amplitude_n8', 'path_calibration_n8')
        vector_rows = vector['document_values']
        assert {row['method'] for row in vector_rows} == {'finite_vector_n8'}
        scalar_rows = [row for row in scalar['document_values'] if row['method'] in keep]
        scalar_reference = {(row['document_sha256'], row['condition']): row for row in scalar_rows if row['method'] == keep[0]}
        assert set(scalar_reference) == {(row['document_sha256'], row['condition']) for row in vector_rows}
        for row in vector_rows:
            old = scalar_reference[row['document_sha256'], row['condition']]
            assert (row['label'], row['gender']) == (old['label'], old['gender'])
        source_difference = max(abs(row['source_effect']-scalar_reference[row['document_sha256'], row['condition']]['source_effect']) for row in vector_rows)
        directory = output/f't{seed}_merged'
        directory.mkdir()
        write_json(directory/'GROUP_RESULTS.json', dict(statistics={**{name: scalar['statistics'][name] for name in keep}, **vector['statistics']},
            document_values=scalar_rows+vector_rows, source_replay_maximum_effect_difference=source_difference,
            identities=dict(vector=identity(vector_directory/'GROUP_RESULTS.json'), scalar=identity(scalar_report)),
            scope='Same 64 exposed evaluation documents; frozen scalar groups reused, new vector group executed; each actual run retains its own source response'))
        merged_paths.append(directory)
    aggregate_groups(merged_paths, output/'cohort', bootstrap)
    write_json(output/'COMPARISON_SCOPE.json', dict(methods=list(keep)+['finite_vector_n8'],
        original_endpoint='Fixed professor/nurse head on the original 64 evaluation documents, both conditions',
        reused_scalar_reports=[identity(path) for path in scalar_reports],
        vector_runs=[run.as_posix() for run in vector_runs],
        native_support=dict(target2='Existing binary-member result in artifacts/scientific_reform_20260923/MEMBER_BINARY.json',
            target3=None, common_cohort_included=False),
        missing_value_meaning='The target3 original-endpoint native result was not measured; no GPU run was added'))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    selection = sub.add_parser('select')
    for name in ('run', 'scalar-run', 'baseline-selection', 'calibration-meta', 'output'):
        selection.add_argument('--'+name, type=Path, required=True)
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--runs', type=Path, nargs='+', required=True)
    for name in ('panel', 'heads-directory', 'original-head', 'output'):
        evaluation.add_argument('--'+name, type=Path, required=True)
    evaluation.add_argument('--bootstrap', type=int, default=1000)
    evaluation.add_argument('--stratum-balanced', action='store_true')
    original = sub.add_parser('original')
    original.add_argument('--vector-runs', type=Path, nargs='+', required=True)
    original.add_argument('--scalar-reports', type=Path, nargs='+', required=True)
    original.add_argument('--output', type=Path, required=True)
    original.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.mode == 'select':
        select(args.run, args.scalar_run, args.baseline_selection, args.calibration_meta, args.output)
    elif args.mode == 'evaluate':
        evaluate(args.runs, args.panel, args.heads_directory, args.original_head, args.output, args.bootstrap, args.stratum_balanced)
    else:
        original_endpoint(args.vector_runs, args.scalar_reports, args.output, args.bootstrap)


if __name__ == '__main__':
    main()
