from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import platform
import sys
import time
import traceback

import numpy as np
import torch
import analyze_operation_granularity as granularity
from analyze_operation_granularity import Progress, digest, fit_direction, load_codes, matrix_from_fit, members, write
from threadpoolctl import threadpool_limits
from scipy.optimize import linear_sum_assignment


def decoder_rows(config, mechanism, seed, ids):
    state = torch.load(Path(config['checkpoint_directory']) / f'{mechanism}_seed{seed}.pt', map_location='cpu', weights_only=True)
    decoder = state['decoder.weight'].T if mechanism == 'topk' else state['W_dec']
    return decoder[ids].numpy().astype(np.float64)


def prepare_pw(config, mechanism, seed, source_ids, target_ids, directory, progress):
    destination = directory / f'{mechanism}_s{seed}_t3_pw.npz'
    if destination.exists():
        return destination
    existing = Path(config['pw_run']) / f'{mechanism}_s{seed}_t3_global_pw.npz'
    started = time.perf_counter()
    if len(source_ids) == len(target_ids) == 8192 and existing.exists():
        with np.load(existing) as saved:
            permutation = saved['permutation']
            correlation = saved['correlation']
        np.savez_compressed(destination, source_ids=source_ids, target_ids=target_ids, permutation=permutation, correlation=correlation)
        write(destination.with_suffix('.json'), dict(reused_from=str(existing), sha256=digest(existing), compute_seconds=0, materialization_seconds=time.perf_counter()-started))
        return destination
    cache = Path(config['reference_run'])
    source = load_codes(cache / f'natural_discovery_{mechanism}_seed{seed}.npz')[:, source_ids]
    target = load_codes(cache / f'natural_discovery_{mechanism}_seed3.npz')[:, target_ids]
    sm, tm = np.asarray(source.mean(0)).ravel(), np.asarray(target.mean(0)).ravel()
    sv = np.asarray(source.power(2).mean(0)).ravel() - sm**2
    tv = np.asarray(target.power(2).mean(0)).ravel() - tm**2
    covariance = (source.T @ target / source.shape[0]).toarray()
    covariance -= sm[:, None] * tm[None, :]
    denominator = np.sqrt(np.maximum(sv, 0)[:, None] * np.maximum(tv, 0)[None, :])
    correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-14)
    del covariance, denominator
    progress.report('HUNGARIAN_START', mechanism=mechanism, source_seed=seed)
    rows, permutation = linear_sum_assignment(-np.abs(correlation))
    assert np.array_equal(rows, np.arange(len(source_ids)))
    np.savez_compressed(destination, source_ids=source_ids, target_ids=target_ids, permutation=permutation, correlation=correlation[rows, permutation])
    write(destination.with_suffix('.json'), dict(reused_from=None, compute_seconds=time.perf_counter()-started, rule='Original complete absolute-Pearson Hungarian recipe; empirical discovery correlation; no task response'))
    return destination


def fit_pair(config, mechanism, source_seed, source_ids, target_ids, directory, progress):
    path = directory / f'{mechanism}_s{source_seed}_t3_ridge.npz'
    if path.exists():
        progress.report('REUSE_RIDGE', mechanism=mechanism, source_seed=source_seed)
        return path
    cache = Path(config['reference_run'])
    source = load_codes(cache / f'natural_discovery_{mechanism}_seed{source_seed}.npz')
    target = load_codes(cache / f'natural_discovery_{mechanism}_seed3.npz')
    assert source.shape == target.shape == (8192, 8192)
    source, target = source[:, source_ids], target[:, target_ids]
    progress.report('DISCOVERY_GRAMS', mechanism=mechanism, source_seed=source_seed)
    source_gram = (source.T @ source).toarray() / source.shape[0]
    target_gram = (target.T @ target).toarray() / target.shape[0]
    cross = (source.T @ target).toarray() / source.shape[0]
    source_rms = np.sqrt(source_gram.diagonal())
    target_rms = np.sqrt(target_gram.diagonal())
    c_st, b_st, r_st = fit_direction(target_gram, cross, source_rms, target_rms, target_ids, config['candidates'], progress, f'3_to_{source_seed}')
    c_ts, b_ts, r_ts = fit_direction(source_gram, cross.T, target_rms, source_rms, source_ids, config['candidates'], progress, f'{source_seed}_to_3')
    np.savez_compressed(path, c_st=c_st, b_st=b_st, r_st=r_st, c_ts=c_ts, b_ts=b_ts, r_ts=r_ts, source_rms=source_rms, target_rms=target_rms, source_ids=source_ids, target_ids=target_ids)
    progress.report('RIDGE_SAVED', mechanism=mechanism, source_seed=source_seed)
    return path


def prepare(config, mechanism, directory, progress):
    completed = directory / f'{mechanism}_target_group.json'
    if completed.exists():
        return json.loads(completed.read_text())
    context = json.loads(Path(config['frozen_groups']).read_text())['nodes'][mechanism]
    group1, group2 = np.array(context['source_ids']), np.array(context['target_ids'])
    tree_path = Path(config['source_tree_run']) / f'{mechanism}_tree.npz'
    with np.load(tree_path) as saved:
        tree = {key: saved[key] for key in ('children', 'source1_ids', 'source2_ids', 'edge_members', 'edge_weight')}
        a, b = members(tree, context['node'], len(tree['source1_ids']), len(tree['source2_ids']))
        assert np.array_equal(tree['source1_ids'][a], group1)
        assert np.array_equal(tree['source2_ids'][b], group2)
        edge_ids = tree['edge_members']
        group_weights = np.zeros((len(group1), len(group2)))
        source_lookup = {int(value): i for i, value in enumerate(group1)}
        second_lookup = {int(value): i for i, value in enumerate(group2)}
        for pair, weight in zip(edge_ids, tree['edge_weight']):
            first, second = int(tree['source1_ids'][pair[0]]), int(tree['source2_ids'][pair[1]])
            if first in source_lookup and second in second_lookup:
                group_weights[source_lookup[first], second_lookup[second]] = weight
        owner = group1[np.argmax(group_weights, axis=0)]
    first_half = group1[:(len(group1)+1)//2]
    second_half = group1[(len(group1)+1)//2:]
    source_masks = {'source1': dict(whole=group1.tolist(), H1=first_half.tolist(), H2=second_half.tolist()), 'source2': dict(whole=group2.tolist(), H1=group2[np.isin(owner, first_half)].tolist(), H2=group2[np.isin(owner, second_half)].tolist())}
    count = config.get('feature_limit', 8192)
    target_ids = np.arange(count)
    packed, costs, paths, pw_paths, weight_matrices, source_indices = {}, [], [], [], [], []
    for seed, group in [(1, group1), (2, group2)]:
        source_ids = np.sort(np.concatenate([group, np.setdiff1d(np.arange(8192), group)[:count-len(group)]]))
        path = fit_pair(config, mechanism, seed, source_ids, target_ids, directory, progress)
        paths.append(str(path))
        with np.load(path) as fit:
            forward = matrix_from_fit(fit['c_st'], np.square(fit['b_st']), len(target_ids))
            reverse = matrix_from_fit(fit['c_ts'], np.square(fit['b_ts']), len(source_ids)).T.tocsr()
            weights = forward.maximum(reverse).tocsr()
            weight_matrices.append(weights)
            source_indices.append(source_ids)
            inside = np.isin(source_ids, group)
            group_weight = np.asarray(weights[inside].sum(axis=0)).ravel()
            outside_weight = np.asarray(weights[~inside].sum(axis=0)).ravel()
            costs.append((group_weight, outside_weight))
            edge = weights.tocoo()
            packed.update({f's{seed}_rows': edge.row, f's{seed}_columns': edge.col, f's{seed}_weights': edge.data, f's{seed}_shape': weights.shape, f's{seed}_source_ids': source_ids, f's{seed}_frozen_group': group, f's{seed}_inside_weight': group_weight, f's{seed}_outside_weight': outside_weight})
        pw_paths.append(prepare_pw(config, mechanism, seed, source_ids, target_ids, directory, progress))
    inside_weight = (costs[0][0] + costs[1][0]) / 2
    outside_weight = (costs[0][1] + costs[1][1]) / 2
    target_mask = inside_weight > outside_weight
    path = directory / f'{mechanism}_weights.npz'
    np.savez_compressed(path, **packed, target_ids=target_ids, inside_weight=inside_weight, outside_weight=outside_weight, target_mask=target_mask)
    cache = Path(config['reference_run'])
    source = load_codes(cache / f'natural_discovery_{mechanism}_seed1.npz')[:, group1] @ decoder_rows(config, mechanism, 1, group1)
    target = load_codes(cache / f'natural_discovery_{mechanism}_seed3.npz')
    pw_options, pw_arrays = [], []
    for seed, group, pw_path in zip((1, 2), (group1, group2), pw_paths):
        with np.load(pw_path) as archive:
            saved = {key: archive[key] for key in ('source_ids', 'target_ids', 'permutation')}
        pw_arrays.append(saved)
        selected = saved['target_ids'][saved['permutation'][np.isin(saved['source_ids'], group)]]
        actual = target[:, selected] @ decoder_rows(config, mechanism, 3, selected)
        pw_options.append(dict(source_seed=seed, target_members=np.sort(selected).tolist(), discovery_source1_contribution_sse=float(np.square(actual-source).sum()), path=str(pw_path)))
    chosen = min(pw_options, key=lambda item: (item['discovery_source1_contribution_sse'], item['source_seed']))
    masks = {**source_masks, 'graph': {}, 'PW_selected': {}}
    for request in ('whole', 'H1', 'H2'):
        inside_sum, outside_sum = np.zeros(len(target_ids)), np.zeros(len(target_ids))
        for seed, weights, ids in zip((1, 2), weight_matrices, source_indices):
            included = np.isin(ids, masks[f'source{seed}'][request])
            inside_sum += np.asarray(weights[included].sum(axis=0)).ravel() / 2
            outside_sum += np.asarray(weights[~included].sum(axis=0)).ravel() / 2
        masks['graph'][request] = target_ids[inside_sum > outside_sum].tolist()
        saved = pw_arrays[chosen['source_seed']-1]
        included = np.isin(saved['source_ids'], masks[f'source{chosen["source_seed"]}'][request])
        masks['PW_selected'][request] = np.sort(saved['target_ids'][saved['permutation'][included]]).tolist()
    saved = pw_arrays[0]
    lookup = dict(zip(saved['source_ids'].tolist(), saved['target_ids'][saved['permutation']].tolist()))
    pw_source1_members = [lookup[int(member)] for member in group1]
    assert masks['graph']['whole'] == target_ids[target_mask].tolist()
    result = dict(mechanism=mechanism, node=context['node'], source1_members=group1.tolist(), source2_members=group2.tolist(), target_seed=3, target_members=target_ids[target_mask].tolist(), ridge_paths=paths, weights_path=str(path), raw_ridge_path=paths[0], raw_definition='Signed full sparse R reconstructed from source_ids/target_ids/c_st/r_st; delta=-(zt@R.T)@source1_decoder, restricting source rows for each request', masks=masks, source2_source1_owner=owner.tolist(), PW_source1_members=pw_source1_members, PW_options=pw_options, PW_selected_source_seed=chosen['source_seed'], PW_selected_members=chosen['target_members'], rule='Two sources equally weighted; select iff summed within-group edge weight exceeds summed outside-group edge weight; equality selects zero', weight_definition='Maximum of two directional RMS-standardized signed ridge coefficient squares; missing edges contribute zero', coefficient_mask_cost=float(np.where(target_mask, outside_weight, inside_weight).sum()), interpretation='Coefficient-level binary mask objective; physical contribution and model effects are evaluated separately', smoke=count < 8192)
    write(completed, result)
    progress.report('TARGET_GROUP_SAVED', mechanism=mechanism, selected_members=int(target_mask.sum()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assert config['source_seeds'] == [1, 2] and config['target_seed'] == 3
    directory = Path(config['run_storage_root']) / config['run_id']
    directory.mkdir(parents=True, exist_ok=args.resume)
    code_paths = [Path(__file__), Path(granularity.__file__)]
    if args.resume:
        assert json.loads((directory / 'config.json').read_text()) == config
        for path in code_paths:
            assert digest(directory / path.name) == digest(path)
    else:
        write(directory / 'config.json', config)
        for path in code_paths:
            (directory / path.name).write_bytes(path.read_bytes())
    progress = Progress(config, directory)
    write(directory / 'status.json', dict(status='RUNNING', started_at_utc=progress.started_at_utc))
    write(directory / 'environment.json', dict(python=sys.executable, python_version=platform.python_version(), numpy=np.__version__, torch=torch.__version__, scipy=granularity.scipy.__version__, psutil=granularity.psutil.__version__, cpu_threads=config['cpu_threads'], gpu_used=False))
    try:
        identity_paths = [Path(config['frozen_groups'])]
        for mechanism in config['mechanisms']:
            identity_paths.append(Path(config['source_tree_run']) / f'{mechanism}_tree.npz')
            for seed in (1, 2, 3):
                identity_paths.append(Path(config['reference_run']) / f'natural_discovery_{mechanism}_seed{seed}.npz')
                checkpoint = Path(config['checkpoint_directory']) / f'{mechanism}_seed{seed}.pt'
                identity_paths.append(checkpoint)
                state = torch.load(checkpoint, map_location='cpu', weights_only=True)
                decoder = state['decoder.weight'].T if mechanism == 'topk' else state['W_dec']
                assert tuple(decoder.shape) == (8192, 1024)
                del state, decoder
            for seed in (1, 2):
                existing = Path(config['pw_run']) / f'{mechanism}_s{seed}_t3_global_pw.npz'
                if existing.exists():
                    identity_paths.append(existing)
        identity = {str(path): digest(path) for path in identity_paths}
        identity_path = directory / 'input_identity.json'
        if args.resume:
            assert json.loads(identity_path.read_text()) == identity
        else:
            write(identity_path, identity)
        torch.set_num_threads(config['cpu_threads'])
        with threadpool_limits(config['cpu_threads']):
            results = [prepare(config, mechanism, directory, progress) for mechanism in config['mechanisms']]
        cost = progress.report('COMPLETE')
        cost.update(started_at_utc=progress.started_at_utc, ended_at_utc=datetime.now(timezone.utc).isoformat(), generated_bytes_before_final_reports=sum(path.stat().st_size for path in directory.iterdir() if path.is_file()))
        assert cost['generated_bytes_before_final_reports'] < config['maximum_new_bytes']
        write(directory / 'result.json', dict(run_id=config['run_id'], status='PASS', results=results, cost=cost, scope=config['scope']))
        write(directory / 'status.json', dict(status='PASS', cost=cost))
    except Exception:
        write(directory / 'status.json', dict(status='FAIL', error=traceback.format_exc(), started_at_utc=progress.started_at_utc, ended_at_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter()-progress.wall, cpu_seconds=time.process_time()-progress.cpu, peak_rss_bytes=progress.peak))
        raise


if __name__ == '__main__':
    main()
