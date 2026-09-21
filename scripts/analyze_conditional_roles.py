import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def load_bank(run):
    index = json.loads((run/'BANK_INDEX.json').read_text())
    values = np.load(run/'responses.npz')['logits'].astype(np.float64)
    assert np.isfinite(values).all()
    clean = values[index['methods'].index('none'), 0]
    lookup = {q['name']: i for i, q in enumerate(index['queries'])}
    edges = [i for i, q in enumerate(index['queries']) if q['role'] == 'restore']
    before = [lookup[index['queries'][i]['ablation_reference']] for i in edges]
    changed = values[:, edges]-values[:, before]
    # 恢复收益采用相对干净输出的平方误差减少，保留放大与反向作用。
    utility = (values[:, before]-clean)**2-(values[:, edges]-clean)**2
    return index, values, clean, np.array(edges), np.array(before), changed, utility


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--target-runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--held-membership', type=Path)
    args = parser.parse_args()
    source_index, sv, sc, edges, before, source_change, su = load_bank(args.source_run)
    keep = np.arange(len(edges))
    if args.held_membership:
        held = set(json.loads(args.held_membership.read_text())['held_requests'])
        keep = np.array([i for i, edge in enumerate(edges) if source_index['queries'][edge]['name'] in held])
        assert len(keep) == len(held)
        edges, before, source_change, su = edges[keep], before[keep], source_change[:, keep], su[:, keep]
    source_method = source_index['methods'].index('source')
    source_utility = su[source_method].mean(-1)
    questions = [source_index['queries'][i] for i in edges]
    early_nodes = list(dict.fromkeys(q['early_node'] for q in questions))
    late_nodes = list(dict.fromkeys(q['restore_node'] for q in questions))
    global_score = {node: float(np.mean([source_utility[i] for i, q in enumerate(questions)
                                         if q['restore_node'] == node])) for node in late_nodes}
    source_lookup = {q['name']: i for i, q in enumerate(source_index['queries'])}
    standalone = {node: float(np.mean((sv[source_method, source_lookup['delete_'+node]]-sc)**2))
                  for node in late_nodes}
    policies = {name: [] for name in ['conditional', 'global', 'standalone']}
    groups = []
    for early in early_nodes:
        indices = [i for i, q in enumerate(questions) if q['early_node'] == early]
        if len(indices) < 2:
            continue
        groups.append(dict(early=early, indices=indices))
        policies['conditional'].append(max(indices, key=lambda i: source_utility[i]))
        policies['global'].append(max(indices, key=lambda i: global_score[questions[i]['restore_node']]))
        policies['standalone'].append(max(indices, key=lambda i: standalone[questions[i]['restore_node']]))
    targets = []
    for run in args.target_runs:
        index, values, clean, e, b, changed, utility = load_bank(run)
        assert index['queries'] == source_index['queries']
        e, b, changed, utility = e[keep], b[keep], changed[:, keep], utility[:, keep]
        source = changed[index['methods'].index('source')]
        summaries = {}
        for mi, method in enumerate(index['methods']):
            if method == 'none':
                continue
            actual = utility[mi].mean(-1)
            rank_groups = [g for g in groups if len(g['indices']) >= 3]
            conditional_rank = [float(spearmanr(source_utility[g['indices']], actual[g['indices']]).statistic)
                                for g in rank_groups
                                if np.ptp(source_utility[g['indices']]) > 0 and np.ptp(actual[g['indices']]) > 0]
            choices = {name: dict(mean_utility=float(actual[ix].mean()),
                        chosen=[questions[i]['restore_node'] for i in ix])
                       for name, ix in policies.items()}
            random_utility = np.mean([actual[g['indices']].mean() for g in groups])
            oracle_utility = np.mean([actual[g['indices']].max() for g in groups])
            identity_indices = [i for i, q in enumerate(index['queries']) if q['role'] == 'identity_control']
            identity_error = float(np.max(np.abs(values[mi, identity_indices]-clean)))
            assert identity_error < 1e-5, (method, identity_error)
            sm = index['methods'].index('source')
            deletion_error = values[mi, b]-values[sm, b]
            restoration_error = values[mi, e]-values[sm, e]
            contrast_error = restoration_error-deletion_error
            error_geometry = dict(deletion_mse=float(np.mean(deletion_error**2)),
                restoration_mse=float(np.mean(restoration_error**2)),
                contrast_mse=float(np.mean(contrast_error**2)),
                deletion_restoration_product=float(np.mean(deletion_error*restoration_error)),
                deletion_contrast_product=float(np.mean(deletion_error*contrast_error)))
            summaries[method] = dict(response_nrmse=float(np.linalg.norm(changed[mi]-source)/np.linalg.norm(source)),
                rank_mean=float(np.mean(conditional_rank)) if conditional_rank else None,
                rank_defined=len(conditional_rank), rank_total=len(rank_groups), policies=choices,
                random_utility=float(random_utility), oracle_utility=float(oracle_utility),
                edge_mean_utility=actual.tolist(), identity_error=identity_error,
                error_geometry=error_geometry)
        targets.append(dict(run=str(run), documents=len(clean), methods=summaries))
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        evidence='Development when source and evaluation documents overlap. Frozen policies require independent confirmation.',
        source_run=str(args.source_run), source_utility=source_utility.tolist(),
        held_membership=str(args.held_membership) if args.held_membership else None,
        groups=groups, policies=policies, questions=questions, targets=targets,
        source_responses_sha256=hashlib.sha256((args.source_run/'responses.npz').read_bytes()).hexdigest())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    for target in targets:
        print(json.dumps(dict(run=target['run'], methods={name: dict(
            response_nrmse=item['response_nrmse'], rank_mean=item['rank_mean'],
            utilities={k: v['mean_utility'] for k, v in item['policies'].items()},
            random=item['random_utility'], oracle=item['oracle_utility'])
            for name, item in target['methods'].items()})))


if __name__ == '__main__':
    main()
