from pathlib import Path
import argparse
import json
import platform
import sys
import time
import traceback

import numpy as np

from run_causalgym_multisite import MultisiteWork, write
from run_shift_explanation import source_groups
from train_shift_dictionaries import site_module
from ccad.native_group_selection import contribution_moments, bounded_quadratic


def binary_single_swap(q, rhs, initial, feature_ids):
    q = np.asarray(q, dtype=np.float64)
    q = (q+q.T)/2
    rhs = np.asarray(rhs, dtype=np.float64)
    binary = np.asarray(initial, dtype=np.float64).copy()
    feature_ids = np.asarray(feature_ids)
    assert np.isfinite(q).all() and np.isfinite(rhs).all()
    assert np.isin(binary, [0., 1.]).all() and len(np.unique(feature_ids)) == len(binary)
    budget = int(binary.sum())
    order = np.argsort(feature_ids, kind='stable')
    objective = float(binary@q@binary-2*rhs@binary)
    initial_objective = objective
    exchanges = []
    while True:
        outgoing = order[binary[order] == 1]
        incoming = order[binary[order] == 0]
        residual = q@binary-rhs
        change = (2*(residual[incoming][None, :]-residual[outgoing][:, None])
            +np.diag(q)[outgoing][:, None]+np.diag(q)[incoming][None, :]
            -2*q[np.ix_(outgoing, incoming)])
        oi, ji = np.unravel_index(np.argmin(change), change.shape)
        minimum = float(change[oi, ji])
        # 浮点精度只用于排除计算舍入产生的交换。
        precision = 32*np.finfo(np.float64).eps*max(1., abs(objective), float(np.max(np.abs(q))), float(np.max(np.abs(rhs))))
        if minimum >= -precision:
            break
        i, j = int(outgoing[oi]), int(incoming[ji])
        binary[i], binary[j] = 0., 1.
        updated = float(binary@q@binary-2*rhs@binary)
        assert updated < objective and abs((updated-objective)-minimum) <= precision*max(1, len(binary))
        assert int(binary.sum()) == budget and np.isin(binary, [0., 1.]).all()
        exchanges.append(dict(removed_feature=int(feature_ids[i]), added_feature=int(feature_ids[j]),
            before=objective, after=updated, predicted_change=minimum))
        objective = updated
    return binary, dict(initial_objective=initial_objective, final_objective=objective,
        exchanges=len(exchanges), exchange_history=exchanges, best_remaining_swap_change=minimum,
        numerical_precision=precision, stopping='No improving single exchange at floating-point precision',
        tie_order='Removed feature ID, then added feature ID, ascending; exact ties', global_optimality_claim=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    work = MultisiteWork(c, args.config, ['scripts/run_shift_conditional_members.py',
        'scripts/run_shift_explanation.py', 'scripts/train_shift_dictionaries.py',
        'scripts/run_causalgym_multisite.py', 'scripts/run_r011s1_raw_hook_asset.py',
        'src/ccad/native_group_selection.py', 'src/ccad/artifacts.py'])
    hooks = []
    error = None
    try:
        import torch
        import transformers
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        work.torch = torch
        work.device = torch.device(c['device'])
        torch.cuda.set_device(work.device)
        torch.cuda.reset_peak_memory_stats(work.device)
        sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        work.environment = dict(python=sys.executable, python_version=platform.python_version(),
            torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
            cpu_threads=2, matmul_precision='highest')
        source = json.loads(work.checked(c['source_manifest'], 'Published source feature IDs', 'MIT').read_text())
        groups, _ = source_groups(work.checked(c['notebook'], 'Published source annotations', 'MIT'), source['members'])
        bank = np.load(work.checked(c['source_parameters'], 'Published source parameters', 'MIT'))
        relation = np.load(work.checked(Path(c['relation_run'])/'relation.npz', 'Frozen R59 candidate pool and relation'))
        sites = list(source['members'])
        wsites = list(c['members_by_site'])
        assert set(wsites) == set(groups['associated_words'])
        parameters, targets, candidates, decoders, source_decoders = {}, {}, {}, {}, {}
        native_scores, basis_export = {}, {}
        for site in sites:
            s = {key: torch.tensor(bank[site+'__'+key], device=work.device)
                 for key in ['encoder', 'encoder_bias', 'decoder', 'center']}
            s['p'] = torch.tensor([i in groups['pronouns'].get(site, []) for i in source['members'][site]], device=work.device)
            s['w'] = torch.tensor([i in groups['associated_words'].get(site, []) for i in source['members'][site]], device=work.device)
            assert not bool((s['p'] & s['w']).any())
            parameters[site] = s
            if site not in wsites:
                continue
            state = torch.load(work.checked(Path(c['target_directory'])/f'{site}_seed{c["target_seed"]}.pt',
                'Original target SAE checkpoint'), map_location=work.device, weights_only=True)
            target = AutoEncoderTopK(512, state['encoder.weight'].shape[0], int(state['k'])).to(work.device)
            target.load_state_dict(state)
            target.eval().requires_grad_(False)
            targets[site] = target
            candidates[site] = torch.tensor(relation[site+'__candidates'], device=work.device, dtype=torch.long)
            assert len(candidates[site].unique()) == len(candidates[site])
            decoders[site] = target.decoder.weight.T.detach()[candidates[site]]
            source_decoders[site] = s['decoder'][s['w']]
            native_scores[site] = torch.tensor(relation[site+'__native'], dtype=torch.float64)[:, s['w'].cpu()].sum(1)[candidates[site].cpu()]
            basis_export.update({site+'__candidates': candidates[site].cpu().numpy(),
                site+'__target_decoder': decoders[site].cpu().numpy(),
                site+'__source_W_decoder': source_decoders[site].cpu().numpy(),
                site+'__source_W_ids': np.array(source['members'][site])[s['w'].cpu().numpy()]})
        assert sum(c['members_by_site'].values()) == 22
        np.savez_compressed(work.run/'candidate_basis.npz', **basis_export)
        probe = np.load(work.checked(Path(c['frozen_source_run'])/'probe.npz', 'Fixed original source profession head'))
        pw = torch.tensor(probe['weight'], device=work.device)
        pb = torch.tensor(probe['bias'], device=work.device)
        panel = json.loads(work.checked(c['evaluation_panel'], 'Previously exposed development panel').read_text())
        rows = []
        for label in [0, 1]:
            for gender in [0, 1]:
                cell = sorted([r for r in panel['rows'] if r['split'] == c['evaluation_split']
                    and r['label'] == label and r['gender'] == gender], key=lambda r: r['document_sha256'])[:32]
                assert len(cell) == 32
                for split, selected in [('fit', cell[:c['fit_per_group']]),
                        ('evaluation', cell[16:16+c['evaluation_per_group']])]:
                    rows.extend([dict(r, selection_split=split) for r in selected])
        assert len({r['document_sha256'] for r in rows}) == len(rows)
        write(work.run/'selection_membership.json', dict(rows=rows, historical_identity='Exposed development documents'))
        for name in ['config.json', 'tokenizer.json', 'model.safetensors']:
            work.checked(Path(c['model_local_dir'])/name, 'Pinned Pythia70M', 'Apache-2.0')
        model = transformers.AutoModelForCausalLM.from_pretrained(c['model_local_dir'], local_files_only=True,
            dtype=torch.float32, attn_implementation='eager').eval().to(work.device)
        model.requires_grad_(False)
        model.config.use_cache = False
        methods = ['native_support', 'native_natural', 'native_operation', 'source_raw_path']
        mode, condition, alpha, gradient = 'source', 0, 0., False
        mask, pooled, selected_masks = None, None, {}
        observed, leaves = {}, {}
        backward_sequences = 0

        def hook(site):
            def apply(module, inputs, out):
                nonlocal pooled
                x = out[0] if isinstance(out, tuple) else out
                s = parameters[site]
                zs = torch.relu((x-s['center'])@s['encoder'].T+s['encoder_bias'])
                delta = -(zs*s['p'])@s['decoder'] if condition else torch.zeros_like(x)
                if site in wsites:
                    zt = targets[site].encode(x.detach())[..., candidates[site]]
                    observed[site] = dict(source=zs[..., s['w']].detach(), target=zt.detach())
                    if mode == 'source':
                        delta = delta-alpha*(zs*s['w'])@s['decoder']
                    else:
                        delta = delta-(zt*selected_masks[site])@decoders[site]
                    if gradient:
                        leaf = torch.zeros_like(x, requires_grad=True)
                        leaves[site] = leaf
                        delta = delta+leaf
                updated = x+delta
                if site == 'resid_4':
                    pooled = (updated*mask[..., None]).sum(1)/mask.sum(1)[:, None]
                return (updated, *out[1:]) if isinstance(out, tuple) else updated
            return apply

        for site in sites:
            hooks.append(site_module(model, site).register_forward_hook(hook(site)))

        def forward(ids):
            nonlocal observed, leaves
            if time.perf_counter()-work.wall_start > c['budget_seconds']:
                raise TimeoutError('Conditional member selection budget exceeded')
            observed, leaves = {}, {}
            with torch.set_grad_enabled(gradient):
                model.gpt_neox(ids, attention_mask=mask, use_cache=False)
                logits = (pooled@pw.T+pb).flatten()
            work.sequence_forwards += len(ids)
            work.token_forwards += ids.numel()
            assert torch.isfinite(logits).all()
            return logits

        def batches(split):
            selected = sorted([i for i, r in enumerate(rows) if r['selection_split'] == split],
                key=lambda i: len(rows[i]['tokens']))
            offset = 0
            while offset < len(selected):
                ix = selected[offset:offset+c['batch_size']]
                while len(ix) > 1 and len(ix)*max(len(rows[i]['tokens']) for i in ix) > c['token_budget']:
                    ix = ix[:-1]
                length = max(len(rows[i]['tokens']) for i in ix)
                ids = torch.zeros((len(ix), length), device=work.device, dtype=torch.long)
                attention = torch.zeros_like(ids)
                for j, i in enumerate(ix):
                    tokens = rows[i]['tokens']
                    ids[j, :len(tokens)] = torch.tensor(tokens, device=work.device)
                    attention[j, :len(tokens)] = 1
                yield ix, ids, attention
                offset += len(ix)

        moments = {method: {site: [torch.zeros((len(candidates[site]), len(candidates[site])), dtype=torch.float64),
            torch.zeros(len(candidates[site]), dtype=torch.float64), 0] for site in wsites}
            for method in ['native_natural', 'native_operation']}
        fit_responses = {site: [] for site in wsites}
        fit_targets = {site: [] for site in wsites}
        fit_references = []
        arrays = work.run/'fit_arrays'
        arrays.mkdir()
        alphas, weights = c['path_alphas'], c['path_weights']
        assert alphas[0] == 0 and alphas[-1] == 1 and len(alphas) == len(weights) and abs(sum(weights)-1) < 1e-12
        for batch_index, (ix, ids, attention) in enumerate(batches('fit')):
            mask = attention
            valid = mask.bool()
            payload = dict(row_indices=np.array(ix), row_ids=np.array([rows[i]['row_id'] for i in ix]),
                document_sha256=np.array([rows[i]['document_sha256'] for i in ix]),
                token_offsets=np.cumsum([0]+[len(rows[i]['tokens']) for i in ix]), token_ids=ids[valid].cpu().numpy())
            common = {}
            mode, gradient, alpha = 'source', False, 0.
            for condition in [0, 1]:
                values = forward(ids).detach().cpu().numpy()
                common[condition] = {site: {key: value[valid].cpu().double() for key, value in observed[site].items()} for site in wsites}
                payload[f'common_c{condition}__logits'] = values
                for site in wsites:
                    for key, value in common[condition][site].items():
                        payload[f'common_c{condition}__{site}__{key}'] = value.numpy().astype(np.float32)
            for site in wsites:
                d = decoders[site].cpu().double()
                ds = source_decoders[site].cpu().double()
                for method in ['native_natural', 'native_operation']:
                    z = common[0][site]['target']
                    source_z = common[0][site]['source']
                    if method == 'native_operation':
                        z = z-common[1][site]['target']
                        source_z = source_z-common[1][site]['source']
                    q, b = contribution_moments(z, d, source_z@ds)
                    moments[method][site][0] += q*len(z)
                    moments[method][site][1] += b*len(z)
                    moments[method][site][2] += len(z)
            for condition in [0, 1]:
                path_projection = {site: torch.zeros((len(ix), len(candidates[site])), device=work.device) for site in wsites}
                path_source = {site: torch.zeros(len(ix), device=work.device) for site in wsites}
                endpoint = []
                for ai, (alpha, weight) in enumerate(zip(alphas, weights)):
                    gradient = True
                    logits = forward(ids)
                    grads = torch.autograd.grad(logits.sum(), [leaves[site] for site in wsites])
                    backward_sequences += len(ix)
                    endpoint.append(logits.detach().cpu().numpy())
                    for site, g in zip(wsites, grads):
                        assert torch.isfinite(g).all()
                        gt = g.detach()@decoders[site].T
                        gs = g.detach()@source_decoders[site].T
                        zt, zs = observed[site]['target'], observed[site]['source']
                        path_projection[site] += weight*(-zt*gt*mask[..., None]).sum(1)
                        path_source[site] += weight*(-zs*gs*mask[..., None]).sum((1, 2))
                        prefix = f'path_c{condition}_a{ai}__{site}'
                        payload[prefix+'__target_code'] = zt[valid].cpu().numpy()
                        payload[prefix+'__source_code'] = zs[valid].cpu().numpy()
                        payload[prefix+'__target_gradient_projection'] = gt[valid].cpu().numpy()
                        payload[prefix+'__source_gradient_projection'] = gs[valid].cpu().numpy()
                    del logits, grads
                endpoint = np.stack(endpoint, axis=1)
                payload[f'path_c{condition}__logits'] = endpoint
                actual = endpoint[:, -1]-endpoint[:, 0]
                fit_references.append(actual)
                approximate = sum(path_source.values()).detach().cpu().numpy()
                for j, i in enumerate(ix):
                    work.record(kind='path_response', task='profession', row_id=rows[i]['row_id'],
                        component=rows[i]['document_sha256'], method='source_raw_path', operation=f'condition_{condition}',
                        seed=c['target_seed'], target_seed=c['target_seed'], split='fit',
                        source_effect=float(actual[j]), integrated_source_effect=float(approximate[j]),
                        clean_logit=float(endpoint[j, 0]), edited_logit=float(endpoint[j, -1]))
                for site in wsites:
                    a = path_projection[site].detach().cpu().double()
                    y = path_source[site].detach().cpu().double()
                    fit_responses[site].append(a)
                    fit_targets[site].append(y)
                    payload[f'integrated_c{condition}__{site}__target'] = a.numpy()
                    payload[f'integrated_c{condition}__{site}__source'] = y.numpy()
            np.savez_compressed(arrays/f'batch_{batch_index:03d}.npz', **payload)
            work.progress('COMMON_STATES_AND_PATHS', fit_documents_completed=sum(len(v) for v in fit_references)//2,
                fit_documents_total=4*c['fit_per_group'], backward_sequences=backward_sequences)

        masks, exported, details = {}, {}, {}
        for method in methods:
            masks[method] = {}
            details[method] = {}
            for site in wsites:
                budget = c['members_by_site'][site]
                if method == 'native_support':
                    value, diagnostic = native_scores[site], dict(source='R59 native W membership row sum')
                elif method in moments:
                    q, b, count = moments[method][site]
                    q, b = q/count, b/count
                    value, diagnostic = bounded_quadratic(q, b, budget)
                else:
                    a = torch.cat(fit_responses[site])
                    y = torch.cat(fit_targets[site])
                    q, b = a.T@a/len(a), a.T@y/len(a)
                    value, diagnostic = bounded_quadratic(q, b, budget)
                    exported[method+'__'+site+'__design'] = a.numpy()
                    exported[method+'__'+site+'__response'] = y.numpy()
                support = torch.argsort(value, descending=True, stable=True)[:budget]
                binary = torch.zeros(len(value), dtype=torch.float32)
                binary[support] = 1
                initial_binary = binary.clone()
                binary_diagnostic = None
                if method != 'native_support':
                    exported[method+'__'+site+'__Q'] = q.numpy()
                    exported[method+'__'+site+'__rhs'] = b.numpy()
                    if c.get('binary_single_swap', False):
                        optimized, binary_diagnostic = binary_single_swap(q.numpy(), b.numpy(), binary.numpy(),
                            candidates[site].cpu().numpy())
                        binary = torch.tensor(optimized, dtype=torch.float32)
                        support = binary.nonzero().flatten()
                        binary_diagnostic['continuous_fit_objective'] = float(value@q@value-2*b@value)
                masks[method][site] = binary.to(work.device)
                ids_selected = candidates[site].cpu()[support].tolist()
                details[method][site] = dict(candidate_ids=candidates[site].cpu().tolist(), selected_ids=ids_selected,
                    members=budget, continuous_fit=value.tolist(), initial_binary_mask=initial_binary.tolist(),
                    binary_mask=binary.tolist(), diagnostics=diagnostic, binary_optimization=binary_diagnostic)
                exported[method+'__'+site+'__continuous'] = value.numpy()
                exported[method+'__'+site+'__initial_binary'] = initial_binary.numpy()
                exported[method+'__'+site+'__binary'] = binary.numpy()
                assert int(binary.sum()) == budget
        np.savez_compressed(work.run/'selection_parameters.npz', **exported)
        write(work.run/'selected_members.json', details)
        write(work.run/'selection_protocol.json', dict(methods=methods, conditions=['none', 'source_P'],
            shared_candidates=True, members_by_site=c['members_by_site'], alphas=alphas, weights=weights,
            endpoint='Fixed original source profession logit; edited logit minus same-condition baseline',
            operation='incoming x minus source P contribution when present minus complete target selected-member contribution; same incoming encoding and preserved residual',
            fit_identity='First 16 document hashes in each exposed development label/gender cell; evaluation uses next16',
            common_states='All source and target codes observed at same incoming hook tensor on source none/P trajectories',
            gradient='Derivative with respect to independent additive hidden leaf after the local source P and alpha-W update; downstream program remains differentiable',
            direct_comparator='Source and raw share identical hidden path information and candidate decoder projections; one combined source_raw_path selector',
            path_objective='Per-site integrated source W attribution across both conditions, equal document-condition weight',
            selection=('Continuous fit initializes a fixed-cardinality binary support; deterministic improving single exchanges optimize the same fit objective; all target executions use binary masks'
                if c.get('binary_single_swap', False) else 'Continuous box/cardinality fit is used only to rank support; all actual target executions use binary masks'),
            target_validation_during_fit=0, backward_sequences=backward_sequences))
        work.progress('SELECTION_FROZEN', methods=methods, total_members=22)

        output = work.run/'evaluation_arrays'
        output.mkdir()
        collected = []
        gradient = False
        for batch_index, (ix, ids, attention) in enumerate(batches('evaluation')):
            mask = attention
            payload = dict(row_indices=np.array(ix), row_ids=np.array([rows[i]['row_id'] for i in ix]),
                document_sha256=np.array([rows[i]['document_sha256'] for i in ix]))
            for condition in [0, 1]:
                mode, alpha = 'source', 0.
                baseline = forward(ids).detach().cpu().numpy()
                alpha = 1.
                reference = forward(ids).detach().cpu().numpy()
                payload[f'c{condition}__baseline'] = baseline
                payload[f'c{condition}__source'] = reference
                for method in ['source']+methods:
                    if method == 'source':
                        values = reference
                    else:
                        mode = 'target'
                        selected_masks = masks[method]
                        values = forward(ids).detach().cpu().numpy()
                    payload[f'c{condition}__{method}__logits'] = values
                    payload[f'c{condition}__{method}__pooled512'] = pooled.detach().cpu().numpy()
                    for j, i in enumerate(ix):
                        r = rows[i]
                        record = dict(kind='conditional_member_effect', task='profession', row_id=r['row_id'],
                            component=r['document_sha256'], split='evaluation', method=method,
                            operation=f'condition_{condition}', condition=condition, seed=c['target_seed'], target_seed=c['target_seed'],
                            label=r['label'], gender=r['gender'], baseline_logit=float(baseline[j]),
                            source_logit=float(reference[j]), logit=float(values[j]), prediction=int(values[j]>0),
                            source_effect=float(reference[j]-baseline[j]), actual_effect=float(values[j]-baseline[j]),
                            correct=bool((values[j]>0)==r['label']), source_correct=bool((reference[j]>0)==r['label']))
                        work.record(**record)
                        collected.append(record)
            np.savez_compressed(output/f'batch_{batch_index:03d}.npz', **payload)
            work.progress('MEMBER_EVALUATION', completed_documents=len(collected)//(2*(len(methods)+1)),
                total_documents=4*c['evaluation_per_group'])
        source_values = np.array([r['source_effect'] for r in collected if r['method']=='source'])
        scale = float(np.sqrt(np.mean(source_values**2)))
        assert scale > 0
        summary = dict(source_rms_scale=scale, methods={}, fit_documents=4*c['fit_per_group'],
            evaluation_documents=4*c['evaluation_per_group'], backward_sequences=backward_sequences)
        for method in ['source']+methods:
            rr = [r for r in collected if r['method']==method]
            summary['methods'][method] = {}
            for key, subset in [('joint', rr)]+[(f'condition_{v}', [r for r in rr if r['condition']==v]) for v in [0, 1]]:
                error_values = np.array([r['actual_effect']-r['source_effect'] for r in subset])
                group_acc = [float(np.mean([r['correct'] for r in subset if r['label']==y and r['gender']==g]))
                    for y in [0, 1] for g in [0, 1]]
                summary['methods'][method][key] = dict(rmse=float(np.sqrt(np.mean(error_values**2))),
                    nrmse=float(np.sqrt(np.mean(error_values**2))/scale), accuracy=float(np.mean([r['correct'] for r in subset])),
                    worst_group_accuracy=min(group_acc), label_gender_group_accuracy=group_acc,
                    actual_effect_rms=float(np.sqrt(np.mean([r['actual_effect']**2 for r in subset]))),
                    source_effect_rms=float(np.sqrt(np.mean([r['source_effect']**2 for r in subset]))))
        write(work.run/'MEMBER_RESULTS.json', summary)
        work.checks.update(common_source_states=True, same_candidate_pool=True, exact_site_member_budgets=True,
            binary_complete_member_deletion=True, source_target_frozen=True, fit_evaluation_documents_disjoint=True,
            selection_saved_before_target_evaluation=True, source_raw_share_path_information=True)
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr)
    for handle in hooks:
        handle.remove()
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
