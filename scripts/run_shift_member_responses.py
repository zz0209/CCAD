from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time
import traceback

import numpy as np

from run_causalgym_multisite import MultisiteWork, write
from run_shift_explanation import source_groups
from train_shift_dictionaries import site_module
from ccad.artifacts import sha256


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    work = MultisiteWork(c, args.config, ['scripts/run_shift_member_responses.py',
        'scripts/run_shift_explanation.py', 'scripts/train_shift_dictionaries.py',
        'scripts/run_causalgym_multisite.py', 'scripts/run_r011s1_raw_hook_asset.py', 'src/ccad/artifacts.py'])
    hooks, error = [], None
    try:
        import torch
        import transformers
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        work.torch, work.device = torch, torch.device(c['device'])
        torch.cuda.set_device(work.device)
        torch.cuda.reset_peak_memory_stats(work.device)
        sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        work.environment = dict(python=sys.executable, python_version=platform.python_version(),
            torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__, cpu_threads=2)
        source_path = work.checked(c['source_manifest'], 'Published source feature IDs', 'MIT')
        source = json.loads(source_path.read_text())
        groups, _ = source_groups(work.checked(c['notebook'], 'Published annotations', 'MIT'), source['members'])
        parameter_path = work.checked(c['source_parameters'], 'Published source parameters', 'MIT')
        bank = np.load(parameter_path)
        relation_path = work.checked(Path(c['relation_run'])/'relation.npz', 'Frozen R59 candidates')
        relation = np.load(relation_path)
        sites = list(source['members'])
        wsites = [site for site in sites if site in groups['associated_words']]
        parameters, targets, candidates, target_hashes = {}, {}, {}, {}
        for site in sites:
            s = {key: torch.tensor(bank[site+'__'+key], device=work.device)
                 for key in ['encoder', 'encoder_bias', 'decoder', 'center']}
            s['p'] = torch.tensor([i in groups['pronouns'].get(site, []) for i in source['members'][site]], device=work.device)
            s['w'] = torch.tensor([i in groups['associated_words'].get(site, []) for i in source['members'][site]], device=work.device)
            assert not bool((s['p'] & s['w']).any())
            parameters[site] = s
            if site in wsites:
                path = work.checked(Path(c['target_directory'])/f'{site}_seed{c["target_seed"]}.pt', 'Original target SAE')
                state = torch.load(path, map_location=work.device, weights_only=True)
                target = AutoEncoderTopK(512, state['encoder.weight'].shape[0], int(state['k'])).to(work.device)
                target.load_state_dict(state)
                target.eval().requires_grad_(False)
                targets[site] = target
                target_hashes[site] = sha256(path)
                candidates[site] = np.asarray(relation[site+'__candidates'], dtype=np.int64)
                assert len(np.unique(candidates[site])) == len(candidates[site])
        probe_path = work.checked(Path(c['frozen_source_run'])/'probe.npz', 'Fixed original profession head')
        probe = np.load(probe_path)
        pw, pb = torch.tensor(probe['weight'], device=work.device), torch.tensor(probe['bias'], device=work.device)
        panel = json.loads(work.checked(c['evaluation_panel'], 'Fixed document panel').read_text())
        rows = []
        if c.get('evaluation_only'):
            assert c.get('selected_groups_file'), 'Evaluation-only panels require frozen selected groups'
            rows = [dict(r, selection_split='evaluation') for r in panel['rows']]
            assert rows and {(r['label'], r['gender']) for r in rows} == {(0, 0), (0, 1), (1, 0), (1, 1)}
        else:
            for label in [0, 1]:
                for gender in [0, 1]:
                    cell = sorted([r for r in panel['rows'] if r['split'] == c['evaluation_split']
                        and r['label'] == label and r['gender'] == gender], key=lambda r: r['document_sha256'])[:32]
                    assert len(cell) == 32
                    for split, selected in [('fit', cell[:c['fit_per_group']]),
                            ('evaluation', cell[16:16+c['evaluation_per_group']])]:
                        rows.extend([dict(r, selection_split=split) for r in selected])
        assert len({r['document_sha256'] for r in rows}) == len(rows)
        metadata = dict(row_ids=np.array([r['row_id'] for r in rows]),
            document_sha256=np.array([r['document_sha256'] for r in rows]),
            splits=np.array([r['selection_split'] for r in rows]), labels=np.array([r['label'] for r in rows]),
            genders=np.array([r['gender'] for r in rows]), token_counts=np.array([len(r['tokens']) for r in rows]))
        common_identity = dict(rows=[dict(document_sha256=r['document_sha256'], tokens=r['tokens'],
            split=r['selection_split'], label=r['label'], gender=r['gender']) for r in rows],
            model_revision=c['model_revision'], source_parameters_sha256=sha256(parameter_path),
            source_manifest_sha256=sha256(source_path), probe_sha256=sha256(probe_path),
            conditions=['none', 'dynamic_source_P'], operation='complete_single_member_deletion_at_own_site')
        source_identity = identity_hash(common_identity)
        target_identity = identity_hash(dict(common=source_identity, target_seed=c['target_seed'],
            target_checkpoint_hashes=target_hashes, relation_sha256=sha256(relation_path)))
        write(work.run/'response_membership.json', dict(rows=rows,
            identity='Frozen new-document evaluation' if c.get('evaluation_only') else 'Historically exposed development documents'))
        write(work.run/'response_identity.json', dict(source=source_identity, target=target_identity,
            common=common_identity, target_checkpoint_hashes=target_hashes, relation_sha256=sha256(relation_path)))
        for name in ['config.json', 'tokenizer.json', 'model.safetensors']:
            work.checked(Path(c['model_local_dir'])/name, 'Pinned Pythia70M', 'Apache-2.0')
        model = transformers.AutoModelForCausalLM.from_pretrained(c['model_local_dir'], local_files_only=True,
            dtype=torch.float32, attn_implementation='eager').eval().to(work.device)
        model.requires_grad_(False)
        model.config.use_cache = False
        condition, family, active_site, operation, alpha = 0, 'source', None, 'none', 0.
        member_index, mask, pooled, active_stats = None, None, None, None
        group_members = {}
        path_models, path_candidates, path_observed, leaves = {}, {}, {}, {}
        backward_sequences = 0

        def hook(site):
            def apply(module, inputs, out):
                nonlocal pooled, active_stats
                x = out[0] if isinstance(out, tuple) else out
                s = parameters[site]
                zs = torch.relu((x-s['center'])@s['encoder'].T+s['encoder_bias'])
                delta = -(zs*s['p'])@s['decoder'] if condition else torch.zeros_like(x)
                if operation in ['whole_W', 'path']:
                    delta = delta-(alpha if operation == 'path' else 1.)*(zs*s['w'])@s['decoder']
                    if operation == 'path' and site in wsites:
                        for seed, codecs in path_models.items():
                            path_observed[seed, site] = codecs[site].encode(x.detach())[..., path_candidates[seed][site]]
                        leaf = torch.zeros_like(x, requires_grad=True)
                        leaves[site] = leaf
                        delta = delta+leaf
                elif operation == 'member' and site == active_site:
                    if family == 'source':
                        z, decoder = zs, s['decoder']
                    else:
                        z, decoder = targets[site].encode(x), targets[site].decoder.weight.T
                    value = z.gather(2, member_index[:, None, None].expand(-1, z.shape[1], 1)).squeeze(-1)
                    d = decoder[member_index]
                    delta = delta-value[..., None]*d[:, None, :]
                    square = (value.square()*mask).sum(1)
                    active_stats = dict(code_sum=(value*mask).sum(1), code_sq_sum=square,
                        active_tokens=((value>0)*mask).sum(1), delta_l2=square.sqrt()*d.norm(dim=1))
                elif operation == 'group' and site in group_members:
                    selected = group_members[site]
                    z = targets[site].encode(x)[..., selected]
                    delta = delta-z@targets[site].decoder.weight.T[selected]
                updated = x+delta
                if site == 'resid_4':
                    pooled = (updated*mask[..., None]).sum(1)/mask.sum(1)[:, None]
                return (updated, *out[1:]) if isinstance(out, tuple) else updated
            return apply

        for site in sites:
            hooks.append(site_module(model, site).register_forward_hook(hook(site)))

        def forward(ids):
            if time.perf_counter()-work.wall_start > c['budget_seconds']:
                raise TimeoutError('Finite member response collection budget exceeded')
            with torch.set_grad_enabled(operation == 'path'):
                model.gpt_neox(ids, attention_mask=mask, use_cache=False)
                logits = (pooled@pw.T+pb).flatten()
            work.sequence_forwards += len(ids)
            work.token_forwards += ids.numel()
            assert torch.isfinite(logits).all()
            if torch.cuda.max_memory_allocated(work.device) > c.get('maximum_cuda_bytes', 4*1024**3):
                raise RuntimeError('Configured CUDA allocation budget exceeded')
            return logits if operation == 'path' else logits.detach().cpu().numpy()

        def batches(members):
            ordered = sorted(range(len(rows)), key=lambda i: len(rows[i]['tokens']))
            offset = 0
            while offset < len(ordered):
                ix = ordered[offset:offset+c['batch_size']]
                while len(ix)>1 and members*len(ix)*max(len(rows[i]['tokens']) for i in ix)>c['token_budget']:
                    ix = ix[:-1]
                length = max(len(rows[i]['tokens']) for i in ix)
                assert members*len(ix)*length <= c['token_budget'], 'Reduce member_batch_size or increase token_budget'
                ids = torch.zeros((len(ix), length), device=work.device, dtype=torch.long)
                attention = torch.zeros_like(ids)
                for j, i in enumerate(ix):
                    tokens = rows[i]['tokens']
                    ids[j, :len(tokens)] = torch.tensor(tokens, device=work.device)
                    attention[j, :len(tokens)] = 1
                yield ix, ids, attention
                offset += len(ix)

        source_dir, target_dir = work.run/'source_bank', work.run/'target_bank'
        source_dir.mkdir()
        target_dir.mkdir()
        index = dict(source_reference=None, source_blocks=[], target_blocks=[],
            axes=['document', 'condition', 'member'], conditions=['none', 'source_P'],
            target_seed=c['target_seed'], candidate_ids={site: values.tolist() for site, values in candidates.items()},
            path_projections={}, target_path_projection=None)
        previous = json.loads((Path(c['resume_run'])/'response_index.json').read_text()) if c.get('resume_run') else None
        shared = json.loads((Path(c['source_run'])/'response_index.json').read_text()) if c.get('source_run') else None

        def verify(path, identity):
            with np.load(path) as saved:
                assert str(saved['status']) == 'PASS' and str(saved['identity']) == identity
                np.testing.assert_array_equal(saved['document_sha256'], metadata['document_sha256'])
                np.testing.assert_array_equal(saved['splits'], metadata['splits'])
            work.checked(path, 'Completed immutable response block')

        reference_path = (shared or previous or {}).get('source_reference')
        if reference_path:
            verify(reference_path, source_identity)
            with np.load(reference_path) as reference:
                baseline = reference['baseline_logits'].copy()
                whole = reference['whole_W_logits'].copy()
        else:
            baseline = np.empty((len(rows), 2), dtype=np.float32)
            whole = np.empty_like(baseline)
            for condition in [0, 1]:
                for ix, ids, attention in batches(1):
                    mask = attention
                    operation = 'none'
                    baseline[ix, condition] = forward(ids)
                    operation = 'whole_W'
                    whole[ix, condition] = forward(ids)
                work.progress('SOURCE_REFERENCE', condition=condition, documents=len(rows))
            reference_path = source_dir/'reference.npz'
            np.savez_compressed(reference_path, **metadata, baseline_logits=baseline, whole_W_logits=whole,
                whole_W_effects=whole-baseline, status=np.array('PASS'), identity=np.array(source_identity))
        index['source_reference'] = str(Path(reference_path).resolve())
        write(work.run/'response_index.json', index)
        if c.get('selected_groups_file'):
            selection_path = work.checked(c['selected_groups_file'], 'Frozen analysis-selected complete target groups')
            selections = json.loads(selection_path.read_text())
            assert selections
            group_dir = work.run/'group_bank'
            group_dir.mkdir()
            index['source_blocks'] = shared['source_blocks'] if shared else []
            index['group_blocks'] = []
            report = {}
            source_effect = whole-baseline
            scale = float(np.sqrt(np.mean(source_effect**2)))
            assert scale > 0
            for gi, (method, selected) in enumerate(selections.items()):
                assert set(selected) == set(wsites)
                group_members = {}
                for site in wsites:
                    members = selected[site]
                    assert len(members) == len(set(members)) == 2*len(groups['associated_words'][site])
                    assert set(members) <= set(candidates[site].tolist())
                    group_members[site] = torch.tensor(members, device=work.device, dtype=torch.long)
                assert sum(len(v) for v in group_members.values()) == 22
                operation = 'group'
                values = np.empty((len(rows), 2), np.float32)
                hidden = np.empty((len(rows), 2, 512), np.float32)
                for condition in [0, 1]:
                    for ix, ids, attention in batches(1):
                        mask = attention
                        values[ix, condition] = forward(ids)
                        hidden[ix, condition] = pooled.detach().cpu().numpy()
                effects = values-baseline
                path = group_dir/f'group_{gi:03d}.npz'
                np.savez_compressed(path, **metadata, logits=values, effects=effects, pooled512=hidden,
                    baseline_logits=baseline, whole_W_logits=whole, whole_W_effects=source_effect,
                    method=np.array(method), selected_members=np.array(json.dumps(selected, sort_keys=True)),
                    status=np.array('PASS'), identity=np.array(target_identity))
                index['group_blocks'].append(str(path.resolve()))
                write(work.run/'response_index.json', index)
                report[method] = {}
                for split in ['fit', 'evaluation']:
                    chosen = metadata['splits'] == split
                    if not chosen.any():
                        continue
                    # 拟合与评价分别汇总，目标比较使用各自共同source尺度。
                    reference_scale = float(np.sqrt(np.mean(source_effect[chosen]**2)))
                    error_values = effects[chosen]-source_effect[chosen]
                    report[method][split] = dict(nrmse=float(np.sqrt(np.mean(error_values**2))/reference_scale),
                        source_rms_scale=reference_scale, conditions={})
                    for condition in [0, 1]:
                        correct = (values[:, condition]>0) == metadata['labels']
                        group_accuracy = [float(correct[chosen & (metadata['labels']==y) & (metadata['genders']==g)].mean())
                            for y in [0, 1] for g in [0, 1]]
                        report[method][split]['conditions'][str(condition)] = dict(
                            rmse=float(np.sqrt(np.mean(error_values[:, condition]**2))),
                            actual_effect_rms=float(np.sqrt(np.mean(effects[chosen, condition]**2))),
                            source_effect_rms=float(np.sqrt(np.mean(source_effect[chosen, condition]**2))),
                            accuracy=float(correct[chosen].mean()), worst_group_accuracy=min(group_accuracy))
                for i, row in enumerate(rows):
                    for condition in [0, 1]:
                        work.record(kind='group_response', task='profession', row_id=row['row_id'],
                            component=row['document_sha256'], split=row['selection_split'], method=method,
                            operation=f'condition_{condition}', condition=condition, seed=c['target_seed'], target_seed=c['target_seed'],
                            label=row['label'], gender=row['gender'], baseline_logit=float(baseline[i, condition]),
                            source_logit=float(whole[i, condition]), logit=float(values[i, condition]),
                            source_effect=float(source_effect[i, condition]), actual_effect=float(effects[i, condition]),
                            prediction=int(values[i, condition]>0), correct=bool((values[i, condition]>0)==row['label']))
                work.progress('COMPLETE_MEMBER_GROUPS', method=method, completed_groups=gi+1,
                    total_groups=len(selections), documents=len(rows), members=22)
            write(work.run/'GROUP_RESULTS.json', report)
            write(work.run/'response_cost.json', dict(backward_sequences=0, source_reused=bool(shared or previous),
                source_path_target_seeds=[], mode='selected_complete_groups'))
            work.checks.update(frozen_selected_groups=True, exact_site_member_budgets=True,
                binary_complete_member_deletion=True, source_bank_identity=True, fixed_head=True)
            for handle in hooks:
                handle.remove()
            return work.finish(None)
        if c.get('path_targets') or shared:
            inherited_paths = (shared or previous or {}).get('path_projections', {})
            requested = c.get('path_targets', {})
            path_hashes = {}
            for seed, spec in requested.items():
                if seed in inherited_paths:
                    verify(inherited_paths[seed], source_identity)
                    index['path_projections'][seed] = inherited_paths[seed]
                    continue
                saved_relation = np.load(work.checked(Path(spec['relation_run'])/'relation.npz', 'Path comparator candidate pool'))
                path_models[seed], path_candidates[seed], path_hashes[seed] = {}, {}, {}
                for site in wsites:
                    indices = saved_relation[site+'__candidates']
                    if c.get('member_limit_per_site'):
                        indices = indices[:c['member_limit_per_site']]
                    path_candidates[seed][site] = torch.tensor(indices, device=work.device, dtype=torch.long)
                    path = work.checked(Path(spec['target_directory'])/f'{site}_seed{seed}.pt', 'Path comparator target SAE')
                    path_hashes[seed][site] = sha256(path)
                    if int(seed) == c['target_seed']:
                        path_models[seed][site] = targets[site]
                    else:
                        state = torch.load(path, map_location=work.device, weights_only=True)
                        codec = AutoEncoderTopK(512, state['encoder.weight'].shape[0], int(state['k'])).to(work.device)
                        codec.load_state_dict(state)
                        path_models[seed][site] = codec.eval().requires_grad_(False)
            if shared:
                assert str(c['target_seed']) in inherited_paths
                index['path_projections'] = inherited_paths
            if path_models:
                alphas, weights = c['path_alphas'], c['path_weights']
                assert alphas == [0., .25, .5, .75, 1.] and weights == [.125, .25, .25, .25, .125]
                projections = {seed: np.zeros((len(rows), 2, sum(len(v) for v in path_candidates[seed].values())), np.float32)
                    for seed in path_models}
                path_logits = np.empty((len(rows), 2, len(alphas)), np.float32)
                operation = 'path'
                for condition in [0, 1]:
                    completed = 0
                    for ix, ids, attention in batches(1):
                        mask = attention
                        for ai, (alpha, weight) in enumerate(zip(alphas, weights)):
                            logits = forward(ids)
                            gradients = torch.autograd.grad(logits.sum(), [leaves[site] for site in wsites])
                            backward_sequences += len(ix)
                            path_logits[ix, condition, ai] = logits.detach().cpu().numpy()
                            for seed in path_models:
                                offset = 0
                                for site, g in zip(wsites, gradients):
                                    z = path_observed[seed, site]
                                    decoder = path_models[seed][site].decoder.weight.T[path_candidates[seed][site]]
                                    value = weight*(-z*(g.detach()@decoder.T)*mask[..., None]).sum(1)
                                    projections[seed][ix, condition, offset:offset+z.shape[-1]] += value.cpu().numpy()
                                    offset += z.shape[-1]
                            del logits, gradients
                        completed += len(ix)
                        work.progress('SHARED_SOURCE_PATH', condition=condition, completed_documents=completed,
                            total_documents=len(rows), target_seeds=list(path_models), backward_sequences=backward_sequences)
                for seed, projection in projections.items():
                    path = source_dir/f'path_seed{seed}.npz'
                    np.savez_compressed(path, **metadata, path_projection=projection, path_logits=path_logits,
                        member_ids=np.concatenate([path_candidates[seed][site].cpu().numpy() for site in wsites]),
                        sites=np.concatenate([np.repeat(site, len(path_candidates[seed][site])) for site in wsites]),
                        target_checkpoint_hashes=np.array(json.dumps(path_hashes[seed], sort_keys=True)),
                        status=np.array('PASS'), identity=np.array(source_identity))
                    index['path_projections'][seed] = str(path.resolve())
                    write(work.run/'response_index.json', index)
                path_models.clear()
                path_observed.clear()
                leaves.clear()
            path = index['path_projections'][str(c['target_seed'])]
            verify(path, source_identity)
            with np.load(path) as saved:
                assert json.loads(str(saved['target_checkpoint_hashes'])) == target_hashes
            index['target_path_projection'] = path
            write(work.run/'response_index.json', index)
        source_expected = {}
        for site in wsites:
            source_expected[site] = np.array(groups['associated_words'][site], dtype=np.int64)
        if shared:
            seen = {site: [] for site in wsites}
            for path in shared['source_blocks']:
                verify(path, source_identity)
                with np.load(path) as saved:
                    seen[str(saved['site'])].extend(saved['member_ids'].tolist())
                index['source_blocks'].append(str(Path(path).resolve()))
            assert all(seen[site] == source_expected[site].tolist() for site in wsites)
        operation = 'member'
        block_candidates = {Path(path).name: path for key in ['source_blocks', 'target_blocks']
            for path in (previous or {}).get(key, [])}
        for family in (['target'] if shared else ['source', 'target']):
            for active_site in wsites:
                values = source_expected[active_site] if family == 'source' else candidates[active_site]
                if family == 'target' and c.get('member_limit_per_site'):
                    values = values[:c['member_limit_per_site']]
                # 成员批量还受最长文档和真实词元预算约束。
                width = min(c['member_batch_size'], max(1, c['token_budget']//max(metadata['token_counts'])))
                for start in range(0, len(values), width):
                    members = values[start:start+width]
                    stop = start+len(members)
                    name = f'{family}__{active_site}__{start:04d}_{stop:04d}.npz'
                    identity = source_identity if family == 'source' else target_identity
                    old = block_candidates.get(name)
                    if old:
                        verify(old, identity)
                        with np.load(old) as saved:
                            np.testing.assert_array_equal(saved['member_ids'], members)
                        path = Path(old)
                        reused = True
                    else:
                        shape = (len(rows), 2, len(members))
                        result = {key: np.empty(shape, dtype=np.float32)
                            for key in ['logits', 'code_sum', 'code_sq_sum', 'active_tokens', 'delta_l2']}
                        local = ([source['members'][active_site].index(int(i)) for i in members]
                            if family == 'source' else members.tolist())
                        for condition in [0, 1]:
                            for ix, ids, attention in batches(len(members)):
                                mask = attention.repeat(len(members), 1)
                                member_index = torch.tensor(local, device=work.device).repeat_interleave(len(ix))
                                logits = forward(ids.repeat(len(members), 1)).reshape(len(members), len(ix)).T
                                result['logits'][ix, condition, :] = logits
                                for key, value in active_stats.items():
                                    result[key][ix, condition, :] = value.cpu().numpy().reshape(len(members), len(ix)).T
                        decoder = (parameters[active_site]['decoder'][local] if family == 'source'
                            else targets[active_site].decoder.weight.T[local]).detach().cpu().numpy()
                        path = (source_dir if family == 'source' else target_dir)/name
                        assert not path.exists()
                        temporary = path.with_suffix('.partial.npz')
                        np.savez_compressed(temporary, **metadata, **result,
                            effects=result['logits']-baseline[..., None], member_ids=members, decoder=decoder,
                            site=np.array(active_site), family=np.array(family), status=np.array('PASS'), identity=np.array(identity))
                        temporary.replace(path)
                        reused = False
                    index[family+'_blocks'].append(str(path.resolve()))
                    write(work.run/'response_index.json', index)
                    work.record(kind='member_response_block', task=active_site, row_id=start, method=family,
                        component=name, mode=name, operation='both_conditions', seed=c['target_seed'], target_seed=c['target_seed'],
                        members=members.tolist(), documents=len(rows), path=str(path.resolve()), reused=reused)
                    work.progress('FINITE_MEMBER_RESPONSES', family=family, site=active_site,
                        completed_members=stop, site_members=len(values), documents=len(rows), reused=reused)
        write(work.run/'response_index.json', index)
        write(work.run/'response_cost.json', dict(backward_sequences=backward_sequences,
            source_reused=bool(shared), source_path_target_seeds=list(index['path_projections']),
            block_recovery='New run ID references completed PASS blocks from resume_run; original blocks are immutable'))
        work.checks.update(complete_response_blocks=True, same_document_conditions=True,
            complete_natural_member_deletion=True, source_P_dynamic=True, original_reconstruction_residual=True,
            source_bank_identity=True, fixed_head=True, disjoint_fit_evaluation=True)
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr)
    for handle in hooks:
        handle.remove()
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
