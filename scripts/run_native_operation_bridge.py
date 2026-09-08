"""Develop actual target-code edits; never mix oracle and deployment evidence."""
import argparse
import json
import traceback
from pathlib import Path

from composition_runtime import CompositionRun, ROOT, write, np
from run_component_correspondence import measure
from ccad.native_operation import writable_support, project_native, adaptive_writable_support


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    work = CompositionRun(args.config, 'scripts/run_native_operation_bridge.py', [
        'src/ccad/native_operation.py', 'scripts/run_component_correspondence.py'])
    error = None
    try:
        from safetensors import safe_open
        work.load()
        cfg = work.cfg
        assets = {}
        development = ROOT / cfg['development_material_run']
        panel = json.loads(work.checked(development / 'panel.json').read_text())
        devdonors = {f: np.asarray([p[f] for p in panel['pairs']]) for f in cfg['source_budgets']}
        for spec in cfg['sae_checkpoints']:
            seed = spec['seed']
            dev = np.load(work.checked(development / f'seed{seed}_codes.npz'))['number_z'].astype(np.float64)
            evaluation = np.load(work.checked(ROOT / cfg['evaluation_code_run'] / f'seed{seed}_codes.npz'))['number_z'].astype(np.float64)
            with safe_open(work.checked(Path(spec['path']) / 'sae.safetensors'), framework='numpy') as f:
                decoder = f.get_tensor('W_dec').astype(np.float64)
            assets[seed] = dict(dev=dev, evaluation=evaluation, decoder=decoder)
        fit_records, projection_records, geometry = [], [], []
        expected = 0
        ids = work.evaluation_ids
        for source_seed, target_seed in cfg['seed_pairs']:
            source, target = assets[source_seed], assets[target_seed]
            target_encoder = None
            if cfg.get('target_reencoding'):
                from sparsify import SparseCoder
                checkpoint = next(s['path'] for s in cfg['sae_checkpoints'] if s['seed'] == target_seed)
                target_encoder = SparseCoder.load_from_disk(checkpoint, device='cuda').eval()
                def encode_target(values):
                    with work.torch.no_grad():
                        acts, indices, _ = target_encoder.encode(work.torch.as_tensor(values, device='cuda', dtype=work.torch.float32))
                        return work.torch.zeros((len(values), target_encoder.num_latents), device='cuda').scatter_(1, indices, acts).cpu().numpy().astype(np.float64)
                recoded_base = encode_target(work.h[ids, 0])
                error_base = float(np.max(np.abs(recoded_base - target['evaluation'][ids])))
                work.checks[f'encoder_matches_cached_target_{target_seed}'] = error_base < 1e-4
                if not work.checks[f'encoder_matches_cached_target_{target_seed}']:
                    raise ValueError(f'Target encoder/cached state mismatch: {error_base}')
            for factor, source_budget in cfg['source_budgets'].items():
                maps = np.load(work.checked(ROOT / cfg['initial_map_run'] / f'maps_s{source_seed}_t{target_seed}_{factor}.npz'))
                source_members = maps['source_members']
                ds = maps['source_decoder']
                zs = source['evaluation'][:, source_members]
                devzs = source['dev'][:, source_members]
                changed = devzs[devdonors[factor]] - devzs
                covariance = (devzs.T @ devzs + changed.T @ changed) / len(devzs)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                signal = (np.sqrt(np.maximum(eigenvalues, 0))[:, None] * eigenvectors.T) @ ds
                eligibility = np.any(target['dev'] > 0, axis=0)
                chosen, selection = writable_support(target['decoder'], signal, max(cfg['write_budgets']), eligibility)
                reading = maps['aggregate_ols_members']
                input_data = dict(aggregate_ols=target['evaluation'], aggregate_full=target['evaluation'], aggregate_raw=work.h[:, 0])
                prediction = {name: np.maximum(values[:, maps[name+'_members']] @ maps[name+'_weights'], 0.) @ ds
                              for name, values in input_data.items()}
                exact = zs @ ds
                saved = dict(source_members=source_members, source_decoder=ds, read_members=reading, write_members=chosen)
                np.savez_compressed(work.run / f'native_s{source_seed}_t{target_seed}_{factor}.npz', **saved)
                fit_records.append(dict(source_seed=source_seed, target_seed=target_seed, factor=factor,
                    reading_members=reading.tolist(), writing_members=chosen.tolist(), selection=selection,
                    fit_split='Previously exposed development only; source-operation covariance plus target decoder, no evaluation outcomes'))
                for consumer in cfg['consumers']:
                    change = lambda a: a[work.donors[factor]] - a if consumer == 'contrast' else -a
                    actual = change(exact)
                    predicted = change(prediction['aggregate_ols'])
                    reference = measure(work, 'source', factor, 'all', consumer, 1., actual, None, source_seed, target_seed)
                    candidates = dict(noop=np.zeros_like(actual),
                                      source_readout=predicted,
                                      full_readout=change(prediction['aggregate_full']),
                                      raw_readout=change(prediction['aggregate_raw']),
                                      old_members_native=change(target['evaluation'][:, reading]) @ target['decoder'][reading])
                    for budget in cfg['write_budgets']:
                        members = chosen[:budget]
                        dt = target['decoder'][members]
                        zt = target['evaluation'][ids][:, members]
                        for name, desired, constrained in [
                            ('target_write_unbounded', predicted, False),
                            ('target_write_bounded', predicted, True),
                            ('source_oracle_write_bounded', actual, True),
                            ('full_readout_write_bounded', change(prediction['aggregate_full']), True)]:
                            u, diag = project_native(desired[ids], zt, dt,
                                constrained=constrained, max_steps=cfg['projection_steps'])
                            vectors = np.zeros_like(actual)
                            vectors[ids] = u @ dt
                            key = name + '_w' + str(budget)
                            candidates[key] = vectors
                            projection_records.append(dict(source_seed=source_seed, target_seed=target_seed, factor=factor,
                                consumer=consumer, method=key, write_budget=len(members),
                                read_budget='oracle' if name.startswith('source_oracle') else target['evaluation'].shape[1] if name.startswith('full_readout') else len(reading),
                                deployment_requires_source=name.startswith('source_oracle'), **diag))
                            if constrained:
                                work.checks[f'feasible_{source_seed}_{target_seed}_{factor}_{consumer}_{key}'] = diag['minimum_final_state'] >= -1e-9
                            if name == 'target_write_unbounded':
                                clipped = np.zeros_like(actual)
                                clipped[ids] = np.maximum(u, -zt) @ dt
                                candidates['target_write_clip_w'+str(budget)] = clipped
                    olddt = target['decoder'][reading]
                    oldu, olddiag = project_native(predicted[ids], target['evaluation'][ids][:, reading], olddt, max_steps=cfg['projection_steps'])
                    oldvectors = np.zeros_like(actual)
                    oldvectors[ids] = oldu @ olddt
                    candidates['read_members_bounded_projection'] = oldvectors
                    projection_records.append(dict(source_seed=source_seed, target_seed=target_seed, factor=factor, consumer=consumer,
                        method='read_members_bounded_projection', read_budget=len(reading), write_budget=len(reading), **olddiag))
                    if cfg.get('adaptive_write_budgets'):
                        for name, desired in [('adaptive_ols', predicted), ('adaptive_full', change(prediction['aggregate_full'])),
                                              ('adaptive_raw', change(prediction['aggregate_raw'])), ('adaptive_source_oracle', actual)]:
                            context_members = adaptive_writable_support(desired[ids], target['evaluation'][ids], target['decoder'],
                                max(cfg['adaptive_write_budgets']), eligibility)
                            context_outputs = {'row_ids': ids, 'members': context_members}
                            for budget in cfg['adaptive_write_budgets']:
                                vectors = np.zeros_like(actual)
                                allu = []
                                all_diagnostics = []
                                for local, rowid in enumerate(ids):
                                    jm = context_members[local, :budget]
                                    u, diag = project_native(desired[rowid:rowid+1], target['evaluation'][rowid:rowid+1, jm],
                                        target['decoder'][jm], max_steps=cfg['projection_steps'])
                                    vectors[rowid] = u[0] @ target['decoder'][jm]
                                    allu.append(u[0])
                                    all_diagnostics.append(diag)
                                key = name+'_w'+str(budget)
                                candidates[key] = vectors
                                context_outputs['increments_w'+str(budget)] = np.asarray(allu)
                                projection_records.append(dict(source_seed=source_seed, target_seed=target_seed, factor=factor,
                                    consumer=consumer, method=key, write_budget=budget, candidate_states_scanned=int(eligibility.sum()),
                                    source_dependent_oracle=name.endswith('oracle'),
                                    max_relative_projected_gradient=max(d['relative_projected_gradient'] for d in all_diagnostics),
                                    min_final_state=min(d['minimum_final_state'] for d in all_diagnostics)))
                                work.checks[f'feasible_{source_seed}_{target_seed}_{factor}_{consumer}_{key}'] = min(d['minimum_final_state'] for d in all_diagnostics) >= -1e-9
                            np.savez_compressed(work.run/f'context_native_s{source_seed}_t{target_seed}_{factor}_{consumer}_{name}.npz', **context_outputs)
                    if target_encoder is not None:
                        for name, desired in [('reencode_ols', predicted), ('reencode_raw', change(prediction['aggregate_raw'])),
                                              ('reencode_source_oracle', actual)]:
                            counterfactual = encode_target(work.h[ids, 0] + desired[ids])
                            changes = counterfactual - recoded_base
                            vectors = np.zeros_like(actual)
                            vectors[ids] = changes @ target['decoder']
                            candidates[name] = vectors
                            projection_records.append(dict(source_seed=source_seed, target_seed=target_seed, factor=factor,
                                consumer=consumer, method=name, actual_write_count_mean=float(np.count_nonzero(changes, axis=1).mean()),
                                actual_write_count_max=int(np.count_nonzero(changes, axis=1).max()),
                                source_dependent_oracle=name.endswith('oracle'), encoder_recomputed=True,
                                minimum_final_state=float(counterfactual.min())))
                    for name, vectors in candidates.items():
                        measure(work, name, factor, 'all', consumer, 1., vectors, reference, source_seed, target_seed)
                        for role in ['temporal', 'quoted']:
                            roleids = np.array([i for i in ids if work.rows[i].get('cue_role', 'temporal') == role], dtype=int)
                            if len(roleids):
                                geometry.append(dict(source_seed=source_seed, target_seed=target_seed, factor=factor,
                                    consumer=consumer, role=role, method=name, n=len(roleids),
                                    error_sse=float(np.sum((vectors[roleids]-actual[roleids])**2)),
                                    source_energy=float(np.sum(actual[roleids]**2))))
                    expected += len(ids) * (1 + len(candidates))
                    write(work.run/'projection_diagnostics.json', dict(rows=projection_records))
                    write(work.run/'geometry.json', dict(rows=geometry))
                    work.progress('NATIVE_CONSUMER_COMPLETE', source_seed=source_seed, target_seed=target_seed,
                                  factor=factor, consumer=consumer, methods=len(candidates))
                write(work.run/'fit_diagnostics.json', dict(rows=fit_records))
            del target_encoder
        work.checks['all_rows'] = len(work.metrics) == expected
        work.checks['unique_rows'] = len(work.metrics) == len({(r['source_seed'], r['target_seed'], r['factor'], r['consumer'], r['method'], r['row_id']) for r in work.metrics})
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        (work.run/'stderr.log').write_text(traceback.format_exc())
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
