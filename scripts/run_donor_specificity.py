"""Exchange donor factors through fixed FCC maps on the exposed role panel.

No fitting, source/member selection or new confirmation. Correct vectors replay
the old implementation; wrong-donor vectors and operation-norm controls then use
the same physical consumer. Portable predictor bundles are exported along the way.
"""
import argparse
import json
import traceback
from pathlib import Path

from composition_runtime import CompositionRun, ROOT, write, np
from ccad.artifacts import sha256
from ccad.predictive_operation import PredictiveOperation


def match_operation_norm(delta, reference):
    # This run requires that both factor slots are at the same final token.
    candidate = np.linalg.norm(delta.sum(axis=1), axis=1)
    target = np.linalg.norm(reference.sum(axis=1), axis=1)
    scale = np.divide(target, candidate, out=np.zeros_like(target), where=candidate > 1e-12)
    return delta * scale[:, None, None], int(np.sum((candidate <= 1e-12) & (target > 1e-12)))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    cfg_path = parser.parse_args().config
    work = CompositionRun(cfg_path, 'scripts/run_donor_specificity.py', ['src/ccad/predictive_operation.py'])
    error = None
    try:
        cfg = work.cfg; old_run = ROOT / cfg['frozen_run']; mapping = ROOT / cfg['correspondence_run']
        freeze = json.loads(work.checked(ROOT / cfg['frozen_dir'] / 'FREEZE.json').read_text())
        work.checks['frozen_inputs_exact'] = all(sha256(ROOT / row['path']) == row['sha256'] for row in freeze['files'])
        if not work.checks['frozen_inputs_exact']: raise ValueError('A frozen input changed')
        work.load()
        work.checks['all_rows_exposed_panel'] = work.n == 512 and np.array_equal(work.evaluation_ids, np.arange(work.n))
        work.checks['common_final_position'] = np.array_equal(work.positions[:, 0], work.positions[:, 1])
        if not all(work.checks.values()): raise ValueError('Input/position contract does not hold')
        np.savez_compressed(work.run / 'base_outputs.npz', label_logprobs=work.base[:, work.labels])
        assets = {}
        for spec in cfg['sae_checkpoints']:
            seed = spec['seed']
            codes = np.load(work.checked(old_run / f'seed{seed}_codes.npz'))
            coordinates = np.load(work.checked(ROOT / cfg['source_run'] / f'seed{seed}_source_coordinates.npz'))
            work.checks[f'identical_final_codes_seed{seed}'] = np.array_equal(codes['number_z'], codes['time_z'])
            factors = {}
            for factor in ['number', 'time']:
                z = codes[factor + '_z'].astype(np.float64)
                dz = z[work.donors[factor]] - z
                prefix = f'{factor}_{cfg["source_budgets"][factor]}_'
                support = coordinates[prefix + 'support']; coef = coordinates[prefix + 'coefficients']; basis = coordinates[prefix + 'basis']
                factors[factor] = dict(dz=dz, support=support, basis=basis, coefficients=coef,
                                       q=(dz[:, support] @ coef) @ basis.T)
            assets[seed] = factors
        old = {}
        work.checked(old_run / 'metrics.raw.jsonl')
        with (old_run / 'metrics.raw.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['method'] == 'fcc_group':
                    old[row['source_seed'], row['target_seed'], row['factor'], row['row_id']] = row
        bundle_dir = work.run / 'operations'; bundle_dir.mkdir()
        replay = []; vector_replay = []; norm_errors = []; zero_cases = []; geometry = []; bundle_index = []
        for source_seed, source in assets.items():
            teacher = {}; source_delta = np.zeros_like(work.h, dtype=np.float64)
            for factor, slot in [('number', 1), ('time', 0)]: source_delta[:, slot] = source[factor]['q']
            for factor, slot in [('number', 1), ('time', 0), ('joint', None)]:
                delta = source_delta.copy()
                if slot is not None: delta[:, 1-slot] = 0
                teacher[factor] = work.measure('source_teacher', factor, delta, source_seed=source_seed, reference_kind='full_donor')
            for target_seed, target in assets.items():
                if target_seed == source_seed: continue
                map_path = work.checked(mapping / f'maps_s{source_seed}_t{target_seed}.npz')
                maps = np.load(map_path)
                correct = np.zeros_like(work.h, dtype=np.float64); wrong = np.zeros_like(correct)
                for factor, slot, other in [('number', 1, 'time'), ('time', 0, 'number')]:
                    sf = source[factor]; weights = maps[factor + '_fcc_group_coefficients'].astype(np.float64)
                    members = maps[factor + '_fcc_members']; vectors = weights[members] @ sf['basis'].T
                    omitted = np.setdiff1d(np.arange(len(weights)), members)
                    if np.any(weights[omitted] != 0): raise ValueError('Export would omit a nonzero coefficient')
                    metadata = dict(factor=factor, source_seed=source_seed, target_seed=target_seed,
                                    model='EleutherAI/pythia-1b-deduped', model_revision='7199d8fc61a6d565cd1f3c62bf11525b563e13b2',
                                    hook='gpt_neox.layers.15:resid_post', position='final token before prediction',
                                    target_sae_path=next(s['path'] for s in cfg['sae_checkpoints'] if s['seed'] == target_seed),
                                    source_map_sha256=sha256(map_path), operation='donor-minus-recipient code contrast -> source-aligned residual update',
                                    native_ablation=False, code_scale='original saved SAE encoder units; no extra normalization')
                    operation = PredictiveOperation(members, vectors, len(weights), metadata)
                    bundle_path = bundle_dir / f's{source_seed}_t{target_seed}_{factor}.npz'; operation.save(bundle_path)
                    correct[:, slot] = operation.predict(target[factor]['dz'])
                    wrong[:, slot] = operation.predict(target[other]['dz'])
                    reference = (target[factor]['dz'] @ weights) @ sf['basis'].T
                    vector_replay.append(float(np.max(np.abs(correct[:, slot] - reference))))
                    bundle_index.append(dict(path=bundle_path.relative_to(work.run).as_posix(), sha256=sha256(bundle_path),
                                             source_seed=source_seed, target_seed=target_seed, factor=factor, members=members.tolist()))
                    if (source_seed, target_seed) == (1, 2):
                        selected_rows = np.array([i for i,r in enumerate(work.rows) if r['block']==0 and r['number']==r['past']==r['distractor']==0])
                        np.savez_compressed(work.run / f'example_{factor}.npz', row_ids=selected_rows, target_members=members,
                                            selected_difference=target[factor]['dz'][selected_rows][:,members],
                                            wrong_factor_selected_difference=target[other]['dz'][selected_rows][:,members],
                                            expected_delta=reference[selected_rows])
                for factor, slot in [('number', 1), ('time', 0), ('joint', None)]:
                    intact = correct.copy(); exchanged = wrong.copy()
                    if slot is not None: intact[:, 1-slot] = 0; exchanged[:, 1-slot] = 0
                    matched, zeros = match_operation_norm(exchanged, intact)
                    zero_cases.append(dict(source_seed=source_seed,target_seed=target_seed,factor=factor,zero_candidate_nonzero_reference= zeros))
                    refnorm = np.linalg.norm(intact.sum(1), axis=1); candnorm = np.linalg.norm(matched.sum(1), axis=1)
                    nonzero = np.linalg.norm(exchanged.sum(1),axis=1)>1e-12
                    if nonzero.any(): norm_errors.append(float(np.max(np.abs(refnorm[nonzero]-candnorm[nonzero]))))
                    for name, delta in [('fcc_group', intact), ('fcc_wrong_donor', exchanged), ('fcc_wrong_donor_norm_matched', matched)]:
                        lp = work.measure(name, factor, delta, teacher[factor], source_seed=source_seed,target_seed=target_seed,reference_kind='source_teacher')
                        if name == 'fcc_group':
                            for i, probs in enumerate(lp):
                                replay.append(float(np.max(np.abs(probs[work.labels]-old[source_seed,target_seed,factor,i]['label_logprobs']))))
                        q = delta.sum(1); actual = intact.sum(1)
                        for role in ['temporal', 'quoted']:
                            ix = np.array([r['cue_role']==role for r in work.rows])
                            geometry.append(dict(source_seed=source_seed,target_seed=target_seed,factor=factor,method=name,role=role,n=int(ix.sum()),
                                                 error_to_intact_energy=float(np.sum((q[ix]-actual[ix])**2)),intact_energy=float(np.sum(actual[ix]**2)),
                                                 candidate_energy=float(np.sum(q[ix]**2))))
                work.progress('PAIR_COMPLETE', source_seed=source_seed,target_seed=target_seed)
        work.checks['original_physical_vectors_replayed'] = max(vector_replay)<1e-8
        work.checks['r4_fcc_label_logprobs_replayed'] = max(replay)<1e-5
        work.checks['norm_matched_nonzero_operations'] = max(norm_errors)<1e-8
        work.checks['complete_rows'] = len(work.metrics)==work.n*(5*3+20*3*3)
        work.checks['unique_rows'] = len(work.metrics)==len({(r['source_seed'],r.get('target_seed'),r['factor'],r['method'],r['row_id']) for r in work.metrics})
        work.checks['frozen_inputs_still_exact'] = all(sha256(ROOT / row['path'])==row['sha256'] for row in freeze['files'])
        write(work.run/'donor_checks.json', dict(original_vector_max_error=max(vector_replay),original_four_label_logprob_max_error=max(replay),
              norm_match_max_error=max(norm_errors),zero_cases=zero_cases,
              definition='Wrong factor means the other factor code difference is fed through the same fixed FCC map; joint swaps both inputs. Norm control matches the realized total operation separately for number, time and joint.',
              scope='All512 exposed rows/all20 dependent seed directions; no fitting or independent confirmation'))
        write(work.run/'donor_geometry.json', dict(rows=geometry))
        write(work.run/'operations/INDEX.json', dict(operations=bundle_index,scope='Predictive residual updates; not native decoder operations or semantic/circuit identity'))
        write(work.run/'example_contexts.json', dict(rows=[dict(row_id=int(i),**work.rows[i]) for i in selected_rows],selection='source1target2,block0,number=past=distractor=0; all roles/cues/templates, chosen by metadata only'))
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'; (work.run/'stderr.log').write_text(traceback.format_exc())
    return work.finish(error)


if __name__=='__main__': raise SystemExit(main())
