"""Member decomposition and interventions on the exposed frozen role panel.

No fitting, support selection, new training, or independent-confirmation claim.
Predictive-member deletion removes a term of the source-aligned edit; native
decoder operations are different interventions and are explicitly named.
"""
import argparse
import json
import traceback
from pathlib import Path
from composition_runtime import CompositionRun, ROOT, write, np
from ccad.artifacts import sha256


def norm_match(candidate, reference):
    cn = np.linalg.norm(candidate, axis=1)
    rn = np.linalg.norm(reference, axis=1)
    scale = np.divide(rn, cn, out=np.zeros_like(rn), where=cn > 1e-12)
    return candidate * scale[:, None], int(np.sum((cn <= 1e-12) & (rn > 1e-12)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', type=Path, required=True); args = ap.parse_args()
    work = CompositionRun(args.config, 'scripts/run_member_mechanism.py', [])
    error = None
    try:
        cfg = work.cfg; frozen = ROOT / cfg['frozen_run']; mapping = ROOT / cfg['correspondence_run']
        freeze = json.loads(work.checked(ROOT / cfg['frozen_dir'] / 'FREEZE.json').read_text())
        work.checks['frozen_inputs_exact'] = all(sha256(ROOT / row['path']) == row['sha256'] for row in freeze['files'])
        assert work.checks['frozen_inputs_exact']
        work.load()
        from sparsify import SparseCoder
        assets = {}
        for spec in cfg['sae_checkpoints']:
            seed = spec['seed']; path = Path(spec['path'])
            work.checked(path / 'sae.safetensors'); work.checked(path / 'cfg.json')
            sae = SparseCoder.load_from_disk(path, device='cpu').eval()
            decoder = sae.W_dec.detach().numpy().astype(np.float64); del sae
            codes = np.load(work.checked(frozen / f'seed{seed}_codes.npz'))
            coordinates = np.load(work.checked(ROOT / cfg['source_run'] / f'seed{seed}_source_coordinates.npz'))
            factors = {}
            for factor in ['number', 'time']:
                z = codes[factor + '_z'].astype(np.float64)
                dz = z[work.donors[factor]] - z
                prefix = f'{factor}_{cfg["source_budgets"][factor]}_'
                basis = coordinates[prefix + 'basis']; support = coordinates[prefix + 'support']; coefficients = coordinates[prefix + 'coefficients']
                q = dz[:, support] @ coefficients @ basis.T
                factors[factor] = dict(dz=dz, support=support, basis=basis, coefficients=coefficients, q=q)
            assets[seed] = dict(decoder=decoder, **factors)
        role = np.array([row['cue_role'] for row in work.rows])
        # The paired role is fixed entirely by generator metadata, not effects.
        key = lambda row: tuple(row[k] for k in ['block', 'cue_id', 'template', 'number', 'past', 'distractor'])
        lookup = {(key(row), row['cue_role']): i for i, row in enumerate(work.rows)}
        twin = np.array([lookup[key(row), 'quoted' if row['cue_role'] == 'temporal' else 'temporal'] for row in work.rows])
        work.checks['role_pair_involution'] = np.array_equal(twin[twin], np.arange(work.n))
        cycle = {tuple(pair) for pair in cfg['cycle_directions']}
        diagnostics = []; physical = {}; memberships = []; sum_errors = []
        for source_seed, source in assets.items():
            for target_seed, target in assets.items():
                if source_seed == target_seed: continue
                maps = np.load(work.checked(mapping / f'maps_s{source_seed}_t{target_seed}.npz'))
                for factor in ['number', 'time']:
                    sf = source[factor]; dz = target[factor]['dz']; members = maps[factor + '_fcc_members']
                    weights = maps[factor + '_fcc_group_coefficients'].astype(np.float64)
                    coeff = weights[members]; vectors = coeff @ sf['basis'].T
                    x = dz[:, members]; q = (x @ coeff) @ sf['basis'].T
                    original = (dz @ weights) @ sf['basis'].T
                    sum_errors.append(float(np.max(np.abs(q-original))))
                    lift = coeff @ np.linalg.pinv(sf['coefficients'])
                    lift_error = float(np.max(np.abs(lift @ source['decoder'][sf['support']] - vectors)))
                    native_vectors = target['decoder'][members]; native = x @ native_vectors
                    lengths = np.linalg.norm(vectors, axis=1)
                    native_lengths = np.linalg.norm(native_vectors, axis=1)
                    cosine = np.sum(vectors*native_vectors, axis=1)/np.maximum(lengths*native_lengths, 1e-30)
                    gram = vectors @ vectors.T
                    for row_role in ['temporal', 'quoted']:
                        mask = role == row_role; xx = x[mask]; qq = q[mask]
                        member_energy = np.sum(xx**2, axis=0) * lengths**2
                        total = float(np.sum(qq**2)); individual = float(member_energy.sum())
                        diagnostics.append(dict(source_seed=source_seed,target_seed=target_seed,factor=factor,role=row_role,n=int(mask.sum()),
                            total_energy=total,individual_energy=individual,cross_energy=total-individual,
                            total_to_individual_energy=total/max(individual,1e-30),native_gap_energy=float(np.sum((qq-native[mask])**2)),
                            mean_total_norm=float(np.linalg.norm(qq,axis=1).mean()),sum_individual_norm_mean=float((np.abs(xx)*lengths).sum(1).mean())))
                    memberships.append(dict(source_seed=source_seed,target_seed=target_seed,factor=factor,members=members.tolist(),
                        source_members=sf['support'].tolist(),source_lift=lift.tolist(),source_lift_error=lift_error,
                        predictive_vector_norm=lengths.tolist(),native_decoder_cosine=cosine.tolist(),
                        temporal_delta_code_rms=np.sqrt(np.mean(x[role=='temporal']**2,axis=0)).tolist(),
                        quoted_delta_code_rms=np.sqrt(np.mean(x[role=='quoted']**2,axis=0)).tolist(),
                        predictive_vector_gram=gram.tolist()))
                    if (source_seed,target_seed) in cycle:
                        physical[source_seed,target_seed,factor] = dict(q=q, native=native, x=x, vectors=vectors, members=members)
        work.checks['full_coefficient_sum_replay'] = max(sum_errors) < 1e-9
        work.checks['source_feature_lift_replay'] = max(row['source_lift_error'] for row in memberships) < 1e-8
        write(work.run/'member_decomposition.json',dict(rows=diagnostics,memberships=memberships,coefficient_sum_max_error=max(sum_errors),
            roles_twin=twin.tolist(),scope='All20maps/all512exposedrows. q=sum_j dz_j v_j; individualenergy=sum_j||dz_jv_j||^2, crossenergy contains ordered cross terms. Finite-contrast predictor rows are not native decoder edges. Source lift uses Moore-Penrose minimum-norm coefficients; uniqueness requires appropriate decoder rank.'))
        work.progress('ALL_MAPS_DECOMPOSED',directions=20)
        # Read old outcomes only for replay, never for selecting cases or members.
        old = {}
        with (frozen/'metrics.raw.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['method']=='fcc_group' and (row.get('source_seed'),row.get('target_seed')) in cycle:
                    old[row['source_seed'],row['target_seed'],row['factor'],row['row_id']] = row
        replay = []; expected_rows = 0; norm_errors = []; zero_norm_cases = 0
        for source_seed,target_seed in cfg['cycle_directions']:
            source = assets[source_seed]; teacher = {}; intact = {}; full_delta = np.zeros_like(work.h,dtype=np.float64)
            for factor,slot in [('number',1),('time',0)]:
                delta = np.zeros_like(full_delta); delta[:,slot] = source[factor]['q']
                teacher[factor] = work.measure('source_teacher',factor,delta,source_seed=source_seed,target_seed=target_seed,reference_kind='full_donor')
                full_delta[:,slot] = physical[source_seed,target_seed,factor]['q']
            source_joint = np.zeros_like(full_delta)
            for factor,slot in [('number',1),('time',0)]: source_joint[:,slot] = source[factor]['q']
            teacher['joint'] = work.measure('source_teacher','joint',source_joint,source_seed=source_seed,target_seed=target_seed,reference_kind='full_donor')
            for factor,slot in [('number',1),('time',0),('joint',None)]:
                delta = full_delta.copy()
                if slot is not None: delta[:,1-slot] = 0
                intact[factor] = work.measure('fcc_group',factor,delta,teacher[factor],source_seed=source_seed,target_seed=target_seed,reference_kind='source_teacher')
                for j,lp in enumerate(intact[factor]):
                    previous = old[source_seed,target_seed,factor,j]
                    replay.append(float(np.max(np.abs(lp[work.labels]-previous['label_logprobs']))))
            expected_rows += 6*work.n
            for factor,slot in [('number',1),('time',0)]:
                asset=physical[source_seed,target_seed,factor]; q=asset['q']
                def measure(name, vector, reference=None, **extra):
                    delta=np.zeros_like(full_delta);delta[:,slot]=vector
                    return work.measure(name,factor,delta,intact[factor] if reference is None else reference,
                        source_seed=source_seed,target_seed=target_seed,reference_kind='full_fcc',**extra)
                native_matched,zeros=norm_match(asset['native'],q);zero_norm_cases+=zeros
                swapped=q[twin];swap_matched,zeros=norm_match(swapped,q);zero_norm_cases+=zeros
                for name,vector in [('same_members_native',asset['native']),('native_norm_matched',native_matched),('role_swap',swapped),('role_swap_norm_matched',swap_matched)]:
                    measure(name,vector)
                expected_rows += 4*work.n
                for index,member in enumerate(asset['members']):
                    component=asset['x'][:,index,None]*asset['vectors'][index]
                    removed=q-component;matched,zeros=norm_match(q,removed);zero_norm_cases+=zeros
                    good=np.linalg.norm(q,axis=1)>1e-12
                    norm_errors.append(float(np.max(np.abs(np.linalg.norm(removed[good],axis=1)-np.linalg.norm(matched[good],axis=1)))))
                    for name,vector in [('member_alone',component),('without_member',removed),('without_member_norm_control',matched)]:
                        measure(name,vector,member=int(member))
                    expected_rows += 3*work.n
                work.progress('MEMBER_FACTOR_COMPLETE',source_seed=source_seed,target_seed=target_seed,factor=factor,members=len(asset['members']))
            # Role-exchanged time while preserving the locally predicted number.
            timeq=physical[source_seed,target_seed,'time']['q'];matched,zeros=norm_match(timeq[twin],timeq);zero_norm_cases+=zeros
            for name,qt in [('joint_time_role_swap',timeq[twin]),('joint_time_role_swap_norm_matched',matched)]:
                delta=full_delta.copy();delta[:,0]=qt
                work.measure(name,'joint',delta,intact['joint'],source_seed=source_seed,target_seed=target_seed,reference_kind='full_fcc')
            expected_rows += 2*work.n
        work.checks['r4_fcc_four_label_logprob_replay'] = max(replay)<1e-5
        work.checks['member_norm_control_matched'] = max(norm_errors)<1e-8
        work.checks['all_rows'] = len(work.metrics)==expected_rows
        work.checks['unique'] = len(work.metrics)==len({(row['source_seed'],row['target_seed'],row['factor'],row['method'],row.get('member'),row['row_id']) for row in work.metrics})
        write(work.run/'mechanism_checks.json',dict(expected_rows=expected_rows,actual_rows=len(work.metrics),r4_label_logprob_max_error=max(replay),norm_match_max_error=max(norm_errors),zero_candidate_nonzero_reference_cases=zero_norm_cases,cycle_directions=cfg['cycle_directions'],
            scope='Member perturbations compare to intactFCC, not donor/sourceKL; fullFCC compares to source teacher. Predictive member deletion changes the source-aligned intervention, not the target native activation stream. All cases retained.'))
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';(work.run/'stderr.log').write_text(traceback.format_exc())
    return work.finish(error)


if __name__=='__main__':raise SystemExit(main())
