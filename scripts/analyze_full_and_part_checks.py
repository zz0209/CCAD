"""Post-hoc paired full/part checks on retained, frozen intervention outcomes.

Measures part-prediction errors on the very same cases whose complete edited
answers agree with the source for every compared method. No model is refitted.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from shift_query_results import ROOT, ART, PARTS, read, load_logits


def interval(a):
    assert np.isfinite(a).all()
    return np.quantile(a, [.025, .975]).tolist()


def human():
    freeze = read(ART/'R59_CONFIRMATION_FREEZE.json')
    panel_path = ART/Path(freeze['panel_path'].replace('\\', '/')).name
    panel = read(panel_path)['rows']
    expected = read(ART/'R59_REPLAY_INDEX.json')['panel']['sha256']
    assert hashlib.sha256(panel_path.read_bytes()).hexdigest() == expected
    order = {r['document_sha256']: i for i, r in enumerate(panel)}
    assert len(order) == len(panel) == 1735
    methods = ['geometry', 'geometry_gain', 'native', 'raw']
    values = [load_logits(ROOT/f'runs/REFORM_R59_shift_confirm_seed{s}_v1_20260915', order) for s in [2, 3, 4, 5]]
    gates, errors, balanced, examples = [], [], [], []
    for seed, v in zip([2, 3, 4, 5], values):
        full = v['source/full'] > 0
        gate = np.all([((v[m+'/full'] > 0) == full) for m in methods], axis=0)
        part_reference = np.stack([v['source/'+q] > 0 for q in PARTS])
        wrong = np.array([(np.stack([v[m+'/'+q] > 0 for q in PARTS]) != part_reference) for m in methods])
        changed = part_reference != (v['none/full'] > 0)[None, :]
        gates.append(gate)
        errors.append(wrong.any(1))
        balanced.append((wrong, changed))
        eligible = gate & wrong[0].any(0) & ~wrong[2].any(0)
        ix = np.flatnonzero(eligible)
        if len(ix):
            i = int(ix[0])
            examples.append(dict(seed=seed, document_sha256=panel[i]['document_sha256'],
                source={q:float(v['source/'+q][i]) for q in ['full',*PARTS]},
                target={m:{q:float(v[m+'/'+q][i]) for q in ['full',*PARTS]} for m in methods}))
    gates, errors = np.stack(gates), np.stack(errors)
    strata = [np.array([i for i, r in enumerate(panel) if r['label'] == y and r['gender'] == g]) for y in [0, 1] for g in [0, 1]]
    rng = np.random.default_rng(610916)
    counts = np.zeros((4000, len(panel)))
    for ix in strata:
        counts[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1/len(ix)), 4000)
    seed_draws = rng.integers(0, 4, (4000, 4))
    conditional_den = counts @ gates.T
    assert (conditional_den > 0).all()
    distributions, summary = {}, {}
    for mi, m in enumerate(methods):
        per_seed = (errors[:, mi] & gates).sum(1)/gates.sum(1)
        estimates = (counts @ (errors[:, mi] & gates).T)/conditional_den
        samples = estimates[np.arange(4000)[:, None], seed_draws].mean(1)
        distributions[m] = samples
        balanced_points = []
        for si, (wrong, changed) in enumerate(balanced):
            g = gates[si]
            terms = []
            for qi in range(len(PARTS)):
                assert (g & changed[qi]).any() and (g & ~changed[qi]).any()
                terms.append(.5*((~wrong[mi, qi])[g & changed[qi]].mean()+
                                  (~wrong[mi, qi])[g & ~changed[qi]].mean()))
            balanced_points.append(float(np.mean(terms)))
        summary[m] = dict(any_part_error=float(per_seed.mean()), paired95=interval(samples),
                           per_seed=per_seed.tolist(), balanced_part_agreement=float(np.mean(balanced_points)),
                           error_counts=(errors[:, mi] & gates).sum(1).tolist())
    difference = distributions['geometry_gain']-distributions['native']
    return dict(documents=len(panel), target_seeds=[2,3,4,5], parts=PARTS,
                gate='Every compared method agrees with the source full-edit answer; identical cases for all part comparisons.',
                gate_counts=gates.sum(1).tolist(), gate_fraction=float(gates.mean()), methods=summary,
                calibrated_geometry_minus_relation=dict(difference=summary['geometry_gain']['any_part_error']-summary['native']['any_part_error'], paired95=interval(difference)),
                examples=examples, example_selection='First eligible document in original panel order, independently for every target; illustrative, not selected for effect size.',
                inference='Paired profession/gender-stratified documents and independent target seeds; source and three semantic parts fixed.')


def arithmetic():
    with np.load(ART/'r39_member_confirmation.npz') as z:
        arrays = {k:z[k] for k in z.files}
    cfg = read(ROOT/read(ART/'R39_CONFIRMATION_FREEZE.json')['config_path'])
    relations = read(ROOT/cfg['member_queries']['relation_run']/'config.resolved.json')['relation_transfer']['seed_pairs']
    source_order = [cfg['seeds'].index(next(s for s, target in relations if target == t)) for t in cfg['seeds']]
    full_names, part_names = arrays['full_methods'].tolist(), arrays['methods'].tolist()
    source = arrays['full_answers'][full_names.index('source')][..., source_order]
    methods = ['assignment', 'member']
    # Only these two target methods have retained full and partial predictions.
    # An unparseable complete answer does not pass a full-answer check.
    gate = (source >= 10) & (source < 100)
    for m in methods:
        gate &= arrays['full_answers'][full_names.index(m)] == source
    # Every bank uses the identical complete intervention: Q x B x O x T x S.
    nb = arrays['answers'].shape[2]
    common = np.broadcast_to(gate[:,None,...], (gate.shape[0], nb, *gate.shape[1:]))
    den = common.sum((2,3,4))
    rng = np.random.default_rng(610917)
    qw = rng.multinomial(len(gate), np.full(len(gate), 1/len(gate)), 4000)
    bw = rng.multinomial(nb, np.full(nb, 1/nb), 4000)
    sample_den = np.einsum('dq,db,qb->d', qw, bw, den)
    assert (sample_den > 0).all()
    summaries, samples = {}, {}
    for m in methods:
        wrong = arrays['answers'][part_names.index(m)] != arrays['source_aligned']
        error = wrong.any(2) & common
        num = error.sum((2,3,4))
        draws = np.einsum('dq,db,qb->d', qw, bw, num)/sample_den
        samples[m] = draws
        expanded_gate = np.broadcast_to(common[:,:,None,...], wrong.shape)
        changed = arrays['changed']
        balanced = .5*((~wrong)[expanded_gate & changed].mean()+(~wrong)[expanded_gate & ~changed].mean())
        summaries[m] = dict(any_part_error=float(num.sum()/den.sum()), paired95=interval(draws),
                            error_count=int(num.sum()), balanced_part_agreement=float(balanced),
                            by_partition=(num.sum(0)/den.sum(0)).tolist())
    return dict(question_clusters=len(gate),partition_banks=nb,gate_count=int(gate.sum()),
                complete_cases=int(gate.size), part_pair_cases=int(common.sum()),
                gate_fraction=float(gate.mean()),methods=summaries,
                assignment_minus_relation=dict(difference=summaries['assignment']['any_part_error']-summaries['member']['any_part_error'],paired95=interval(samples['assignment']-samples['member'])),
                gate='Both compared target full answers equal the valid, aligned source full answer; each later partition compares both complementary halves.',
                inference='Paired question-cluster and partition-bank bootstrap; all operations, prompts and dependent SAE directions kept together.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new output; retained results are immutable.')
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
                scope='Post-hoc paired-outcome analysis on already exposed R59/R39 confirmation outputs. No new model runs, fitted parameters, primary endpoints or independent confirmation.',
                human=human(),arithmetic=arithmetic())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:{a:b for a,b in result[k].items() if a in ['gate_counts','gate_fraction','gate_count','part_pair_cases','methods','calibrated_geometry_minus_relation','assignment_minus_relation']} for k in ['human','arithmetic']}))


if __name__=='__main__':main()
