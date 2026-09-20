from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import numpy as np
from analyze_profile_confirmation import query_order

ROOT = Path(__file__).resolve().parents[1]
BULK = Path('D:/CCAD_Storage/runs/final_science_20260920')
TASKS = ['composer_surgeon_orientation0', 'composer_surgeon_orientation1',
         'model_software_engineer_orientation0', 'model_software_engineer_orientation1']
METHODS = ['initial_refined_common', 'initial_refined_source_metric', 'task_adapted_tangent']


def decomposition(values, clean, source):
    increments = np.diff(values, axis=0)
    error = values[-1]-source
    assert np.allclose(increments.sum(0), error, atol=1e-4, rtol=1e-5)
    scale = float(np.square(source-clean).mean())
    assert scale > 0
    flat = increments.reshape(len(increments), -1)
    moment = flat @ flat.T / flat.shape[-1] / scale
    full = float(np.square(error).mean()/scale)
    diagonal = float(np.trace(moment))
    assert np.allclose(moment.sum(), full, atol=1e-5, rtol=1e-4)
    return dict(relative_mse=full, diagonal=diagonal, cross_site=full-diagonal,
                cancellation_ratio=full/diagonal if diagonal>0 else None,
                source_energy=scale, site_second_moment=moment.tolist(),
                normalized_prefix_error=(np.square(values-source[None]).mean((1, 2))/scale).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='PROGRAM_PATH_DIAGNOSTIC_20260920')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run = BULK / args.run
    assert json.loads((run/'status.json').read_text())['status']=='PASS'
    members = json.loads((run/'membership.json').read_text())
    prefixes = [k for k in json.loads((run/'config.resolved.json').read_text())['hybrid_prefixes'] if k is not None]
    assert prefixes[0]==0 and prefixes[-1]==11
    reference = BULK/'PROFILE_CONFIRMATION_T2_20260920'
    old = json.loads((reference/'membership.json').read_text())
    queries = query_order(run, 'human')
    qi = [query_order(reference, 'human').index(q) for q in queries]
    identities = [r['document_sha256'] for r in old['human_rows']]
    ix = [identities.index(r['document_sha256']) for r in members['human_rows']]
    local = dict(np.load(run/'responses.npz'))
    prior = dict(np.load(reference/'responses.npz'))
    clean, source = local['human__none'].astype(float), local['human__source'].astype(float)
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), run=str(run),
                  scope='Already-exposed biographies and fixed semantic requests, one target seed; exact finite program decomposition at the recorded prefixes, exploratory.',
                  prefixes=prefixes, biographies=len(members['human_rows']),
                  sites=list(json.loads(Path(json.loads((run/'config.resolved.json').read_text())['source_manifest']).read_text())['members']), queries=queries,
                  original_head={}, later_heads={}, endpoint_replay_max_error={})
    result['tolerances'] = dict(same_batch_atol=1e-6, cross_batch_atol=1e-3, rtol=1e-5)
    pooled = dict(np.load(run/'pooled.npz'))
    replay_valid = []
    for method in METHODS:
        values = np.stack([local[f'human__{method}_prefix{k}'].astype(float) for k in prefixes])
        assert np.allclose(values[0], source, atol=1e-4, rtol=1e-5)
        old_end = prior['human__'+method][np.ix_(qi, ix)]
        difference = float(np.abs(values[-1]-old_end).max())
        print(json.dumps(dict(method=method, endpoint_max_difference=difference,
            source_max_difference=float(np.abs(values[0]-source).max()),
            replay_difference_by_query=np.abs(values[-1]-old_end).tolist())), flush=True)
        assert np.allclose(values[-1], local['human__'+method], atol=1e-6, rtol=1e-5)
        replay_valid.append(np.allclose(values[-1], old_end, atol=1e-3, rtol=1e-5))
        result['endpoint_replay_max_error'][method] = difference
        result['original_head'][method] = decomposition(values, clean, source)
        result['later_heads'][method] = {}
        for h, task in enumerate(TASKS):
            probe = np.load(ROOT/'runs/IR04_shift_consumer_seed2_v1_20260916'/f'none__full__{task}__probe42.npz')
            weight, bias = probe['weight'].astype(float).ravel(), float(probe['bias'].item())
            professions = [5, 25] if h<2 else [12, 24]
            mask = np.array([r['profession'] in professions for r in members['human_rows']])
            levels = np.stack([pooled[f'{method}_prefix{k}'][:, mask].astype(float) @ weight + bias for k in prefixes])
            baseline = pooled['none'][:, mask].astype(float) @ weight + bias
            truth = pooled['source'][:, mask].astype(float) @ weight + bias
            result['later_heads'][method][task] = decomposition(levels, baseline, truth)
    assert all(replay_valid), result['endpoint_replay_max_error']
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({m:{k:v for k,v in value.items() if k not in ['site_second_moment','normalized_prefix_error']}
                      for m,value in result['original_head'].items()}, indent=2))


if __name__=='__main__':
    main()
