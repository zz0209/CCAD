from pathlib import Path
from datetime import datetime, timezone
import argparse
import json

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def analyze(search_run):
    config = read(search_run/'config.resolved.json')
    seed = config['target_seed']
    index = read(Path(config['calibration_run'])/'response_index.json')
    with np.load(index['source_reference']) as data:
        documents = data['document_sha256'].copy()
        baseline = data['baseline_pooled512'].astype(np.float64)
        reference = data['whole_W_pooled512'].astype(np.float64)-baseline
    singletons = {}
    for path in index['target_blocks']:
        with np.load(path) as block:
            np.testing.assert_array_equal(block['document_sha256'], documents)
            for j, member in enumerate(block['member_ids']):
                singletons[str(block['site']), int(member)] = block['delta_pooled512'][:, :, j].astype(np.float64)
    results = read(search_run/'SEARCH_RESULTS.json')
    methods = {}
    for method, result in results.items():
        with np.load(result['best']['path']) as data:
            np.testing.assert_array_equal(data['document_sha256'], documents)
            methods[method] = (result['best']['members'], data['pooled512'].astype(np.float64)-baseline)
    previous = Path('D:/CCAD_Storage/runs/finite_vector_reuse_20260923')/f'CC23_SHIFT_VECTOR_ORIGINAL_T{seed}_20260923'
    previous_index = read(previous/'response_index.json')
    for path in previous_index['group_blocks']:
        with np.load(path) as data:
            if str(data['method']) != 'finite_vector_n8':
                continue
            positions = {str(value): i for i, value in enumerate(data['document_sha256'])}
            selected = np.array([positions[str(value)] for value in documents])
            actual = data['pooled512'][selected].astype(np.float64)-data['baseline_pooled512'][selected]
            methods['finite_vector_n8'] = (json.loads(str(data['selected_members'])), actual)
    assert set(methods) == {'adaptive', 'passive', 'finite_vector_n8'}
    output = {}
    scale = float(np.sum(reference**2))
    for method, (members, actual) in methods.items():
        predicted = sum(singletons[site, member] for site, values in members.items() for member in values)
        additive_error = predicted-reference
        interaction = actual-predicted
        error = actual-reference
        np.testing.assert_allclose(error, additive_error+interaction, atol=1e-12, rtol=1e-12)
        terms = dict(actual_error_sse=float(np.sum(error**2)),
            additive_error_sse=float(np.sum(additive_error**2)),
            interaction_sse=float(np.sum(interaction**2)),
            signed_cross_term=float(2*np.sum(additive_error*interaction)))
        np.testing.assert_allclose(terms['actual_error_sse'],
            terms['additive_error_sse']+terms['interaction_sse']+terms['signed_cross_term'], atol=1e-10)
        output[method] = dict(terms, actual_nrmse=(terms['actual_error_sse']/scale)**.5,
            additive_nrmse=(terms['additive_error_sse']/scale)**.5, members=members,
            per_document_actual_sse=np.sum(error**2, axis=(1, 2)).tolist(),
            per_document_additive_sse=np.sum(additive_error**2, axis=(1, 2)).tolist())
    return dict(target_seed=seed, methods=output, source_energy=scale,
        documents=documents.tolist(), search_run=str(search_run),
        singleton_run=config['calibration_run'], previous_vector_run=str(previous))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    results = [analyze(run) for run in args.runs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Post-selection calibration mechanism analysis; singleton arrays were not available to group-feedback search',
        definition='Interaction is whole-group finite response minus sum of finite singleton responses; it includes all orders and dynamic re-encoding',
        results=results)
    args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({str(item['target_seed']): {method: {key: value for key, value in values.items()
        if key in ('actual_error_sse', 'additive_error_sse', 'interaction_sse', 'signed_cross_term')}
        for method, values in item['methods'].items()} for item in results}), flush=True)


if __name__ == '__main__':
    main()
