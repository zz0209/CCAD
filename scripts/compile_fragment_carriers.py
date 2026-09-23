from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import time

import numpy as np


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_compiled(directory):
    directory = Path(directory)
    metadata = json.loads((directory/'deployment.json').read_text())
    path = directory/'carriers.npz'
    assert digest(path) == metadata['arrays_sha256']
    with np.load(path) as saved:
        arrays = {key:saved[key] for key in saved.files}
    assert arrays['direction_energy'].dtype == np.float64
    assert arrays['mean_codes'].dtype == np.float64
    return arrays, metadata


def apply_whole_carriers(codes, decoder, compiled, role):
    codes, decoder = np.asarray(codes), np.asarray(decoder)
    role_index = compiled['roles'].tolist().index(role)
    centered = codes-compiled['mean_codes']
    energies = np.square(centered)*compiled['direction_energy'][role_index]
    assert codes.ndim == 2 and energies.shape == codes.shape
    assert decoder.shape[0] == codes.shape[1] and np.isfinite(energies[codes>0]).all()
    ids = np.full((len(codes), 4), -1, np.int64)
    for row in range(len(codes)):
        candidates = np.flatnonzero(codes[row]>0)
        chosen = candidates[np.lexsort((candidates, -energies[row, candidates]))[:4]]
        ids[row, :len(chosen)] = chosen
    valid = ids >= 0
    selected = np.maximum(ids, 0)
    coefficients = np.take_along_axis(centered, selected, axis=1)*valid
    delta = -np.einsum('nk,nkd->nd', coefficients, decoder[selected])
    return dict(ids=ids, valid=valid, delta=delta)


def compile_run(run, output):
    import torch
    from functional_fragment_model import load_basis, ROLES

    started, cpu_started = time.perf_counter(), time.process_time()
    started_utc = datetime.now(timezone.utc).isoformat()
    run, output = Path(run), Path(output)
    index_path = run/'response_index.json'
    index = json.loads(index_path.read_text())
    config = json.loads((run/'config.resolved.json').read_text())
    seed = config['target_seed']
    checkpoint = index['identity']['checkpoint_identities'][str(seed)]
    assert digest(checkpoint['path']) == checkpoint['sha256']
    decoder = torch.load(checkpoint['path'], map_location='cpu', weights_only=True)['decoder.weight'].T.numpy()
    source_run = Path(index['source_run'])
    basis = load_basis(source_run/'basis')
    basis_path = source_run/'basis/basis.npz'
    assert digest(basis_path) == index['source_basis_sha256']
    mean_path = source_run/'independent_means.npz'
    with np.load(mean_path) as saved:
        mean_codes = saved[f'seed{seed}_mean_codes']
    energies = []
    for role in ('F', 'L'):
        columns = ROLES[role]
        coordinates = decoder@basis.B[:, columns]
        dual_gram = basis.V[:, columns].T@basis.V[:, columns]
        energy = np.einsum('ij,ij->i', coordinates@dual_gram, coordinates)
        assert np.all(energy >= -1e-10*np.max(np.abs(energy)))
        energies.append(np.maximum(energy, 0))
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'carriers.npz', roles=['F', 'L'], direction_energy=np.stack(energies), mean_codes=mean_codes)
    metadata = dict(target_seed=seed, checkpoint=checkpoint, model_revision=index['identity']['model_revision'], hook=index['identity']['hook'], arrays_sha256=digest(output/'carriers.npz'), preparation_source_basis_sha256=digest(basis_path), preparation_mean_sha256=digest(mean_path), source_run=str(source_run), target_run=str(run), target_response_index_sha256=digest(index_path), runtime_inputs=['target natural codes', 'target decoder', 'carriers.npz'], selection='Four z>0 members by descending float64 (z-mean_codes)^2*direction_energy; feature-ID ascending ties', operation='Negative sum of selected centered target decoder contributions; incoming residual retained', compiled_roles=['F', 'L'], code_sha256=digest(__file__))
    (output/'deployment.json').write_text(json.dumps(metadata, indent=2)+'\n')
    compiled, _ = load_compiled(output)
    response_paths = {(record['start'], record['operation']):record['path'] for record in index['response_blocks']}
    checked_words, maximum_delta_error = 0, 0.
    for block_index, capture_path in enumerate(index['capture_blocks']):
        with np.load(capture_path) as saved:
            codes, families = saved['target_codes'], saved['word_family_sha256']
        start = checked_words
        for role in ('F', 'L'):
            actual = apply_whole_carriers(codes, decoder, compiled, role)
            with np.load(response_paths[start, f'{role}_carrier4_whole']) as saved:
                np.testing.assert_array_equal(families, saved['word_family_sha256'])
                np.testing.assert_array_equal(actual['ids'], saved[f'{role}_carrier_ids'])
                np.testing.assert_array_equal(actual['delta'], saved['delta'])
                maximum_delta_error = max(maximum_delta_error, float(np.max(np.abs(actual['delta']-saved['delta']))))
        checked_words += len(codes)
        if (block_index+1)%16 == 0 or block_index+1 == len(index['capture_blocks']):
            print(json.dumps(dict(target_seed=seed, checked_words=checked_words, capture_blocks=block_index+1)), flush=True)
    result = dict(status='PASS', target_seed=seed, checked_words=checked_words, roles=['F', 'L'], checked_role_word_pairs=2*checked_words, carrier_ids_exact=True, delta_exact=True, maximum_absolute_delta_error=maximum_delta_error, source_assets_read_during_apply=False, started_utc=started_utc, finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu_started, input_index_sha256=digest(index_path), arrays_bytes=(output/'carriers.npz').stat().st_size, code_sha256=digest(__file__))
    (output/'CHECK.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compile_run(args.run, args.output)


if __name__ == '__main__':
    main()
