import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np


ROLES = {'F': slice(0, 25), 'L': slice(25, 50), 'F+L': slice(0, 50)}


@dataclass(frozen=True)
class FragmentBasis:
    B: np.ndarray
    V: np.ndarray
    mean_reconstruction: np.ndarray
    mean_labels: np.ndarray
    mean_codes: np.ndarray
    metadata: dict


def file_identity(path):
    path = Path(path)
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return dict(path=str(path.resolve()), sha256=digest)


def array_identity(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(memoryview(value).cast('B'))
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=digest.hexdigest())


def contrast_labels(first, last):
    first, last = np.asarray(first), np.asarray(last)
    assert first.ndim == last.ndim == 1 and len(first) == len(last)
    assert np.issubdtype(first.dtype, np.integer) and np.issubdtype(last.dtype, np.integer)
    assert np.all((first >= 0) & (first < 26)) and np.all((last >= 0) & (last < 26))
    columns = np.arange(25)
    return np.concatenate([(first[:, None] == columns)-(first[:, None] == 25).astype(np.float64),
                           (last[:, None] == columns)-(last[:, None] == 25).astype(np.float64)], axis=1)


def factor_norm_squared(left, middle, right):
    return float(np.trace(middle.T@(left.T@left)@middle@(right.T@right)))


def basis_diagnostics(B, V, gram_eigenvalues):
    singular = np.linalg.svd(B, compute_uv=False)
    numerical_rank = int(np.linalg.matrix_rank(B))
    retained = np.abs(gram_eigenvalues) > np.max(np.abs(gram_eigenvalues))*1e-6
    response = B.T@V
    results = dict(label_response_matrix=response.tolist(),
        singular_values=singular.tolist(), gram_eigenvalues=gram_eigenvalues.tolist(),
        numerical_rank=numerical_rank, pseudoinverse_retained_rank=int(retained.sum()),
        full_50_coordinate_identity_applicable=bool(retained.sum() == 50),
        full_column_condition_number=float(singular[0]/singular[-1]) if numerical_rank == 50 else None,
        retained_gram_condition_number=float(np.max(np.abs(gram_eigenvalues[retained]))/
            np.min(np.abs(gram_eigenvalues[retained]))) if retained.any() else None,
        identity_maximum_absolute_error=float(np.max(np.abs(response-np.eye(50)))),
        projection_identities={}, cross_projections={})
    for role, columns in ROLES.items():
        W, dual = B[:, columns], V[:, columns]
        defect = W.T@dual-np.eye(W.shape[1])
        results['projection_identities'][role] = dict(
            idempotence_frobenius_squared=factor_norm_squared(dual, defect, W),
            same_role_label_response=(W.T@dual).tolist())
    for first, second in [('F', 'L'), ('L', 'F')]:
        left, right = V[:, ROLES[first]], B[:, ROLES[second]]
        cross = B[:, ROLES[first]].T@V[:, ROLES[second]]
        results['cross_projections'][first+'_after_'+second] = dict(
            label_cross_response=cross.tolist(),
            operator_frobenius_squared=factor_norm_squared(left, cross, right))
    return results


def fit_basis(fit_reconstruction, fit_first, fit_last, mean_reconstruction, mean_first,
              mean_last, mean_codes, fit_ids, mean_ids, input_identity=None):
    started = time.perf_counter()
    fit = np.asarray(fit_reconstruction, dtype=np.float64)
    mean = np.asarray(mean_reconstruction, dtype=np.float64)
    fit_ids, mean_ids = np.asarray(fit_ids).astype(str), np.asarray(mean_ids).astype(str)
    assert fit.ndim == mean.ndim == 2 and fit.shape[1] == mean.shape[1]
    assert len(fit_ids) == len(fit) and len(mean_ids) == len(mean)
    assert len(np.unique(fit_ids)) == len(fit_ids) and len(np.unique(mean_ids)) == len(mean_ids)
    assert not np.intersect1d(fit_ids, mean_ids).size
    Y, Ymean = contrast_labels(fit_first, fit_last), contrast_labels(mean_first, mean_last)
    assert len(Y) == len(fit) and len(Ymean) == len(mean)
    assert np.isfinite(fit).all() and np.isfinite(mean).all()
    mean_h, mean_y = mean.mean(0), Ymean.mean(0)
    mean_codes = np.asarray(mean_codes, dtype=np.float64)
    assert mean_codes.ndim in (1, 2) and np.isfinite(mean_codes).all()
    if mean_codes.ndim == 2:
        assert len(mean_codes) == len(mean)
        mean_codes = mean_codes.mean(0)
    X, target = fit-mean_h, Y-mean_y
    gram = X.T@X
    ridge = 1e-3*np.trace(gram)/fit.shape[1]
    assert ridge > 0, 'Source reconstruction has zero centered fitting energy'
    rhs = X.T@target
    system = gram+ridge*np.eye(gram.shape[0])
    B = np.linalg.solve(system, rhs)
    label_gram = B.T@B
    V = B@np.linalg.pinv(label_gram, rtol=1e-6, hermitian=True)
    diagnostics = basis_diagnostics(B, V, np.linalg.eigvalsh(label_gram))
    metadata = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        fitting_rows=len(fit), independent_mean_rows=len(mean), hidden_dimension=fit.shape[1],
        label_coordinates=50, ridge_lambda=float(ridge), ridge_multiplier=1e-3,
        pseudoinverse_rtol=1e-6, normal_equation_residual_norm=float(np.linalg.norm(system@B-rhs)),
        relative_normal_equation_residual=float(np.linalg.norm(system@B-rhs)/np.linalg.norm(rhs)) if np.linalg.norm(rhs) else None,
        fit_label_squared_error=float(np.sum((X@B-target)**2)),
        fit_ids=fit_ids.tolist(), mean_ids=mean_ids.tolist(),
        input_identity=input_identity,
        arrays=dict(fit_reconstruction=array_identity(fit), mean_reconstruction=array_identity(mean),
                    fit_labels=array_identity(Y), independent_mean_labels=array_identity(Ymean),
                    mean_codes=array_identity(mean_codes)),
        label_definition='A through Y indicator minus Z indicator, first and last letters separately',
        intercept_definition='Independent mean label contrast minus independent mean reconstruction times B; intercept is fixed by the mean split',
        operation_definition='Subtract role-projected decoder contributions using z minus independent mean code; original residual and decoder bias remain',
        identity_scope='Individual role projectors are complementary oblique projectors when all 50 coordinates survive the pseudoinverse; actual response matrices and defects are saved at every rank',
        diagnostics=diagnostics, fit_wall_seconds=time.perf_counter()-started)
    return FragmentBasis(B, V, mean_h, mean_y, mean_codes, metadata)


def save_basis(basis, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    array_path, metadata_path = directory/'basis.npz', directory/'BASIS.json'
    assert not array_path.exists() and not metadata_path.exists()
    np.savez_compressed(array_path, B=basis.B, V=basis.V,
        mean_reconstruction=basis.mean_reconstruction, mean_labels=basis.mean_labels,
        mean_codes=basis.mean_codes)
    metadata_path.write_text(json.dumps(dict(basis.metadata, basis=file_identity(array_path),
        generator=file_identity(Path(__file__))), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return dict(arrays=str(array_path.resolve()), metadata=str(metadata_path.resolve()))


def load_basis(directory):
    directory = Path(directory)
    metadata = json.loads((directory/'BASIS.json').read_text(encoding='utf-8'))
    assert file_identity(directory/'basis.npz')['sha256'] == metadata['basis']['sha256']
    with np.load(directory/'basis.npz') as arrays:
        return FragmentBasis(arrays['B'], arrays['V'], arrays['mean_reconstruction'],
                             arrays['mean_labels'], arrays['mean_codes'], metadata)


def role_projection(values, basis, role):
    assert role in ROLES
    values = np.asarray(values)
    assert values.shape[-1] == len(basis.mean_reconstruction)
    columns = ROLES[role]
    return (values@basis.B[:, columns])@basis.V[:, columns].T


def label_scores(reconstruction, basis):
    contrasts = (np.asarray(reconstruction)-basis.mean_reconstruction)@basis.B+basis.mean_labels
    zeros = np.zeros((*contrasts.shape[:-1], 1), dtype=contrasts.dtype)
    return dict(F=np.concatenate([contrasts[..., :25], zeros], axis=-1),
                L=np.concatenate([contrasts[..., 25:], zeros], axis=-1))


def carrier_masks(codes, decoder, basis, role, mean_codes=None, k=4):
    assert role in ['F', 'L'] and k == 4
    codes, decoder = np.asarray(codes), np.asarray(decoder)
    center = basis.mean_codes if mean_codes is None else np.asarray(mean_codes)
    assert codes.ndim == 2 and decoder.shape == (codes.shape[1], len(basis.mean_reconstruction))
    assert center.shape == (codes.shape[1],)
    columns = ROLES[role]
    coordinates = decoder@basis.B[:, columns]
    dual_gram = basis.V[:, columns].T@basis.V[:, columns]
    direction_energy = np.einsum('ij,ij->i', coordinates@dual_gram, coordinates)
    assert np.all(direction_energy >= -1e-10*np.max(np.abs(direction_energy)))
    direction_energy = np.maximum(direction_energy, 0)
    energies = np.square(codes-center)*direction_energy
    return rank_active_carriers(codes, energies, k)


def rank_active_carriers(codes, energies, k=4):
    codes, energies = np.asarray(codes), np.asarray(energies)
    assert codes.ndim == 2 and codes.shape == energies.shape and k == 4
    assert np.isfinite(energies[codes > 0]).all()
    active = codes > 0
    ids = np.full((len(codes), k), -1, dtype=np.int64)
    valid = np.zeros_like(ids, dtype=bool)
    selected_energy = np.full(ids.shape, np.nan)
    mask = np.zeros(codes.shape, dtype=bool)
    for row in range(len(codes)):
        candidates = np.flatnonzero(active[row])
        selected = candidates[np.lexsort((candidates, -energies[row, candidates]))[:k]]
        ids[row, :len(selected)], valid[row, :len(selected)] = selected, True
        selected_energy[row, :len(selected)] = energies[row, selected]
        mask[row, selected] = True
    return dict(mask=mask, ids=ids, valid=valid, energies=selected_energy, active_count=active.sum(1))


def whole_energy_carriers(codes, decoder, basis, mean_codes=None):
    codes, decoder = np.asarray(codes), np.asarray(decoder)
    center = basis.mean_codes if mean_codes is None else np.asarray(mean_codes)
    assert decoder.shape == (codes.shape[1], len(basis.mean_reconstruction))
    energy = np.square(codes-center)*np.sum(decoder**2, axis=1)
    return rank_active_carriers(codes, energy)


def match_source_carriers(source_ids, source_decoder, target_decoder, target_codes=None):
    from scipy.optimize import linear_sum_assignment
    source_ids = np.asarray(source_ids, dtype=np.int64)
    source_decoder, target_decoder = np.asarray(source_decoder), np.asarray(target_decoder)
    assert source_ids.ndim == 2 and source_ids.shape[1] == 4 and np.all(source_ids >= 0)
    assert source_decoder.shape[1] == target_decoder.shape[1]
    source_norm = np.linalg.norm(source_decoder, axis=1)
    target_norm = np.linalg.norm(target_decoder, axis=1)
    assert np.all(source_norm[np.unique(source_ids)] > 0) and np.all(target_norm > 0)
    normalized_target = target_decoder/target_norm[:, None]
    matches, similarities = np.empty_like(source_ids), np.empty(source_ids.shape)
    for row, members in enumerate(source_ids):
        cosine = (source_decoder[members]/source_norm[members, None])@normalized_target.T
        selected_rows, selected_columns = linear_sum_assignment(-cosine)
        np.testing.assert_array_equal(selected_rows, np.arange(4))
        matches[row], similarities[row] = selected_columns, cosine[selected_rows, selected_columns]
    result = dict(ids=matches, signed_cosines=similarities,
        definition='Maximum total signed cosine over four source rows and the complete target dictionary; unique target columns')
    if target_codes is not None:
        target_codes = np.asarray(target_codes)
        assert target_codes.shape == (len(matches), len(target_decoder))
        result['inactive_count'] = np.sum(np.take_along_axis(target_codes, matches, axis=1) <= 0, axis=1)
    return result


def normalized_utility(delta_margins, scales, role):
    assert role in ['F', 'L']
    delta_margins, scales = np.asarray(delta_margins), np.asarray(scales)
    assert delta_margins.shape[-1] == 2 and scales.shape == (2,)
    assert np.isfinite(scales).all() and np.all(scales >= 0)
    if np.any(scales == 0):
        return np.full(delta_margins.shape[:-1], np.nan)
    requested = 0 if role == 'F' else 1
    protected = 1-requested
    return -delta_margins[..., requested]/scales[requested]-np.abs(delta_margins[..., protected])/scales[protected]


def singleton_oracle(member_ids, delta_margins, scales, role):
    member_ids, delta_margins = np.asarray(member_ids, dtype=np.int64), np.asarray(delta_margins)
    assert len(np.unique(member_ids)) == len(member_ids) and delta_margins.shape == (len(member_ids), 2)
    utility = normalized_utility(delta_margins, scales, role)
    if not np.isfinite(utility).all():
        return dict(ids=np.empty(0, dtype=np.int64), utilities=np.empty(0), defined=False,
                    reason='Independent calibration contains a zero response scale')
    chosen = np.lexsort((member_ids, -utility))[:4]
    return dict(ids=member_ids[chosen], utilities=utility[chosen], defined=True,
                definition='Top four natural-active singleton utilities; joint operation optimum is not inferred')


def operation_delta(codes, decoder, basis, role, mean_codes=None, carrier_ids=None, whole_carriers=False):
    codes, decoder = np.asarray(codes), np.asarray(decoder)
    center = basis.mean_codes if mean_codes is None else np.asarray(mean_codes)
    assert codes.ndim == 2 and decoder.shape == (codes.shape[1], len(basis.mean_reconstruction))
    assert center.shape == (codes.shape[1],)
    centered = codes-center
    if carrier_ids is None:
        assert not whole_carriers
        return -role_projection(centered@decoder, basis, role)
    carrier_ids = np.asarray(carrier_ids, dtype=np.int64)
    assert carrier_ids.shape == (len(codes), 4)
    valid = carrier_ids >= 0
    assert np.all(carrier_ids[valid] < codes.shape[1])
    selected = np.maximum(carrier_ids, 0)
    coefficients = np.take_along_axis(centered, selected, axis=1)*valid
    contribution = np.einsum('nk,nkd->nd', coefficients, decoder[selected])
    return -contribution if whole_carriers else -role_projection(contribution, basis, role)


def response_arrays(data):
    logits = np.asarray(data['letter_logits'], dtype=np.float64)
    logprobs = np.asarray(data['letter_logprobs'], dtype=np.float64)
    operations = np.asarray(data['operation_names']).astype(str).tolist()
    tasks = np.asarray(data['task_names']).astype(str).tolist()
    assert tasks == ['first', 'last'] and logits.shape == logprobs.shape
    assert logits.ndim == 4 and logits.shape[1:] == (len(operations), 2, 26)
    labels = np.stack([data['first_labels'], data['last_labels']], axis=1).astype(np.int64)
    assert labels.shape == (len(logits), 2) and np.all((labels >= 0) & (labels < 26))
    assert np.isfinite(logits).all() and np.isfinite(logprobs).all()
    expected = np.broadcast_to(labels[:, None, :, None], (*logits.shape[:3], 1))
    correct_logits = np.take_along_axis(logits, expected, axis=-1)[..., 0]
    margin = correct_logits-(logits.sum(-1)-correct_logits)/25
    probability = np.exp(np.take_along_axis(logprobs, expected, axis=-1)[..., 0])
    correct = logits.argmax(-1) == labels[:, None]
    clean = operations.index('clean')
    result = dict(operations=operations, margins=margin, delta_margins=margin-margin[:, clean:clean+1],
                  correct=correct, correct_token_probability=probability)
    if 'full_vocab_prediction' in data:
        answer_ids = np.asarray(data['answer_ids'])
        if answer_ids.ndim == 1:
            answer_ids = np.broadcast_to(answer_ids, (2, 26))
        expected_ids = np.stack([answer_ids[task, labels[:, task]] for task in range(2)], axis=1)
        result['full_vocab_correct'] = np.asarray(data['full_vocab_prediction']) == expected_ids[:, None]
    return result


def calibration_scales(calibration):
    arrays = response_arrays(calibration)
    return np.array([np.sqrt(np.mean(arrays['delta_margins'][:, arrays['operations'].index(role), axis]**2))
                     for role, axis in [('F', 0), ('L', 1)]])


def select_response_split(data, split):
    mask = data['split'].astype(str) == split
    assert mask.any(), 'Requested response split has no real observations'
    row_keys = {'letter_logits', 'letter_logprobs', 'full_vocab_prediction', 'first_labels',
                'last_labels', 'word_family_sha256', 'split'}
    return {key: value[mask] if key in row_keys or key.startswith('carrier_') else value
            for key, value in data.items()}


def summarize_source_responses(response_path, calibration_path, output, draws=1000, seed=20260923,
                               response_split='development', calibration_split='calibration'):
    with np.load(response_path) as data:
        response = {key: data[key] for key in data.files}
    calibration_path = response_path if calibration_path is None else calibration_path
    with np.load(calibration_path) as data:
        calibration = {key: data[key] for key in data.files}
    if response_split is not None:
        response = select_response_split(response, response_split)
    if calibration_split is not None:
        calibration = select_response_split(calibration, calibration_split)
    arrays, scales = response_arrays(response), calibration_scales(calibration)
    ids = response['word_family_sha256'].astype(str)
    calibration_ids = calibration['word_family_sha256'].astype(str)
    assert not np.intersect1d(ids, calibration_ids).size
    families, inverse = np.unique(ids, return_inverse=True)
    counts = np.random.default_rng(seed).multinomial(len(families),
        np.full(len(families), 1/len(families)), size=draws)[:, inverse].astype(np.float64)
    counts /= counts.sum(1, keepdims=True)
    weights = np.concatenate([np.full((1, len(ids)), 1/len(ids)), counts])

    def summary(values):
        means = weights@values
        return dict(value=float(means[0]), ci95=np.quantile(means[1:], [.025, .975]).tolist())

    statistics, utilities = {}, {}
    for operation_index, operation in enumerate(arrays['operations']):
        cell = {}
        for axis, name in enumerate(['first', 'last']):
            cell[name] = dict(delta_margin=summary(arrays['delta_margins'][:, operation_index, axis]),
                alphabet_restricted_accuracy=summary(arrays['correct'][:, operation_index, axis]),
                full_vocab_accuracy=summary(arrays['full_vocab_correct'][:, operation_index, axis]),
                correct_token_probability=summary(arrays['correct_token_probability'][:, operation_index, axis]))
        role = operation.split('_')[0]
        if role in ['F', 'L']:
            value = normalized_utility(arrays['delta_margins'][:, operation_index], scales, role)
            utilities[operation] = value
            cell['requested_damage_minus_absolute_protected_change'] = summary(value) if np.isfinite(value).all() else dict(
                value=None, ci95=None, reason='Zero source calibration scale; original margins retained')
        statistics[operation] = cell
    contrasts = {}
    for role in ['F', 'L']:
        fragment, whole = role+'_carrier4_fragment', role+'_carrier4_whole'
        if fragment in utilities and whole in utilities:
            difference = utilities[fragment]-utilities[whole]
            contrasts[role+'_fragment_minus_same_carriers_whole'] = summary(difference) if np.isfinite(difference).all() else dict(value=None, ci95=None)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'WORD_RESPONSES.npz', word_family_sha256=ids,
        operation_names=np.array(arrays['operations']), margins=arrays['margins'],
        delta_margins=arrays['delta_margins'], alphabet_restricted_correct=arrays['correct'],
        full_vocab_correct=arrays['full_vocab_correct'],
        correct_token_probability=arrays['correct_token_probability'], scales=scales, **utilities)
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        inputs=dict(responses=file_identity(response_path), calibration=file_identity(calibration_path)),
        analyzer=file_identity(Path(__file__)), words=len(ids), independent_word_families=len(families),
        scales=dict(first=float(scales[0]), last=float(scales[1]), definition='RMS of source1 full-F first-question and full-L last-question margin changes on the fixed calibration words'),
        statistics=statistics, paired_contrasts=contrasts,
        bootstrap=dict(draws=draws, seed=seed, unit='Shared word family clusters across questions and operations; calibration scales held fixed'),
        endpoint='Actual 26-uppercase-token margins, 26-letter argmax accuracy and full-vocabulary correct-token probability; probe scores excluded')
    (output/'RESULTS.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return result


def check_source_arrays(response_path, basis_directory, analysis_directory):
    with np.load(response_path) as saved:
        data = {name: saved[name] for name in saved.files}
    data = select_response_split(data, 'development')
    result = json.loads((Path(analysis_directory)/'RESULTS.json').read_text())
    basis = load_basis(basis_directory)
    logits = data['letter_logits'].astype(np.float64)
    logprobs = data['letter_logprobs'].astype(np.float64)
    labels = np.stack([data['first_labels'], data['last_labels']], axis=1)
    operations = data['operation_names'].astype(str).tolist()
    margins = np.empty(logits.shape[:3])
    probabilities = np.empty_like(margins)
    correct = np.empty_like(margins)
    for word in range(len(logits)):
        for operation in range(len(operations)):
            for task in range(2):
                target = labels[word, task]
                other = np.arange(26) != target
                margins[word, operation, task] = logits[word, operation, task, target]-logits[word, operation, task, other].mean()
                probabilities[word, operation, task] = np.exp(logprobs[word, operation, task, target])
                correct[word, operation, task] = np.argmax(logprobs[word, operation, task]) == target
    clean = operations.index('clean')
    changes = margins-margins[:, clean:clean+1]
    differences, matrix = {}, {}
    for index, operation in enumerate(operations):
        matrix[operation] = dict(delta_margins=changes[:, index].mean(0).tolist(),
            letter_accuracy=correct[:, index].mean(0).tolist(), correct_token_probability=probabilities[:, index].mean(0).tolist())
        for task, task_name in enumerate(['first', 'last']):
            expected = result['statistics'][operation][task_name]
            differences[operation+'/'+task_name] = dict(
                margin=float(changes[:, index, task].mean()-expected['delta_margin']['value']),
                accuracy=float(correct[:, index, task].mean()-expected['alphabet_restricted_accuracy']['value']),
                probability=float(probabilities[:, index, task].mean()-expected['correct_token_probability']['value']))
    mass = np.exp(logprobs).sum(-1)
    conditional = np.exp(logits-logits.max(-1, keepdims=True))
    conditional /= conditional.sum(-1, keepdims=True)
    reconstructed_probability = conditional*mass[..., None]
    offset = logprobs-logits
    answer_ids = data['answer_ids']
    if answer_ids.ndim == 1:
        answer_ids = np.broadcast_to(answer_ids, (2, 26))
    expected_tokens = np.stack([answer_ids[task, labels[:, task]] for task in range(2)], axis=1)
    check = dict(inputs=dict(responses=file_identity(response_path), basis=file_identity(Path(basis_directory)/'BASIS.json')),
        fit_rows=basis.metadata['fitting_rows'], independent_mean_rows=basis.metadata['independent_mean_rows'],
        diagnostics=basis.metadata['diagnostics'], action_matrix=matrix, independent_summary_differences=differences,
        logprob_logit_common_offset_max_spread=float(np.max(np.ptp(offset, axis=-1))),
        full_probability_vs_conditional_times_letter_mass_max_error=float(np.max(np.abs(np.exp(logprobs)-reconstructed_probability))),
        logits_vs_logprobs_letter_argmax_mismatches=int(np.sum(logits.argmax(-1) != logprobs.argmax(-1))),
        baseline_letter_probability_mass=mass[:, clean].mean(0).tolist(),
        baseline_full_vocab_correct=np.mean(data['full_vocab_prediction'][:, clean] == expected_tokens, axis=0).tolist(),
        baseline_full_vocab_predictions=data['full_vocab_prediction'][:, clean].tolist(),
        baseline_expected_token_ids=expected_tokens.tolist(),
        interpretation='Letter accuracy is conditional on the 26 candidate letters; correct-token probability uses the full vocabulary denominator. Smoke fitting size is retained and supplies no method-selection conclusion.')
    path = Path(analysis_directory)/'RAW_CHECK.json'
    assert not path.exists()
    path.write_text(json.dumps(check, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['# Source response check', '',
        f"Fitting rows {check['fit_rows']}; independent mean rows {check['independent_mean_rows']}; numerical rank {basis.metadata['diagnostics']['numerical_rank']}; retained pseudoinverse rank {basis.metadata['diagnostics']['pseudoinverse_retained_rank']}.", '',
        '| Operation | First margin change | Last margin change | First accuracy | Last accuracy | First token probability | Last token probability |',
        '|---|---|---|---|---|---|---|']
    for operation, cell in matrix.items():
        values = cell['delta_margins']+cell['letter_accuracy']+cell['correct_token_probability']
        lines.append('| '+operation+' | '+' | '.join(f'{value:.8g}' for value in values)+' |')
    (Path(analysis_directory)/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return check


def fit_states(path, output):
    with np.load(path) as data:
        fit, mean = data['split'].astype(str) == 'fit', data['split'].astype(str) == 'mean'
        assert fit.any() and mean.any()
        basis = fit_basis(data['source_reconstruction'][fit], data['first_labels'][fit], data['last_labels'][fit],
            data['source_reconstruction'][mean], data['first_labels'][mean], data['last_labels'][mean],
            data['source_codes'][mean], data['word_family_sha256'][fit], data['word_family_sha256'][mean],
            input_identity=file_identity(path))
    paths = save_basis(basis, output)
    print(json.dumps(dict(paths=paths, diagnostics=basis.metadata['diagnostics'],
                         fit_wall_seconds=basis.metadata['fit_wall_seconds'])))


def read_arrays(path):
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


def family_weights(ids, draws, seed):
    ids = np.asarray(ids).astype(str)
    assert len(ids) == len(np.unique(ids)) and len(ids) > 0
    counts = np.random.default_rng(seed).multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=draws)
    return np.concatenate([np.full((1, len(ids)), 1/len(ids)), counts/len(ids)])


def weighted_summary(values, weights):
    values = np.asarray(values, dtype=np.float64)
    assert values.shape == (weights.shape[1],)
    valid = np.isfinite(values)
    if not valid.all():
        return dict(value=None, ci95=None, defined_words=int(valid.sum()), total_words=len(values),
                    reason='Some requested observations are undefined; no missing value is replaced by zero')
    means = weights@values
    return dict(value=float(means[0]), ci95=np.quantile(means[1:], [.025, .975]).tolist(),
                defined_words=len(values), total_words=len(values))


def operation_role(operation):
    parts = operation.split('_')
    return next((part for part in parts if part in ('F', 'L')), None)


def row_metrics(arrays, operation_index, scales, role=None):
    delta = arrays['delta_margins'][:, operation_index]
    result = {}
    for axis, task in enumerate(('first', 'last')):
        result[task+'_delta_margin'] = delta[:, axis]
        result[task+'_alphabet_restricted_accuracy'] = arrays['correct'][:, operation_index, axis].astype(float)
        result[task+'_full_vocab_accuracy'] = arrays['full_vocab_correct'][:, operation_index, axis].astype(float)
        result[task+'_correct_token_probability'] = arrays['correct_token_probability'][:, operation_index, axis]
    if role is not None:
        requested = 0 if role == 'F' else 1
        result.update(requested_damage=-delta[:, requested], protected_signed_change=delta[:, 1-requested],
            protected_absolute_change=np.abs(delta[:, 1-requested]), utility=normalized_utility(delta, scales, role))
        for name, key in (('alphabet_restricted_accuracy', 'correct'), ('full_vocab_accuracy', 'full_vocab_correct'),
                          ('correct_token_probability', 'correct_token_probability')):
            result['requested_'+name] = arrays[key][:, operation_index, requested].astype(float)
            result['protected_'+name] = arrays[key][:, operation_index, 1-requested].astype(float)
    return result


def summarize_metric_rows(rows, weights):
    return {name: weighted_summary(values, weights) for name, values in rows.items()}


def cohort_metric_rows(per_target):
    assert per_target and all(set(item) == set(per_target[0]) for item in per_target)
    return {name: np.mean(np.stack([item[name] for item in per_target]), axis=0) for name in per_target[0]}


def combine_requested_roles(rows, names):
    first, last = rows[names[0]], rows[names[1]]
    keys = [key for key in first if key == 'utility' or key.startswith(('requested_', 'protected_', 'singleton_', 'joint_'))]
    return {key: (first[key]+last[key])/2 for key in keys if key in last}


def response_at_ids(data, ids):
    lookup = {str(value): index for index, value in enumerate(data['word_family_sha256'])}
    assert len(lookup) == len(data['word_family_sha256']) and all(value in lookup for value in ids)
    indices = np.array([lookup[value] for value in ids])
    keys = {'letter_logits', 'letter_logprobs', 'full_vocab_prediction', 'first_labels', 'last_labels',
            'word_family_sha256', 'split'}
    return {key: value[indices] if key in keys or key.startswith('carrier_') else value for key, value in data.items()}


def compact_observation(logits, logprobs, prediction, labels, answer_ids, baseline_margin):
    logits, logprobs = np.asarray(logits, dtype=np.float64), np.asarray(logprobs, dtype=np.float64)
    assert logits.shape == logprobs.shape and logits.shape[-2:] == (2, 26)
    labels = np.asarray(labels, dtype=np.int64)
    correct_logits = np.take_along_axis(logits, np.broadcast_to(labels[:, None], (*logits.shape[:-1], 1)), -1)[..., 0]
    margins = (26*correct_logits-logits.sum(-1))/25
    ids = np.asarray(answer_ids)
    expected_tokens = ids[labels] if ids.ndim == 1 else ids[np.arange(2), labels]
    return dict(margins=margins, delta_margins=margins-baseline_margin,
                correct=logits.argmax(-1) == labels,
                full_vocab_correct=np.asarray(prediction) == expected_tokens,
                correct_token_probability=np.exp(np.take_along_axis(logprobs,
                    np.broadcast_to(labels[:, None], (*logprobs.shape[:-1], 1)), -1)[..., 0]))


def analyze_audit_run(run, data, scales, split):
    index = json.loads((Path(run)/'response_index.json').read_text(encoding='utf-8'))
    records = [record for record in index['audit_blocks'] if record['split'] == split]
    ids = sorted(record['word_family_sha256'] for record in records)
    if not ids:
        return None
    assert len(ids) == len(set(ids))
    selected_data = response_at_ids(data, ids)
    arrays = response_arrays(selected_data)
    lookup = {value: i for i, value in enumerate(ids)}
    metadata, singleton_rows, metric_rows, selected_records = {}, [], {}, []
    for record in index['response_blocks']:
        if record['operation'] != 'clean':
            continue
        with np.load(record['path']) as saved:
            for row, family in enumerate(saved['word_family_sha256'].astype(str)):
                if family in lookup:
                    metadata[family] = {key: saved[key][row] for key in saved.files
                        if key.endswith('_ids') and key != 'row_ids' or key.endswith('_inactive_count')}
    for role in ('F', 'L'):
        methods = {role+'_carrier4_fragment': role+'_carrier4_fragment',
                   role+'_carrier4_whole': role+'_carrier4_whole'}
        if 'whole_energy4' in arrays['operations']:
            methods.update(whole_energy4='whole_energy4', PW='PW_'+role+'_carrier4_whole')
        for method, operation in methods.items():
            metric_rows[role+'/'+method] = row_metrics(arrays, arrays['operations'].index(operation), scales, role)
        metric_rows[role+'/singleton_oracle_joint'] = {
            key: np.full(len(ids), np.nan) for key in row_metrics(arrays, 0, scales, role)}
        for method in list(metric_rows):
            if not method.startswith(role+'/') or method.endswith('_fragment'):
                continue
            for key in ('selected_inactive_count', 'singleton_requested_damage_sum',
                        'singleton_protected_signed_sum', 'singleton_protected_absolute_sum',
                        'singleton_utility_sum', 'joint_minus_singleton_requested_damage'):
                metric_rows[method][key] = np.full(len(ids), np.nan)
    for record in records:
        family = record['word_family_sha256']
        row = lookup[family]
        saved = read_arrays(record['path'])
        assert saved['word_family_sha256'].astype(str).tolist() == [family]
        labels = np.array([selected_data['first_labels'][row], selected_data['last_labels'][row]])
        base = arrays['margins'][row, arrays['operations'].index('clean')]
        singleton = compact_observation(saved['letter_logits'], saved['letter_logprobs'],
            saved['full_vocab_prediction'], labels, selected_data['answer_ids'], base)
        members = saved['member_ids'].astype(int)
        assert len(members) == len(set(members))
        member_lookup = {int(value): position for position, value in enumerate(members)}
        singleton_rows.append(dict(word_family_sha256=family, member_ids=members.tolist(),
            delta_margins=singleton['delta_margins'].tolist(), alphabet_restricted_correct=singleton['correct'].tolist(),
            full_vocab_correct=singleton['full_vocab_correct'].tolist(),
            correct_token_probability=singleton['correct_token_probability'].tolist(), raw=file_identity(record['path'])))
        for joint_path in record['oracle_joint_paths']:
            joint = read_arrays(joint_path)
            role = str(joint['role'])
            expected = singleton_oracle(members, singleton['delta_margins'], scales, role)
            assert bool(joint['oracle_defined']) == expected['defined']
            np.testing.assert_array_equal(joint['member_ids'], expected['ids'])
            if expected['defined']:
                observed = compact_observation(joint['logits26'], joint['logprobs26'], joint['full_vocab_prediction'],
                    labels, selected_data['answer_ids'], base)
                operation_arrays = {key: value[:, None] for key, value in observed.items()}
                current = row_metrics(operation_arrays, 0, scales, role)
                for key, value in current.items():
                    metric_rows[role+'/singleton_oracle_joint'][key][row] = value[0]
            selections = dict(fragment=metadata[family][role+'_carrier_ids'], oracle=joint['member_ids'])
            if 'whole_energy_ids' in metadata[family]:
                selections.update(whole_energy=metadata[family]['whole_energy_ids'], PW=metadata[family]['PW_'+role+'_ids'])
            for method, selected in selections.items():
                selected = np.asarray(selected, dtype=int)
                selected = selected[selected >= 0]
                missing = [int(member) for member in selected if member not in member_lookup]
                sum_effect = None if missing else singleton['delta_margins'][[member_lookup[int(member)] for member in selected]].sum(0)
                entry = dict(word_family_sha256=family, role=role, method=method, member_ids=selected.tolist(),
                    inactive_or_unmeasured_singletons=missing,
                    singleton_sum_delta_margins=None if sum_effect is None else sum_effect.tolist(),
                    singleton_sum_utility=None if sum_effect is None or np.any(scales == 0)
                    else float(normalized_utility(sum_effect, scales, role)))
                selected_records.append(entry)
                method_key = dict(fragment=role+'/'+role+'_carrier4_whole', oracle=role+'/singleton_oracle_joint',
                                  whole_energy=role+'/whole_energy4', PW=role+'/PW')[method]
                metrics = metric_rows[method_key]
                metrics['selected_inactive_count'][row] = len(missing)
                if sum_effect is not None and len(selected):
                    selected_delta = singleton['delta_margins'][[member_lookup[int(member)] for member in selected]]
                    requested = 0 if role == 'F' else 1
                    metrics['singleton_requested_damage_sum'][row] = -sum_effect[requested]
                    metrics['singleton_protected_signed_sum'][row] = sum_effect[1-requested]
                    metrics['singleton_protected_absolute_sum'][row] = np.abs(selected_delta[:, 1-requested]).sum()
                    metrics['singleton_utility_sum'][row] = normalized_utility(selected_delta, scales, role).sum()
                    metrics['joint_minus_singleton_requested_damage'][row] = metrics['requested_damage'][row]+sum_effect[requested]
    return dict(ids=ids, metric_rows=metric_rows, singleton_rows=singleton_rows, selected_records=selected_records,
                index=file_identity(Path(run)/'response_index.json'))


def summarize_source_audit(run, output, split='development', draws=1000, seed=20260923):
    run, output = Path(run), Path(output)
    data = read_arrays(run/'source_results.npz')
    scales = calibration_scales(select_response_split(data, 'calibration'))
    audit = analyze_audit_run(run, data, scales, split)
    assert audit is not None
    weights = family_weights(audit['ids'], draws, seed)
    statistics = {method: summarize_metric_rows(values, weights) for method, values in audit['metric_rows'].items()}
    contrasts = {}
    for role in ('F', 'L'):
        for left, right in ((role+'/'+role+'_carrier4_fragment', role+'/'+role+'_carrier4_whole'),
                            (role+'/'+role+'_carrier4_whole', role+'/singleton_oracle_joint')):
            common = set(audit['metric_rows'][left]) & set(audit['metric_rows'][right])
            contrasts[left+'_minus_'+right] = summarize_metric_rows(
                {key: audit['metric_rows'][left][key]-audit['metric_rows'][right][key] for key in sorted(common)}, weights)
    result = dict(words=len(audit['ids']), word_family_sha256=audit['ids'], scales=scales.tolist(),
        statistics=statistics, paired_contrasts=contrasts, singleton_rows=audit['singleton_rows'],
        selected_records=audit['selected_records'], input=audit['index'], analyzer=file_identity(__file__),
        bootstrap=dict(draws=draws, seed=seed, unit='Common word families across both requested roles; active members stay inside their word family'))
    path = output/'CARRIER_AUDIT.json'
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return result


def analyze_source_readout(run, output, split='development'):
    run, output = Path(run), Path(output)
    index_path = run/'response_index.json'
    index = json.loads(index_path.read_text(encoding='utf-8'))
    basis = load_basis(run/'basis')
    means = read_arrays(run/'independent_means.npz')
    summaries, development = {}, None
    for part in ('mean', 'fit', split):
        pieces = [read_arrays(path) for path in index['capture_blocks'][part]]
        data = {key: np.concatenate([piece[key] for piece in pieces])
                for key in ('hidden', 'source_reconstruction', 'first_labels', 'last_labels', 'word_family_sha256')}
        scores = label_scores(data['source_reconstruction'], basis)
        summaries[part] = dict(words=len(data['hidden']),
            first_label_accuracy=float(np.mean(scores['F'].argmax(-1) == data['first_labels'])),
            last_label_accuracy=float(np.mean(scores['L'].argmax(-1) == data['last_labels'])))
        if part == split:
            development = data
    data = development
    hidden, reconstruction = data['hidden'].astype(np.float64), data['source_reconstruction'].astype(np.float64)
    centered_h = hidden-means['mean_hidden']
    centered_reconstruction = reconstruction-basis.mean_reconstruction
    energy = dict(hidden=np.sum(hidden**2, axis=1), centered_hidden=np.sum(centered_h**2, axis=1),
                  reconstruction=np.sum(reconstruction**2, axis=1), centered_reconstruction=np.sum(centered_reconstruction**2, axis=1))
    lookup = {str(value): row for row, value in enumerate(data['word_family_sha256'])}
    operations, arrays = {}, dict(word_family_sha256=data['word_family_sha256'], **energy)
    for operation in ('F', 'L', 'F+L', 'F_carrier4_fragment', 'L_carrier4_fragment', 'F_carrier4_whole', 'L_carrier4_whole'):
        delta = np.empty_like(hidden)
        observed_rows = np.zeros(len(hidden), dtype=bool)
        for record in index['response_blocks']:
            if record.get('split') != split or record['operation'] != operation:
                continue
            saved = read_arrays(record['path'])
            positions = [lookup[value] for value in saved['word_family_sha256'].astype(str)]
            delta[positions] = saved['delta']
            observed_rows[positions] = True
        assert observed_rows.all()
        after = label_scores(reconstruction+delta, basis)
        delta_energy = np.sum(delta**2, axis=1)
        cell = dict(delta_rms_norm=float(np.sqrt(delta_energy.mean())), delta_mean_squared_norm=float(delta_energy.mean()),
            energy_ratio={key: float(delta_energy.mean()/value.mean()) if value.mean() > 0 else None for key, value in energy.items()},
            first_readout_accuracy_after=float(np.mean(after['F'].argmax(-1) == data['first_labels'])),
            last_readout_accuracy_after=float(np.mean(after['L'].argmax(-1) == data['last_labels'])))
        if operation in ROLES:
            expected = -role_projection(centered_reconstruction, basis, operation)
            cell['saved_delta_vs_projected_reconstruction_maximum_error'] = float(np.max(np.abs(delta-expected)))
            for role in (('F', 'L') if operation == 'F+L' else (operation,)):
                cell[role+'_erased_readout_vs_independent_mean_maximum_error'] = float(
                    np.max(np.abs(after[role][:, :25]-basis.mean_labels[ROLES[role]])))
        operations[operation] = cell
        arrays[operation+'_delta_energy'] = delta_energy
        arrays[operation+'_first_readout_correct'] = after['F'].argmax(-1) == data['first_labels']
        arrays[operation+'_last_readout_correct'] = after['L'].argmax(-1) == data['last_labels']
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), input=file_identity(index_path),
        analyzer=file_identity(__file__), basis=file_identity(run/'basis/BASIS.json'), label_readout=summaries,
        natural_energy={key: float(value.mean()) for key, value in energy.items()}, operations=operations,
        definition='Fixed source ridge label scores on saved reconstruction, before and after the exact saved word-position delta; independent means retained; no additional model forward or fitting',
        interpretation='Label readout erasure and actual LM effects are separate observations. Full-dictionary target-minus-source intervention equals P times the centered target-minus-source reconstruction residual difference, so full-dictionary agreement alone does not establish member responsibility.')
    path = output/'READOUT_AND_ENERGY.json'
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    np.savez_compressed(output/'READOUT_AND_ENERGY.npz', **arrays)
    return result


def check_cohort_arrays(analysis_directory):
    directory = Path(analysis_directory)
    result = json.loads((directory/'RESULTS.json').read_text(encoding='utf-8'))
    split = result['split']
    source_path = Path(result['inputs']['source']['path'])
    source = read_arrays(source_path)
    basis = json.loads((source_path.parent/'basis/BASIS.json').read_text(encoding='utf-8'))
    observed = []
    for identity in result['inputs']['targets']:
        data = select_response_split(read_arrays(identity['path']), split)
        observed.append(data)
    ids = observed[0]['word_family_sha256'].astype(str)
    assert len(ids) == len(set(ids))
    labels = np.stack([observed[0]['first_labels'], observed[0]['last_labels']], axis=1)
    source_ids = source['word_family_sha256'].astype(str)
    baseline_differences, numeric_differences, direct_rows = {}, {}, []
    first_baseline = observed[0]['letter_logits'][:, observed[0]['operation_names'].astype(str).tolist().index('clean')]
    for data in observed:
        data = response_at_ids(data, ids)
        np.testing.assert_array_equal(np.stack([data['first_labels'], data['last_labels']], axis=1), labels)
        operations = data['operation_names'].astype(str).tolist()
        logits = data['letter_logits'].astype(np.float64)
        margins = np.empty(logits.shape[:3])
        for row in range(len(ids)):
            for task in range(2):
                correct = labels[row, task]
                other = np.arange(26) != correct
                margins[row, :, task] = logits[row, :, task, correct]-logits[row, :, task][:, other].mean(-1)
        delta = margins-margins[:, operations.index('clean'):operations.index('clean')+1]
        values = {}
        for role, axis in (('F', 0), ('L', 1)):
            for operation in (role+'_carrier4_whole', 'PW_'+role+'_carrier4_whole', 'whole_energy4'):
                name = operation if operation != 'whole_energy4' else operation+'_for_'+role
                index = operations.index(operation)
                values[name] = dict(requested_damage=-delta[:, index, axis],
                                    protected_absolute_change=np.abs(delta[:, index, 1-axis]))
        target = str(int(data['target_seed']))
        for name, metrics in values.items():
            for metric, value in metrics.items():
                numeric_differences[target+'/'+name+'/'+metric] = float(value.mean()-result['per_target'][target][name][metric]['value'])
        baseline_differences[target] = float(np.max(np.abs(logits[:, operations.index('clean')]-first_baseline)))
        direct_rows.append(values)
    for name in direct_rows[0]:
        for metric in direct_rows[0][name]:
            actual = np.stack([values[name][metric] for values in direct_rows]).mean(0).mean()
            numeric_differences['cohort/'+name+'/'+metric] = float(actual-result['equal_target'][name][metric]['value'])
    direct_source_baseline_difference = None
    if result['inputs']['direct_source_reference'] is not None:
        reference = select_response_split(read_arrays(result['inputs']['direct_source_reference']['path']), split)
        reference = response_at_ids(reference, ids)
        np.testing.assert_array_equal(np.stack([reference['first_labels'], reference['last_labels']], axis=1), labels)
        direct_source_baseline_difference = float(np.max(np.abs(reference['letter_logits'][:, 0]-first_baseline)))
    overlaps = dict(mean=int(np.intersect1d(ids, basis['mean_ids']).size),
        fit=int(np.intersect1d(ids, basis['fit_ids']).size),
        source_calibration=int(np.intersect1d(ids, source_ids[source['split'].astype(str) == 'calibration']).size))
    if split == 'confirmation':
        overlaps['source_development'] = int(np.intersect1d(ids, source_ids[source['split'].astype(str) == 'development']).size)
    assert all(value == 0 for value in overlaps.values())
    check = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), analysis=file_identity(directory/'RESULTS.json'),
        analyzer=file_identity(__file__), words=len(ids), targets=result['target_seeds'],
        target_word_sets_equal=True, labels_equal=True, historical_word_family_overlap_counts=overlaps,
        baseline_logits_maximum_absolute_differences=baseline_differences,
        direct_source_baseline_logits_maximum_absolute_difference=direct_source_baseline_difference,
        independent_requested_and_protected_mean_differences=numeric_differences,
        maximum_absolute_independent_difference=max(abs(value) for value in numeric_differences.values()),
        scope='Direct correct-letter-minus-other-25 loops on real response arrays; source/target identity and baseline checks. Bootstrap intervals are not independently rerun. Mean effects do not imply per-word response fidelity.')
    path = directory/'RAW_CHECK.json'
    assert not path.exists()
    path.write_text(json.dumps(check, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return check


def summarize_target_cohort(target_runs, source_run, output, split='development', draws=1000, seed=20260923,
                            source_audit_summary_path=None, direct_source_reference=None):
    source_run = Path(source_run)
    assert json.loads((source_run/'status.json').read_text())['status'] == 'PASS'
    source_path = source_run/'source_results.npz'
    source_all = read_arrays(source_path)
    calibration = select_response_split(source_all, 'calibration')
    scales = calibration_scales(calibration)
    first_target = select_response_split(read_arrays(Path(target_runs[0])/'target_results.npz'), split)
    ids = first_target['word_family_sha256'].astype(str)
    source_reference_available = bool(np.any(source_all['split'].astype(str) == split))
    assert source_reference_available or split == 'confirmation', 'Development requires measured paired source responses'
    source = response_at_ids(select_response_split(source_all, split), ids) if source_reference_available else None
    reference = source if source_reference_available else first_target
    assert not np.intersect1d(ids, calibration['word_family_sha256']).size
    weights = family_weights(ids, draws, seed)
    source_arrays = response_arrays(source) if source_reference_available else None
    direct_reference_identity = None
    direct_arrays = source_arrays
    if direct_source_reference is not None:
        direct_data = select_response_split(read_arrays(direct_source_reference), split)
        assert set(direct_data['word_family_sha256'].astype(str)) == set(ids)
        direct_data = response_at_ids(direct_data, ids)
        for key in ('first_labels', 'last_labels', 'answer_ids'):
            np.testing.assert_array_equal(direct_data[key], first_target[key])
        assert direct_data['operation_names'].astype(str).tolist() == ['clean', 'F_carrier4_whole', 'L_carrier4_whole']
        direct_arrays = response_arrays(direct_data)
        direct_reference_identity = file_identity(direct_source_reference)
    run_data, target_arrays, identities, target_seeds, baseline_checks = [], [], [], [], {}
    reference_clean = first_target['operation_names'].astype(str).tolist().index('clean')
    for run in map(Path, target_runs):
        assert json.loads((run/'status.json').read_text())['status'] == 'PASS'
        path = run/'target_results.npz'
        data = select_response_split(read_arrays(path), split)
        assert set(data['word_family_sha256'].astype(str)) == set(ids)
        data = response_at_ids(data, ids)
        np.testing.assert_array_equal(data['first_labels'], reference['first_labels'])
        np.testing.assert_array_equal(data['last_labels'], reference['last_labels'])
        np.testing.assert_array_equal(data['answer_ids'], reference['answer_ids'])
        np.testing.assert_allclose(data['calibration_scales'], scales, rtol=0, atol=1e-10)
        target_seeds.append(int(data['target_seed']))
        clean = data['operation_names'].astype(str).tolist().index('clean')
        baseline_checks[str(int(data['target_seed']))] = dict(
            same_word_families_and_labels=True,
            logits_maximum_absolute_difference_from_first_target=float(np.max(np.abs(
                data['letter_logits'][:, clean].astype(float)-first_target['letter_logits'][:, reference_clean]))),
            full_vocab_prediction_disagreements=int(np.sum(
                data['full_vocab_prediction'][:, clean] != first_target['full_vocab_prediction'][:, reference_clean])))
        run_data.append(data)
        target_arrays.append(response_arrays(data))
        identities.append(file_identity(path))
    assert target_arrays and len(set(target_seeds)) == len(target_seeds)
    operations = target_arrays[0]['operations']
    assert all(item['operations'] == operations for item in target_arrays)
    rows_by_target, statistics = [], {}
    for target_seed, arrays in zip(target_seeds, target_arrays):
        rows = {operation: row_metrics(arrays, i, scales, operation_role(operation))
                for i, operation in enumerate(operations)}
        if source_reference_available:
            for operation in ('F', 'L', 'F+L'):
                source_change = source_arrays['delta_margins'][:, source_arrays['operations'].index(operation)]
                error = arrays['delta_margins'][:, operations.index(operation)]-source_change
                for axis, task in enumerate(('first', 'last')):
                    rows[operation][task+'_source_response_error'] = error[:, axis]
                    rows[operation][task+'_source_response_error_squared'] = error[:, axis]**2
        # 同一个四成员 whole-energy 调用分别承担两个请求。
        for role in ('F', 'L'):
            rows['whole_energy4_for_'+role] = row_metrics(arrays, operations.index('whole_energy4'), scales, role)
        rows_by_target.append(rows)
        statistics[str(target_seed)] = {operation: summarize_metric_rows(values, weights) for operation, values in rows.items()}
    cohort_rows = {operation: cohort_metric_rows([item[operation] for item in rows_by_target]) for operation in rows_by_target[0]}
    cohort = {operation: summarize_metric_rows(values, weights) for operation, values in cohort_rows.items()}
    combined_rows = {method: combine_requested_roles(cohort_rows, names) for method, names in dict(
        full_fragment=('F', 'L'), carrier4_fragment=('F_carrier4_fragment', 'L_carrier4_fragment'),
        carrier4_whole=('F_carrier4_whole', 'L_carrier4_whole'),
        whole_energy4=('whole_energy4_for_F', 'whole_energy4_for_L'),
        PW=('PW_F_carrier4_whole', 'PW_L_carrier4_whole'), raw=('raw_F', 'raw_L')).items()}
    combined = {method: summarize_metric_rows(values, weights) for method, values in combined_rows.items()}
    contrasts = {}
    for role in ('F', 'L'):
        fragment = role+'_carrier4_fragment'
        for other in (role+'_carrier4_whole', 'whole_energy4_for_'+role, 'PW_'+role+'_carrier4_whole'):
            contrasts[fragment+'_minus_'+other] = summarize_metric_rows(
                {key: cohort_rows[fragment][key]-cohort_rows[other][key] for key in cohort_rows[fragment]}, weights)
        whole = role+'_carrier4_whole'
        for other in ('whole_energy4_for_'+role, 'PW_'+role+'_carrier4_whole'):
            contrasts[whole+'_minus_'+other] = summarize_metric_rows(
                {key: cohort_rows[whole][key]-cohort_rows[other][key] for key in cohort_rows[whole]}, weights)
        contrasts[role+'_minus_raw_'+role] = summarize_metric_rows(
            {key: cohort_rows[role][key]-cohort_rows['raw_'+role][key]
             for key in cohort_rows[role] if key in cohort_rows['raw_'+role]}, weights)
    source_statistics = {operation: summarize_metric_rows(row_metrics(source_arrays, i, scales, operation_role(operation)), weights)
                         for i, operation in enumerate(source_arrays['operations'])} if source_reference_available else None
    source_direct_execution, source_direct_baseline_check, direct_rows = None, None, {}
    if direct_arrays is not None:
        for operation in ('clean', 'F_carrier4_whole', 'L_carrier4_whole'):
            direct_rows[operation] = row_metrics(direct_arrays, direct_arrays['operations'].index(operation), scales, operation_role(operation))
        source_direct_execution = {operation: summarize_metric_rows(values, weights) for operation, values in direct_rows.items()}
        direct_clean = direct_arrays['operations'].index('clean')
        target_clean = target_arrays[0]['operations'].index('clean')
        source_direct_baseline_check = dict(common_word_families_labels_and_answer_ids=True,
            margin_maximum_absolute_difference_from_first_target=float(np.max(np.abs(
                direct_arrays['margins'][:, direct_clean]-target_arrays[0]['margins'][:, target_clean]))),
            full_vocab_accuracy_disagreements=int(np.sum(
                direct_arrays['full_vocab_correct'][:, direct_clean] != target_arrays[0]['full_vocab_correct'][:, target_clean])))
        for role in ('F', 'L'):
            whole = role+'_carrier4_whole'
            contrasts[whole+'_minus_source1_'+whole] = summarize_metric_rows(
                {key: cohort_rows[whole][key]-direct_rows[whole][key] for key in cohort_rows[whole]}, weights)
    per_target_contrasts = {}
    for target, rows in zip(target_seeds, rows_by_target):
        target_contrasts = {}
        for role in ('F', 'L'):
            whole = role+'_carrier4_whole'
            comparisons = {name: rows[name] for name in ('whole_energy4_for_'+role, 'PW_'+role+'_carrier4_whole')}
            if direct_arrays is not None:
                comparisons['source1_'+whole] = direct_rows[whole]
            for name, values in comparisons.items():
                target_contrasts[whole+'_minus_'+name] = summarize_metric_rows(
                    {key: rows[whole][key]-values[key] for key in rows[whole]}, weights)
        per_target_contrasts[str(target)] = target_contrasts
    audits = [analyze_audit_run(run, data, scales, split) for run, data in zip(target_runs, run_data)]
    audit_result, audit_arrays = None, {}
    if any(item is not None for item in audits):
        assert all(item is not None for item in audits)
        audit_ids = audits[0]['ids']
        assert all(item['ids'] == audit_ids for item in audits)
        audit_weights = family_weights(audit_ids, draws, seed)
        audit_cohort_rows = {method: cohort_metric_rows([item['metric_rows'][method] for item in audits])
                             for method in audits[0]['metric_rows']}
        audit_contrasts = {}
        for role in ('F', 'L'):
            method = role+'/'+role+'_carrier4_whole'
            for other in (role+'/whole_energy4', role+'/PW', role+'/singleton_oracle_joint'):
                audit_contrasts[method+'_minus_'+other] = summarize_metric_rows(
                    {key: audit_cohort_rows[method][key]-audit_cohort_rows[other][key] for key in audit_cohort_rows[method]}, audit_weights)
        audit_result = dict(words=len(audit_ids), word_family_sha256=audit_ids,
            per_target={str(target): {method: summarize_metric_rows(values, audit_weights) for method, values in item['metric_rows'].items()}
                        for target, item in zip(target_seeds, audits)},
            equal_target={method: summarize_metric_rows(values, audit_weights) for method, values in audit_cohort_rows.items()},
            paired_contrasts=audit_contrasts,
            definition='Actual four-member joint calls on the common frozen audit word panel; oracle ranks natural-active singleton utility and is not a joint optimum')
        audit_combined = {method: combine_requested_roles(audit_cohort_rows, names) for method, names in dict(
            carrier4_fragment=('F/F_carrier4_fragment', 'L/L_carrier4_fragment'),
            carrier4_whole=('F/F_carrier4_whole', 'L/L_carrier4_whole'),
            whole_energy4=('F/whole_energy4', 'L/whole_energy4'), PW=('F/PW', 'L/PW'),
            singleton_oracle_joint=('F/singleton_oracle_joint', 'L/singleton_oracle_joint')).items()}
        audit_result['equal_target_and_requested_role'] = {
            method: summarize_metric_rows(values, audit_weights) for method, values in audit_combined.items()}
        for target, audit in zip(target_seeds, audits):
            for method, metrics in audit['metric_rows'].items():
                for metric, values in metrics.items():
                    audit_arrays[f'seed{target}__{method}__{metric}'] = values
    source_audit_summary = None
    if source_audit_summary_path is not None:
        saved_audit = json.loads(Path(source_audit_summary_path).read_text(encoding='utf-8'))
        source_audit_summary = {key: saved_audit[key] for key in ('words', 'statistics', 'paired_contrasts')}
        source_audit_summary['saved_analysis'] = file_identity(source_audit_summary_path)
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), analyzer=file_identity(__file__),
        inputs=dict(source=file_identity(source_path), targets=identities, direct_source_reference=direct_reference_identity),
        split=split, words=len(ids), target_seeds=target_seeds,
        source_reference_available=source_reference_available,
        source_response_error_status='measured' if source_reference_available else 'not_measured_on_confirmation_words',
        evaluation_identity_reference='paired_source' if source_reference_available else 'first_target_words_labels_and_baseline',
        target_baseline_checks=baseline_checks,
        direct_source_reference_available=direct_arrays is not None,
        source_direct_execution=source_direct_execution, source_direct_baseline_check=source_direct_baseline_check,
        scales=scales.tolist(), source=source_statistics, per_target=statistics, equal_target=cohort,
        equal_target_and_requested_role=combined,
        paired_contrasts=contrasts, per_target_paired_contrasts=per_target_contrasts,
        carrier_audit=audit_result, source_carrier_audit=source_audit_summary,
        bootstrap=dict(draws=draws, seed=seed, unit='Common word families across fixed targets, operations and tasks; target directions averaged within each word; audit families resampled jointly',
            calibration='Source1 calibration RMS held fixed; no gain fitted'),
        utility_definition='Negative requested margin change divided by source requested RMS minus absolute protected change divided by source protected RMS')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'RESULTS.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    np.savez_compressed(output/'WORD_RESPONSES.npz', word_family_sha256=ids, target_seeds=target_seeds,
        operation_names=operations, scales=scales,
        delta_margins=np.stack([item['delta_margins'] for item in target_arrays]),
        alphabet_restricted_correct=np.stack([item['correct'] for item in target_arrays]),
        full_vocab_correct=np.stack([item['full_vocab_correct'] for item in target_arrays]),
        correct_token_probability=np.stack([item['correct_token_probability'] for item in target_arrays]))
    if audit_result is not None:
        np.savez_compressed(output/'AUDIT_WORD_RESPONSES.npz', word_family_sha256=audit_ids, **audit_arrays)
    audit_raw = {str(target): {key: item[key] for key in ('singleton_rows', 'selected_records', 'index')}
                 for target, item in zip(target_seeds, audits) if item is not None}
    (output/'CARRIER_AUDIT.json').write_text(json.dumps(audit_raw, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    def display(value):
        return 'undefined' if value is None else f'{value:.6g}'

    lines = ['# Functional fragment responses', '', f'{len(ids)} word families; fixed targets {target_seeds}; {draws} common-family bootstrap draws.', '',
        '| Operation | Requested damage | Mean absolute protected change | Requested full-vocab accuracy | Protected full-vocab accuracy |',
        '|---|---|---|---|---|']
    for operation, metrics in cohort.items():
        if 'requested_damage' not in metrics:
            continue
        keys = ['requested_damage', 'protected_absolute_change', 'requested_full_vocab_accuracy', 'protected_full_vocab_accuracy']
        lines.append('| '+operation+' | '+' | '.join(display(metrics[key]['value']) for key in keys)+' |')
    lines += ['', '| Operation | First margin change | Last margin change | First full-vocab accuracy | Last full-vocab accuracy |',
              '|---|---|---|---|---|']
    for operation in ('clean', 'F+L', 'raw_F+L'):
        keys = ['first_delta_margin', 'last_delta_margin', 'first_full_vocab_accuracy', 'last_full_vocab_accuracy']
        lines.append('| '+operation+' | '+' | '.join(display(cohort[operation][key]['value']) for key in keys)+' |')
    lines += ['', '| Paired contrast | Requested damage difference [95% CI] | Absolute protected change difference [95% CI] |',
              '|---|---|---|']
    for contrast, metrics in contrasts.items():
        cells = []
        for key in ('requested_damage', 'protected_absolute_change'):
            metric = metrics[key]
            cells.append('undefined' if metric['value'] is None else
                         f"{metric['value']:.6g} [{metric['ci95'][0]:.6g}, {metric['ci95'][1]:.6g}]")
        lines.append('| '+contrast+' | '+' | '.join(cells)+' |')
    if source_direct_execution is not None:
        lines += ['', '| Source direct execution | Requested damage | Mean absolute protected change | Requested full-vocab accuracy | Protected full-vocab accuracy |',
                  '|---|---|---|---|---|']
        for operation in ('F_carrier4_whole', 'L_carrier4_whole'):
            metrics = source_direct_execution[operation]
            keys = ['requested_damage', 'protected_absolute_change', 'requested_full_vocab_accuracy', 'protected_full_vocab_accuracy']
            lines.append('| '+operation+' | '+' | '.join(display(metrics[key]['value']) for key in keys)+' |')
    if audit_result is not None:
        lines += ['', f"Carrier audit uses {audit_result['words']} common word families and actual joint calls.", '',
            '| Request / selection | Requested damage | Mean absolute protected change | Requested full-vocab accuracy | Protected full-vocab accuracy |',
            '|---|---|---|---|---|']
        for method, metrics in audit_result['equal_target'].items():
            keys = ['requested_damage', 'protected_absolute_change', 'requested_full_vocab_accuracy', 'protected_full_vocab_accuracy']
            lines.append('| '+method+' | '+' | '.join(display(metrics[key]['value']) for key in keys)+' |')
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--states', type=Path)
    mode.add_argument('--responses', type=Path)
    mode.add_argument('--target-runs', type=Path, nargs='+')
    parser.add_argument('--source-run', type=Path)
    parser.add_argument('--source-audit-run', type=Path)
    parser.add_argument('--source-audit-summary', type=Path)
    parser.add_argument('--direct-source-reference', type=Path)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--response-split', default='development')
    parser.add_argument('--calibration-split', default='calibration')
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.states:
        fit_states(args.states, args.output)
    elif args.target_runs:
        assert args.source_run is not None
        result = summarize_target_cohort(args.target_runs, args.source_run, args.output,
            split=args.response_split, draws=args.bootstrap, source_audit_summary_path=args.source_audit_summary,
            direct_source_reference=args.direct_source_reference)
        print(json.dumps(dict(output=str(args.output), scales=result['scales'], targets=result['target_seeds'])))
    else:
        result = summarize_source_responses(args.responses, args.calibration, args.output,
            draws=args.bootstrap, response_split=args.response_split, calibration_split=args.calibration_split)
        if args.source_audit_run:
            summarize_source_audit(args.source_audit_run, args.output, split=args.response_split, draws=args.bootstrap)
        print(json.dumps(dict(output=str(args.output), scales=result['scales'], statistics=result['statistics'])))
