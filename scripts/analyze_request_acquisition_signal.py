import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_single_part_composition import identity, read_json


def fit_predict(train_x, train_y, test_x):
    scale = np.sqrt(np.mean(train_x ** 2, axis=0))
    scale[scale == 0] = 1
    coefficients = np.linalg.lstsq(train_x / scale, train_y, rcond=1e-8)[0]
    return (test_x / scale) @ coefficients, coefficients / scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    started = time.perf_counter()
    summary = read_json(args.input / 'RESULTS.json')
    panel = read_json(args.input / 'PANEL.json')
    assert summary['target_seeds'] == [2], 'Only the exposed development target is allowed'
    with np.load(args.input / 'ARRAYS.npz', allow_pickle=False) as archive:
        values = archive['responses'].copy()
        residuals = archive['interaction_residuals'].copy()
    methods = summary['methods']
    queries = summary['families']['all_unseen8']
    query_ids = [panel['query_order'].index(name) for name in queries]
    source = values[methods.index('source'), 0, query_ids].T
    clean = values[methods.index('none'), 0, query_ids].T
    single = values[methods.index('single_program'), 0, query_ids].T
    mixed = values[methods.index('mixed_program'), 0, query_ids].T
    initial = values[methods.index('initial'), 0, query_ids].T
    interaction = residuals[methods.index('source'), 0, query_ids].T
    assert source.shape == (len(panel['rows']), 8)
    assert np.isfinite(values).all() and np.isfinite(residuals).all()
    raw_gain = (single - source) ** 2 - (mixed - source) ** 2
    raw_initial = (initial - source) ** 2
    source_energy = (source - clean) ** 2
    interaction_energy = interaction ** 2
    folds = np.empty(len(panel['rows']), dtype=np.int64)
    row_hashes = []
    for row in panel['rows']:
        content = json.dumps([row['good'], row['bad'], row['position']], separators=(',', ':'))
        row_hashes.append(hashlib.sha256(content.encode()).hexdigest())
    assert len(set(row_hashes)) == len(row_hashes)
    predictions = np.empty((2, len(panel['rows']), len(queries)))
    normalized_gain = np.empty_like(raw_gain)
    reports = {}
    for task in summary['tasks']:
        rows = np.array([i for i, row in enumerate(panel['rows']) if row['task'] == task])
        ordered = sorted(rows, key=lambda i: row_hashes[i])
        folds[ordered] = np.arange(len(rows)) % 4
        slopes = []
        for fold in range(4):
            train = rows[folds[rows] != fold]
            test = rows[folds[rows] == fold]
            assert not set(train) & set(test)
            energy = source_energy[train].mean()
            assert energy > 0
            features = []
            for selected in [train, test]:
                # 请求身份、作用尺度及已有执行误差共同控制；这不是部署选择器。
                request_id = np.tile(np.eye(len(queries)), (len(selected), 1))
                x = np.column_stack([
                    request_id,
                    source_energy[selected].ravel() / energy,
                    raw_initial[selected].ravel() / energy,
                    np.repeat(raw_initial[selected].mean(1), len(queries)) / energy,
                ])
                extra = interaction_energy[selected].ravel()[:, None] / energy
                features.append((x, np.column_stack([x, extra])))
            y = raw_gain[train].ravel() / energy
            normalized_gain[test] = raw_gain[test] / energy
            for model_id in range(2):
                pred, coefficients = fit_predict(features[0][model_id], y, features[1][model_id])
                predictions[model_id, test] = pred.reshape(len(test), len(queries))
                if model_id == 1:
                    slopes.append(float(coefficients[-1]))
        y = normalized_gain[rows]
        errors = ((predictions[:, rows] - y) ** 2).mean((1, 2))
        residual = y - predictions[0, rows]
        extra_prediction = predictions[1, rows] - predictions[0, rows]
        norm = np.linalg.norm(residual) * np.linalg.norm(extra_prediction)
        assert norm > 0
        reports[task] = {
            'rows': len(rows), 'fold_sizes': np.bincount(folds[rows], minlength=4).tolist(),
            'control_mse': float(errors[0]), 'interaction_mse': float(errors[1]),
            'mse_change': float(errors[1] - errors[0]),
            'heldout_residual_cosine': float(np.sum(residual * extra_prediction) / norm),
            'training_partial_slopes': slopes,
            'raw_single_minus_mixed_squared_error_by_request': dict(zip(queries, raw_gain[rows].mean(0).tolist())),
            'source_interaction_rms_by_request': dict(zip(queries, np.sqrt(interaction_energy[rows].mean(0)).tolist())),
        }
        print(json.dumps({'completed_task': task, **reports[task]}), flush=True)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'CROSSFIT.npz', folds=folds, predictions=predictions,
                        normalized_gain=normalized_gain, raw_gain=raw_gain,
                        source_energy=source_energy, interaction_energy=interaction_energy,
                        initial_error=raw_initial)
    result = {
        'status': 'PASS', 'written_utc': datetime.now(timezone.utc).isoformat(),
        'question': 'Does source non-additivity predict the single-versus-mixed error gap on held-out development texts after controlling for request identity, source scale and initial execution error?',
        'evidence_level': 'Retrospective development diagnostic; no new independent confirmation',
        'controls': ['request identity', 'source effect squared', 'initial error squared', 'mean initial error across requests'],
        'predictor': 'Squared actual source residual from singleton-additive response',
        'outcome': 'Single squared response error minus mixed squared response error',
        'split': 'Four folds within grammar from sorted content hashes; each text and all its requests remain together',
        'target_seeds': summary['target_seeds'], 'queries': queries, 'by_grammar': reports,
        'acquisition_boundary': 'Predictor requires actual source combination measurements. Initial target errors are diagnostic controls. Neither is a free prospective acquisition signal.',
        'new_model_forward_calls': 0, 'new_trainable_parameters': 0,
        'wall_seconds': time.perf_counter() - started,
        'numpy_version': np.__version__,
        'inputs': {name: identity(args.input / name) for name in ['RESULTS.json', 'PANEL.json', 'ARRAYS.npz']},
        'script': identity(Path(__file__)), 'output': identity(args.output / 'CROSSFIT.npz'),
    }
    (args.output / 'RESULTS.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
