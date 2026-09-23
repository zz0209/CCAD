import hashlib
import json
from pathlib import Path

import numpy as np


def identity(path):
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fit_operator(hidden, teacher, source_decoder):
    hidden, teacher, decoder = [np.asarray(value, dtype=np.float64) for value in (hidden, teacher, source_decoder)]
    assert hidden.ndim == teacher.ndim == decoder.ndim == 2
    assert hidden.shape == teacher.shape and decoder.shape[1] == hidden.shape[1]
    assert all(np.isfinite(value).all() for value in (hidden, teacher, decoder))
    _, decoder_s, decoder_v = np.linalg.svd(decoder, full_matrices=False)
    decoder_rank = int(np.sum(decoder_s > decoder_s[0]*max(decoder.shape)*np.finfo(np.float64).eps))
    assert decoder_rank > 0
    basis = decoder_v[:decoder_rank]
    x_mean = hidden.mean(axis=0)
    centered = hidden-x_mean
    u, singular, v = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*max(centered.shape)*np.finfo(np.float64).eps))
    assert rank > 0
    ridge = 1e-3*float(np.sum(centered**2))/rank
    coordinates = teacher@basis.T
    y_mean = coordinates.mean(axis=0)
    coef = v.T@((singular/(singular**2+ridge))[:, None]*(u.T@(coordinates-y_mean)))
    intercept = y_mean-x_mean@coef
    normal = centered.T@(centered@coef-(coordinates-y_mean))+ridge*coef
    scale = np.linalg.norm(centered.T@(coordinates-y_mean))+ridge*np.linalg.norm(coef)
    return dict(coef=coef, intercept=intercept, basis=basis), dict(
        rows=len(hidden), input_rank=rank, source_decoder_rank=decoder_rank, ridge=ridge,
        ridge_relative_normal_residual=float(np.linalg.norm(normal)/max(scale, np.finfo(float).tiny)),
        teacher_outside_span_rms=float(np.sqrt(np.mean((teacher-coordinates@basis)**2))))


def load_operators(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {str(part): {key: arrays[str(part)+'__'+key].copy() for key in ('coef', 'intercept', 'basis')}
                for part in arrays['part_names']}


def predict(operators, hidden, part_name):
    operator = operators[part_name]
    hidden = np.asarray(hidden, dtype=np.float64)
    return (hidden@operator['coef']+operator['intercept'])@operator['basis']


def save(path, operators, metadata):
    path = Path(path)
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, part_names=np.array(list(operators)),
                        **{part+'__'+key: value for part, operator in operators.items() for key, value in operator.items()})
    metadata.update(output=identity(path), code=identity(__file__),
                    method='Per-function affine ridge in the fixed source-decoder span; centered inputs and an unpenalized intercept')
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return operators


def fit_stage1(cache_path, output_path):
    operators, diagnostics = {}, {}
    with np.load(cache_path, allow_pickle=False) as arrays:
        parts = arrays['part_names'].astype(str).tolist()
        assert len(parts) == 2 and len(set(parts)) == 2
        for index, part in enumerate(parts):
            operators[part], diagnostics[part] = fit_operator(arrays['hidden'], arrays['teachers'][:, index], arrays['source_decoders'][index])
    return save(output_path, operators, dict(stage=1, cache=identity(cache_path), diagnostics=diagnostics))


def fit_stage2(cache_path, stage1_path, output_path):
    operators = load_operators(stage1_path)
    with np.load(cache_path, allow_pickle=False) as arrays:
        part = str(arrays['part_name'].item())
        assert part not in operators and len(operators) == 2
        operators[part], diagnostics = fit_operator(arrays['hidden'], arrays['teacher'], arrays['source_decoder'])
    return save(output_path, operators, dict(stage=2, cache=identity(cache_path),
                frozen_stage1=identity(stage1_path), new_part=part, diagnostics={part: diagnostics}))
