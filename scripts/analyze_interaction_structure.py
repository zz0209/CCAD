from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np


OPERATIONS = ('P', 'N', 'W', 'PN', 'PW', 'NW', 'PNW')
BITS = (1, 2, 4, 3, 5, 6, 7)
HIGHER = ('PN', 'PW', 'NW', 'PNW')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decompose(values, baseline):
    effects = values-baseline[:, None, :]
    terms = {}
    for index, bits in enumerate(BITS):
        terms[OPERATIONS[index]] = effects[:, index].copy()
        for earlier, earlier_bits in zip(OPERATIONS[:index], BITS[:index]):
            if earlier_bits & bits == earlier_bits:
                terms[OPERATIONS[index]] -= terms[earlier]
    reconstructed = np.stack([sum(terms[name] for name, subset in zip(OPERATIONS, BITS)
        if subset & bits == subset) for bits in BITS], axis=1)
    error = float(np.max(np.abs(reconstructed-effects)))
    np.testing.assert_allclose(reconstructed, effects, atol=1e-11, rtol=1e-12)
    return terms, effects, error


def coefficient(source, target):
    denominator = float(np.sum(source*source))
    return dict(value=float(np.sum(source*target)/denominator) if denominator > 0 else None,
        source_sum_squares=denominator, target_sum_squares=float(np.sum(target*target)),
        status='defined' if denominator > 0 else 'not_applicable_zero_source_energy')


def summary(prediction, target, source, weight):
    source_energy, target_energy = float(np.mean(source**2)), float(np.mean(target**2))
    error = prediction-target
    rmse = float(np.sqrt(np.mean(error**2)))
    norm = float(np.linalg.norm(prediction)*np.linalg.norm(target))
    source_norm = float(np.linalg.norm(source)*np.linalg.norm(target))
    projected_target, projected_error = target@weight, error@weight
    head_energy = float(np.mean(projected_target**2))
    return dict(rmse=rmse, nrmse=rmse/np.sqrt(target_energy) if target_energy > 0 else None,
        cosine=float(np.sum(prediction*target)/norm) if norm > 0 else None,
        source_target_cosine=float(np.sum(source*target)/source_norm) if source_norm > 0 else None,
        source_mean_square_energy=source_energy, target_mean_square_energy=target_energy,
        prediction_mean_square_energy=float(np.mean(prediction**2)),
        head_rmse=float(np.sqrt(np.mean(projected_error**2))),
        head_nrmse=float(np.sqrt(np.mean(projected_error**2)/head_energy)) if head_energy > 0 else None,
        target_head_mean_square_energy=head_energy,
        zero_energy_policy='Undefined ratios and cosines are null')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, default=Path('D:/CCAD_Storage/runs/finite_parts_program_20260923'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/scientific_reform_20260923/INTERACTION_STRUCTURE_ADDITIVE_SUMMARY.json'))
    parser.add_argument('--save-full-vectors', action='store_true')
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.with_suffix('.md').exists()
    started = time.perf_counter()
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Exposed original-task development documents. Hash-fixed eight-document fit and remaining 56-document checks; no independent confirmation.',
        definitions=dict(terms='Exact Boolean Moebius components of pooled512 relative to the same document baseline',
            J='PN + PW + NW + PNW Moebius terms; equals full effect minus three singleton effects',
            metrics='RMSE per hidden coordinate. nRMSE divides by the actual target term RMS. Cosine flattens documents and hidden coordinates.',
            common='Targets are stacked with equal document counts, giving equal target weight.',
            full_response='Singleton coefficients use fit8 only; pairs predict new requests and PNW predicts new documents for the calibrated full request.',
            information=dict(singleton_only='Higher-order terms are zero; complete predictions use identical fit8 singleton coefficients',
                source_unit='No target higher-order fit', joint_scalar='Fit8 J only',
                separate_scalar='Fit8 each higher-order term; three extra pair responses',
                constant_mean='Fit8 mean target higher-order vectors, same information as separate_scalar')),
        inputs=[], targets={}, shared_split=None, common={})
    aggregated = {}
    shared_documents, shared_source, shared_weight = None, None, None
    for seed in (2, 3):
        run = args.run_root/f'CC23_SHIFT_FINITE_PARTS_T{seed}_20260923'
        path = run/'evaluation/original.npz'
        config = json.loads((run/'config.resolved.json').read_text())
        assert json.loads((run/'status.json').read_text())['status'] == 'PASS'
        probe_path = Path(config['frozen_source_run'])/'probe.npz'
        with np.load(probe_path) as probe:
            weight = probe['weight'].reshape(-1).astype(np.float64)
        with np.load(path) as data:
            documents = data['document_sha256'].tolist()
            labels, genders = data['labels'], data['genders']
            names, ops = data['method_names'].tolist(), data['operation_names'].tolist()
            order = [ops.index(name) for name in OPERATIONS]
            baseline = data['baseline_pooled512'].astype(np.float64)
            zero_errors = {method: float(np.max(np.abs(data['pooled512'][:, mi, ops.index('0')]-baseline)))
                for mi, method in enumerate(names)}
            assert all(value == 0 for value in zero_errors.values())
            source_values = data['pooled512'][:, names.index('source'), order].astype(np.float64)
            target_values = data['pooled512'][:, names.index('native'), order].astype(np.float64)
        if shared_documents is None:
            shared_documents, shared_source, shared_weight = documents, source_values, weight
            fit_indices = []
            cells = []
            for label, gender in sorted(set(zip(labels.tolist(), genders.tolist()))):
                available = sorted([i for i in range(len(documents)) if labels[i] == label and genders[i] == gender],
                    key=lambda i: documents[i])
                fit_indices.extend(available[:2])
                cells.append(dict(label=int(label), gender=int(gender), count=len(available),
                    fit_documents=[documents[i] for i in available[:2]]))
            fit_indices = sorted(fit_indices)
            check_indices = [i for i in range(len(documents)) if i not in fit_indices]
            assert len(cells) == 4 and len(fit_indices) == 8 and len(check_indices) == 56
            result['shared_split'] = dict(rule='Within each original label x gender cell, smallest two document SHA256 values',
                fit_count=8, check_count=56, cells=cells, fit_indices=fit_indices, check_indices=check_indices,
                fit_documents=[documents[i] for i in fit_indices], check_documents=[documents[i] for i in check_indices])
        else:
            assert documents == shared_documents
            np.testing.assert_array_equal(weight, shared_weight)
            np.testing.assert_allclose(source_values, shared_source, atol=1e-6, rtol=0)
        source, source_effect, source_error = decompose(source_values, baseline)
        target, target_effect, target_error = decompose(target_values, baseline)
        source['J'], target['J'] = sum(source[name] for name in HIGHER), sum(target[name] for name in HIGHER)
        joint = coefficient(source['J'][fit_indices], target['J'][fit_indices])
        singles = {name: coefficient(source[name][fit_indices], target[name][fit_indices]) for name in OPERATIONS[:3]}
        separate = {name: coefficient(source[name][fit_indices], target[name][fit_indices]) for name in HIGHER}
        # 当前真实数据若能量为零，相关拟合保持不适用并明确终止依赖预测。
        assert joint['value'] is not None and all(item['value'] is not None for item in [*singles.values(), *separate.values()])
        predictions = {method: {} for method in ('singleton_only', 'source_unit', 'joint_scalar', 'separate_scalar', 'constant_mean')}
        for name in HIGHER:
            x = source[name][check_indices]
            predictions['singleton_only'][name] = np.zeros_like(x)
            predictions['source_unit'][name] = x
            predictions['joint_scalar'][name] = joint['value']*x
            predictions['separate_scalar'][name] = separate[name]['value']*x
            predictions['constant_mean'][name] = np.broadcast_to(target[name][fit_indices].mean(0), x.shape)
        for terms in predictions.values():
            terms['J'] = sum(terms[name] for name in HIGHER)
        target_result = dict(coefficients=dict(joint_scalar=joint, singleton=singles, separate_scalar=separate),
            zero_request_baseline_maximum_absolute_error=zero_errors,
            reconstruction_maximum_absolute_error=dict(source=source_error, native=target_error),
            higher_order={}, complete_response={}, per_document_predictions=[])
        for name in (*HIGHER, 'J'):
            target_result['higher_order'][name] = {}
            for method, terms in predictions.items():
                pred, y, x = terms[name], target[name][check_indices], source[name][check_indices]
                target_result['higher_order'][name][method] = summary(pred, y, x, weight)
                aggregated.setdefault(('higher_order', name, method), []).append((pred, y, x))
        full_predictions = {}
        for method in ('singleton_only', 'source_unit', 'joint_scalar', 'separate_scalar'):
            full_predictions[method] = {}
            for name, bits in zip(OPERATIONS[3:], BITS[3:]):
                pred = sum(singles[part]['value']*source[part][check_indices]
                    for part, bit in zip(OPERATIONS[:3], BITS[:3]) if bit & bits == bit)
                pred = pred+sum(predictions[method][part] for part, bit in zip(HIGHER, BITS[3:]) if bit & bits == bit)
                full_predictions[method][name] = pred
                j = OPERATIONS.index(name)
                y, x = target_effect[check_indices, j], source_effect[check_indices, j]
                target_result['complete_response'].setdefault(name, {})[method] = summary(pred, y, x, weight)
                aggregated.setdefault(('complete_response', name, method), []).append((pred, y, x))
        for position, index in enumerate(check_indices):
            row = dict(document_sha256=documents[index], label=int(labels[index]), gender=int(genders[index]),
                higher_order={}, complete_response={})
            for section, methods in [('higher_order', predictions), ('complete_response', full_predictions)]:
                for method, terms in methods.items():
                    row[section][method] = {}
                    for name, values in terms.items():
                        observed = target[name][index] if section == 'higher_order' else target_effect[index, OPERATIONS.index(name)]
                        error = values[position]-observed
                        entry = dict(vector_rmse=float(np.sqrt(np.mean(error**2))),
                            head_prediction=float(values[position]@weight), head_error=float(error@weight))
                        if args.save_full_vectors:
                            entry['pooled512'] = values[position].tolist()
                        row[section][method][name] = entry
            target_result['per_document_predictions'].append(row)
        target_result['cancellation'] = {phase: {family: dict(sum_component_squared_energy=float(sum(np.mean(values[name][indices]**2) for name in HIGHER)),
            joint_squared_energy=float(np.mean(values['J'][indices]**2))) for family, values in [('source', source), ('native', target)]}
            for phase, indices in [('fit', fit_indices), ('check', check_indices)]}
        result['targets'][str(seed)] = target_result
        result['inputs'].append(dict(run=str(run), array=str(path), sha256=digest(path), probe=str(probe_path), probe_sha256=digest(probe_path)))
    for (section, name, method), values in aggregated.items():
        pred, y, x = [np.concatenate([item[i] for item in values], axis=0) for i in range(3)]
        result['common'].setdefault(section, {}).setdefault(name, {})[method] = summary(pred, y, x, shared_weight)
    result['script_sha256'] = digest(Path(__file__))
    result['cpu_analysis_wall_seconds_before_serialization'] = time.perf_counter()-started
    args.output.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    lines = ['# 有限交互结构', '', '原64篇均为已经曝光的开发文档。固定SHA256规则选择8篇拟合，另外56篇检查；两个target共用划分。', '',
        '高阶项按Boolean Moebius公式精确计算。J为四个高阶项之和；逐项结果与J分别显示。pair完整响应使用拟合的singleton尺度和预测交互，full用于已校准请求的新文档预测。', '',
        '| Target | J尺度b | P尺度g | N尺度g | W尺度g |', '|---|---:|---:|---:|---:|']
    for seed, data in result['targets'].items():
        c = data['coefficients']
        lines.append('| '+' | '.join([seed, f"{c['joint_scalar']['value']:.6g}"]+[f"{c['singleton'][p]['value']:.6g}" for p in OPERATIONS[:3]])+' |')
    for section in ('higher_order', 'complete_response'):
        lines.extend(['', '## '+section, '', '| Target | 项 | 方法 | RMSE | nRMSE | cosine | head RMSE |', '|---|---|---|---:|---:|---:|---:|'])
        for seed, container in [*result['targets'].items(), ('共同', result['common'])]:
            for name, methods in container[section].items():
                for method, data in methods.items():
                    numbers = ['不适用' if data[k] is None else f'{data[k]:.6g}' for k in ('rmse','nrmse','cosine','head_rmse')]
                    lines.append('| '+' | '.join([seed,name,method,*numbers])+' |')
    lines.extend(['', '原量纲能量、交互抵消量、逐文档向量误差、head预测、拟合系数、输入hash及代数重构误差保存在同名JSON。512维预测由输入响应和已保存系数复算。source_unit与joint_scalar的信息量低于separate_scalar和constant_mean；后两者使用拟合文档的三个额外pair响应。'])
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), bytes=args.output.stat().st_size,
        coefficients={seed: data['coefficients'] for seed,data in result['targets'].items()}, common=result['common'])))


if __name__ == '__main__':
    main()
