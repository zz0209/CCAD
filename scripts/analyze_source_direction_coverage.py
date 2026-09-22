import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PARTS = ['verb', 'number', 'gender']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    started = datetime.now(timezone.utc).isoformat()
    timer = time.perf_counter()
    torch.set_num_threads(2)
    inputs, dictionaries, spaces = {}, {}, {}

    def checked(path):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        assert path.is_file(), path
        if path.as_posix() not in inputs:
            inputs[path.as_posix()] = dict(bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return path

    def read(path):
        return json.loads(checked(path).read_text())

    def source(config):
        key = (config['source_checkpoint'], config.get('source_seed', 1))
        if key not in dictionaries:
            state = torch.load(checked(key[0]), weights_only=True, map_location='cpu')
            with np.load(checked(Path(config['source_run']) / f'topk_s{key[1]}_source.npz')) as asset:
                gate = asset['gate']
            assert gate.shape[1] == 3 and (gate.sum(1) <= 1).all()
            groups = [np.flatnonzero(gate[:, part] > 0) for part in range(3)]
            assert all(len(ids) == 64 for ids in groups)
            decoder = state['decoder.weight'].numpy().astype(np.float64)
            dictionaries[key] = [decoder[:, ids].copy() for ids in groups]
        return dictionaries[key]

    def space(columns, key):
        if key not in spaces:
            u, singular, _ = np.linalg.svd(columns, full_matrices=False)
            tolerance = max(columns.shape) * np.finfo(columns.dtype).eps * singular[0]
            rank = int((singular > tolerance).sum())
            basis = u[:, :rank]
            assert np.allclose(basis.T @ basis, np.eye(rank), atol=1e-10)
            assert np.linalg.norm(columns - basis @ (basis.T @ columns)) < 1e-9
            spaces[key] = (basis, dict(rank=rank, columns=columns.shape[1],
                singular_min=float(singular[rank-1]), singular_max=float(singular[0])))
        return spaces[key]

    def coverage(basis, columns):
        projected = basis @ (basis.T @ columns)
        fraction = np.square(columns-projected).sum(0) / np.square(columns).sum(0)
        assert np.isfinite(fraction).all() and (fraction >= 0).all() and (fraction <= 1+1e-10).all()
        return dict(mean_squared_relative_residual=float(fraction.mean()),
                    per_member_squared_relative_residual=fraction.tolist())

    reference = read('configs/rg10_grammar_function_holdout_development.json')
    training_source = source(reference)
    full_basis, full_info = space(np.concatenate(training_source, axis=1), 'three_functions')
    cases = []
    reuse = read('artifacts/reuse_generalization_20260921_round09/CONFIRMATION_ANALYSIS.json')
    holdout = [read(f'artifacts/reuse_generalization_20260921_round10/DEV_{part.upper()}_ANALYSIS.json')
               for part in PARTS]
    jobs = [('cross_source', entry, None) for entry in reuse['inputs']]
    jobs += [('function_holdout', analysis['inputs'][0], part) for part, analysis in zip(PARTS, holdout)]
    if args.smoke:
        jobs = [jobs[0], jobs[-3]]
    for index, (cohort, entry, omitted) in enumerate(jobs):
        run = Path(entry['run'])
        assert read(run / 'status.json')['status'] == 'PASS'
        config, panel = read(run / 'config.resolved.json'), read(run / 'panel.json')
        assert config['model_revision'] == reference['model_revision']
        assert config['hook_module_path'] == reference['hook_module_path']
        response_path = checked(run / 'responses.npz')
        assert inputs[response_path.as_posix()]['sha256'] == entry['response_sha256']
        with np.load(response_path) as data:
            values = {key: data[key].astype(np.float64) for key in data.files}
        orders = panel['query_order']
        assert values['source'].shape == (len(orders), len(panel['rows']))
        current_source = source(config)
        if omitted is None:
            basis, info = full_basis, full_info
        else:
            assert config['heldout_part'] == omitted
            chosen = [columns for part, columns in zip(PARTS, training_source) if part != omitted]
            basis, info = space(np.concatenate(chosen, axis=1), 'omit_'+omitted)
        for part_index, part in enumerate(PARTS):
            task = config['tasks'][part_index]
            ri = np.array([i for i, row in enumerate(panel['rows']) if row['task'] == task])
            qi = orders.index(part)
            ref, clean = values['source'][qi, ri], values['none'][qi, ri]
            denominator = np.square(ref-clean).sum()
            assert denominator > 0
            metrics = {method: float(np.sqrt(np.square(value[qi, ri]-ref).sum()/denominator))
                       for method, value in values.items() if method not in ['none', 'source']}
            program = 'reuse_program' if cohort == 'cross_source' else 'program'
            case = dict(cohort=cohort, run=run.as_posix(), source_seed=config.get('source_seed', 1),
                target_seed=config['target_seed'], part=part, omitted=omitted,
                trained_function=omitted != part, training_space=info,
                geometry=coverage(basis, current_source[part_index]), metrics=metrics,
                program_minus_initial=metrics[program]-metrics['initial'], contexts=len(ri))
            if omitted == part:
                expected = holdout[part_index]['summary']['program']['heldout']['mean']
                assert abs(expected-metrics[program]) < 1e-7
            cases.append(case)
        print(f'{index+1}/{len(jobs)} saved runs analyzed', flush=True)
    grouped = []
    keys = sorted({(c['cohort'], c['source_seed'], c['part'], c['omitted'] or '') for c in cases})
    for cohort, seed, part, omitted in keys:
        selected = [c for c in cases if (c['cohort'], c['source_seed'], c['part'], c['omitted'] or '')
                    == (cohort, seed, part, omitted)]
        fractions = [c['geometry']['mean_squared_relative_residual'] for c in selected]
        assert np.ptp(fractions) < 1e-12
        grouped.append(dict(cohort=cohort, source_seed=seed, part=part, omitted=omitted or None,
            rank=selected[0]['training_space']['rank'], targets=[c['target_seed'] for c in selected],
            squared_relative_residual=fractions[0],
            program_minus_initial=float(np.mean([c['program_minus_initial'] for c in selected]))))
    result = dict(run_id='FIELD_COVERAGE_13_SMOKE' if args.smoke else 'FIELD_COVERAGE_13',
        started_at_utc=started, ended_at_utc=datetime.now(timezone.utc).isoformat(),
        driver_seconds=time.perf_counter()-timer, status='PASS', environment=dict(python=sys.executable,
        numpy=np.__version__, torch=torch.__version__, threads=2),
        scope='Retrospective decoder-direction coverage and saved functional response analysis. '
              'Fixed functions and shared source-target seeds. Members are not independent samples. '
              'Each cohort uses its actual training span; cross-cohort rank differs. No new model outcomes.',
        inputs=inputs, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases=cases, grouped=grouped)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(grouped, indent=2))


if __name__ == '__main__':
    main()
