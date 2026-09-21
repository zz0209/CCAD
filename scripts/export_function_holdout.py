import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    assert not args.receipt.exists()
    parts = ['verb', 'number', 'gender']
    studies, identities, coverage, own_function_errors, source_checks = {}, [], {}, {}, {}
    for part in parts:
        studies[part] = {}
        reference = None
        for suffix, label in [('', 'groups'), ('_MEMBER', 'members'),
                              ('_ISOLATED', 'isolated'), ('_FINITE', 'finite'),
                              ('_FINITE_FIT', 'finite_fit')]:
            path = args.directory/f'DEV_{part.upper()}{suffix}_ANALYSIS.json'
            data = json.loads(path.read_text())
            assert len(data['inputs']) == 1 and data['inputs'][0]['target_seed'] == 2
            studies[part][label] = data
            identities.append(dict(path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            run = Path(data['inputs'][0]['run'])
            panel = json.loads((run/'panel.json').read_text())
            selected = [i for i, row in enumerate(panel['rows']) if row['task'] == data['tasks'][0]]
            qi = panel['query_order'].index(part)
            with np.load(run/'responses.npz') as responses:
                source = responses['source'][qi, selected]
                if reference is None:
                    reference = source.copy()
                difference = float(np.max(np.abs(source-reference)))
                effect = source-responses['none'][qi, selected]
                source_checks[part+'_'+label] = dict(maximum_absolute_margin_difference=difference,
                    relative_rms_difference=float(np.linalg.norm(source-reference)/np.linalg.norm(effect)),
                    absolute_tolerance=1e-4)
                assert difference <= 1e-4, source_checks[part+'_'+label]
                tasks = json.loads((run/'config.resolved.json').read_text())['tasks']
                own_function_errors[part+'_'+label] = {}
                for function, task in zip(parts, tasks):
                    indices = [i for i, row in enumerate(panel['rows']) if row['task'] == task]
                    coordinate = panel['query_order'].index(function)
                    expected = responses['source'][coordinate, indices]
                    effect = expected-responses['none'][coordinate, indices]
                    energy = float(np.mean(effect**2))
                    if energy == 0:
                        assert label == 'isolated' and function != part
                        continue
                    own_function_errors[part+'_'+label][function] = {
                        method: float(np.sqrt(np.mean((responses[method][coordinate, indices]-expected)**2)/energy))
                        for method in responses.files if method not in ['none', 'source']}
            if label in ['groups', 'members']:
                with np.load(run/'training_schedule.npz') as schedule:
                    members = np.array(json.loads((run/'method_summary.json').read_text())['source_members'])
                    columns = np.where(np.isin(members, schedule['source_members']))[0]
                    requests = (schedule['member_requests'][:, columns] if 'member_requests' in schedule.files
                                else schedule['requests'][:, schedule['source_parts']])
                    coverage[part+'_'+label] = dict(request_rank=int(np.linalg.matrix_rank(requests)),
                        source_members=len(columns), updates=len(requests))
                    indices = (schedule['rows'].copy(), schedule['natural'].copy())
                    if label == 'groups':
                        group_indices = indices
                    else:
                        assert all(np.array_equal(a, b) for a, b in zip(group_indices, indices))
    selections = [
        ('groups', 'initial', 'Initial derivative columns'),
        ('groups', 'whole', 'Complete-request training'),
        ('groups', 'program', 'Part-request training'),
        ('members', 'program', 'Member-request training'),
        ('groups', 'readout_initial', 'Source-direction readout'),
        ('groups', 'readout_program', 'Readout after part training'),
        ('isolated', 'initial', 'Isolated explanation, initial'),
        ('isolated', 'program', 'Isolated explanation, part-trained'),
        ('finite', 'initial', 'Finite columns, initial'),
        ('finite', 'program', 'Finite columns, derivative-trained'),
        ('finite_fit', 'whole', 'Finite columns, complete-trained'),
        ('finite_fit', 'program', 'Finite columns, part-trained'),
        ('finite_fit', 'readout_program', 'Finite-trained readout')]
    lines = [r'\begin{tabular}{lrrr}', r'\toprule',
             r'Execution & Verb & Number & Gender \\', r'\midrule']
    for index, (setting, method, label) in enumerate(selections):
        if index in [6, 8]:
            lines.append(r'\midrule')
        values = [studies[part][setting]['summary'][method]['heldout']['mean'] for part in parts]
        lines.append(label+' & '+' & '.join(f'{value:.3f}' for value in values)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    paper = Path(__file__).resolve().parents[1]/'paper'
    table = paper/'tables/function_holdout_development.tex'
    table.write_text('\n'.join(lines)+'\n')
    output = paper/'data/function_holdout_development.json'
    output.write_text(json.dumps(dict(studies=studies, coverage=coverage,
        own_function_errors=own_function_errors, source_checks=source_checks,
        inputs=identities), indent=2)+'\n')
    args.receipt.write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        inputs=identities, coverage=coverage, source_checks=source_checks,
        outputs=[dict(path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                 for path in [table, output]]), indent=2)+'\n')
    print(json.dumps(coverage))


if __name__ == '__main__':
    main()
