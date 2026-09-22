import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--capacity-analysis', type=Path, required=True)
    parser.add_argument('--confirmation', type=Path)
    parser.add_argument('--confirmation-trained', type=Path)
    parser.add_argument('--capacity-holdout', action='store_true')
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    assert not args.receipt.exists()
    paper = Path(__file__).resolve().parents[1]/'paper'
    inputs, adaptation, outputs = [], [], []
    lines = [r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'New function & Contexts & \multicolumn{2}{c}{New-function error} & \multicolumn{2}{c}{Old-function error} \\',
             r' & & Initial & Prior & Initial & Prior \\', r'\midrule']
    for part in ['verb', 'number', 'gender']:
        for count in [8, 32]:
            path = args.directory/f'DEV_{part.upper()}_N{count}_ANALYSIS.json'
            data = json.loads(path.read_text())
            assert len(data['studies']) == 1
            study = data['studies'][0]
            assert study['new_function'] == part and study['new_contexts'] == count
            assert study['source_seed'] == 1 and study['target_seed'] == 2
            adaptation.append(study)
            inputs.append(dict(path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            values = [study['summary'][m][endpoint] for endpoint in ['new', 'old']
                      for m in ['program_fresh', 'program_warm']]
            lines.append(part.title()+f' & {count} & '+' & '.join(f'{v:.3f}' for v in values)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    output = paper/'tables/function_adaptation_development.tex'
    output.write_text('\n'.join(lines)+'\n')
    outputs.append(output)

    development = json.loads(args.capacity_analysis.read_text())
    capacity = development
    trained_confirmation = None
    inputs.append(dict(path=args.capacity_analysis.as_posix(), sha256=hashlib.sha256(args.capacity_analysis.read_bytes()).hexdigest()))
    if args.confirmation:
        assert args.confirmation_trained is not None
        capacity = json.loads(args.confirmation.read_text())
        trained_confirmation = json.loads(args.confirmation_trained.read_text())
        assert capacity['summary'] == trained_confirmation['summary']
        assert capacity['inputs'] == trained_confirmation['inputs']
        for path in [args.confirmation, args.confirmation_trained]:
            inputs.append(dict(path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    selections = [('initial_member', 'Original, member capacity'), ('initial', 'Original, part capacity'),
                  ('trained_member', 'Saved program, member capacity'),
                  ('trained_capacity_only', 'Saved program, part capacity, same support'),
                  ('trained_part', 'Saved program, part capacity'),
                  ('program', 'Part-capacity program training'), ('readout_initial', 'Original source-direction readout'),
                  ('readout_program', 'Part-trained source-direction readout')]
    lines = [r'\begin{tabular}{lrrr}', r'\toprule', r'Execution & Complete & Parts & New requests \\', r'\midrule']
    for method, label in selections:
        if method not in capacity['summary']:
            continue
        values = [capacity['summary'][method][endpoint]['mean'] for endpoint in ['full', 'parts', 'primary']]
        lines.append(label+' & '+' & '.join(f'{v:.3f}' for v in values)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    output = paper/'tables/request_capacity.tex'
    output.write_text('\n'.join(lines)+'\n')
    outputs.append(output)
    checks = []
    for item in development['inputs']:
        run = Path(item['run'])
        config = json.loads((run/'config.resolved.json').read_text())
        if config['steps'] == 512:
            original = Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round08/RG08_GPT2_PROGRAM_DEV_T2_20260921')
            with np.load(original/'training_schedule.npz') as old, np.load(run/'training_schedule.npz') as current:
                matched = {key: np.array_equal(old[key], current[key]) for key in ['requests', 'rows', 'natural']}
                assert all(matched.values()), matched
            checks.append(dict(run=run.as_posix(), reference=original.as_posix(), matched=matched))
    holdouts = {}
    if args.capacity_holdout:
        for part in ['verb', 'number', 'gender']:
            path = args.directory/f'CAPACITY_HOLDOUT_{part.upper()}_ANALYSIS.json'
            study = json.loads(path.read_text())
            assert study['queries']['heldout'] == [part] and len(study['inputs']) == 1
            run = Path(study['inputs'][0]['run'])
            previous = Path(f'D:/CCAD_Storage/runs/reuse_generalization_20260921_round10/RG10_FUNCTION_DEV_{part}_T2_20260921')
            with np.load(previous/'training_schedule.npz') as old, np.load(run/'training_schedule.npz') as current:
                matched = {key: np.array_equal(old[key], current[key]) for key in old.files}
                assert set(current.files)-set(old.files) == {'row_batches', 'member_requests'}
                config = json.loads((run/'config.resolved.json').read_text())
                expected_rows = np.array([[old['rows'][(step*config['batch_pairs']+j) % len(old['rows'])]
                    for j in range(config['batch_pairs'])] for step in range(config['steps'])])
                parts = np.array(json.loads((run/'method_summary.json').read_text())['source_parts'])
                matched['row_batches'] = np.array_equal(expected_rows, current['row_batches'])
                matched['member_requests'] = np.array_equal(old['requests'][:, parts], current['member_requests'])
                assert all(matched.values()), matched
            previous_analysis = args.directory.parent/'reuse_generalization_20260921_round10'/f'DEV_{part.upper()}_ANALYSIS.json'
            old_analysis = json.loads(previous_analysis.read_text())
            holdouts[part] = dict(analysis=study, previous_analysis=old_analysis, matched_schedule=matched)
            for source in [path, previous_analysis]:
                inputs.append(dict(path=source.as_posix(), sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    output = paper/'data/request_capacity_and_adaptation.json'
    output.write_text(json.dumps(dict(adaptation=adaptation, capacity=capacity, capacity_development=development,
        trained_confirmation=trained_confirmation, capacity_holdouts=holdouts, matched_schedules=checks, inputs=inputs), indent=2)+'\n')
    outputs.append(output)
    if args.confirmation:
        macros = {}
        for method, label in [('initial_member', 'InitialMember'), ('initial', 'InitialPart'),
                              ('trained_member', 'TrainedMember'), ('trained_part', 'TrainedPart'),
                              ('readout_initial', 'Readout')]:
            for family, family_label in [('primary', 'New'), ('full', 'Full'), ('parts', 'Parts')]:
                macros[label+family_label] = capacity['summary'][method][family]['mean']
        for analysis, contrast, label in [(capacity, 'initial_minus_initial_member', 'InitialDelta'),
                (trained_confirmation, 'trained_part_minus_trained_member', 'TrainedDelta')]:
            for family, family_label in [('primary', 'New'), ('full', 'Full'), ('parts', 'Parts')]:
                value = analysis['comparisons'][contrast][family]
                macros[label+family_label] = value['mean']
                macros[label+family_label+'Lower'], macros[label+family_label+'Upper'] = value['ci']
        output = paper/'tables/request_capacity_values.tex'
        output.write_text('\n'.join(r'\newcommand{\Cap'+name+'}{'+f'{value:.3f}'+'}' for name, value in macros.items())+'\n')
        outputs.append(output)
    manifest_path = paper/'data/DATA_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['request_capacity_and_adaptation'] = dict(generator='scripts/export_request_capacity.py',
        receipt=args.receipt.as_posix(), inputs=inputs,
        outputs=[dict(path=p.as_posix(), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in outputs])
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    args.receipt.write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(), inputs=inputs,
        matched_schedules=checks, outputs=[dict(path=p.as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in outputs]), indent=2)+'\n')
    print(json.dumps(dict(studies=len(adaptation), capacity_methods=list(capacity['summary']), matched_schedules=checks)))


if __name__ == '__main__':
    main()
