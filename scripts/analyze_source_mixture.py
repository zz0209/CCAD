import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_observable_transfer import family_analysis, query_analysis


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def training_check(run, reference):
    current, original = read(run/'config.resolved.json'), read(reference/'config.resolved.json')
    assert read(run/'status.json')['status'] == 'PASS'
    fields = ['steps', 'batch_pairs', 'natural_batch_states', 'dictionary_lr', 'response_weight',
              'reconstruction_weight', 'training_seed', 'fit_pairs_per_task', 'target_seed', 'tasks',
              'target_checkpoint', 'natural_states', 'members_per_source', 'queries', 'primary_requests']
    matched = {key: current[key] == original[key] for key in fields}
    assert all(matched.values()), matched
    assert current['steps'] == 512
    with np.load(run/'training_schedule.npz') as now, np.load(reference/'training_schedule.npz') as old:
        arrays = {key: bool(np.array_equal(now[key], old[key])) for key in ['requests', 'rows', 'natural']}
        expected_rows = np.array([[old['rows'][(step*current['batch_pairs']+j) % len(old['rows'])]
                                  for j in range(current['batch_pairs'])] for step in range(current['steps'])])
        arrays['row_batches'] = bool(np.array_equal(now['row_batches'], expected_rows))
        assert all(arrays.values()), arrays
        source_schedule = now['source_seed_by_step'].tolist() if 'source_seed_by_step' in now else [current['source_seed']]*current['steps']
    assert read(run/'panel.json')['fit'] == read(reference/'panel.json')['fit']
    scales = read(run/'loss_scales.json')
    original_scales = read(reference/'loss_scales.json')
    source1_scales = scales['by_source'].get('1') if 'by_source' in scales else (scales if current['source_seed'] == 1 else None)
    if source1_scales is not None:
        assert all(source1_scales[key] == original_scales[key] for key in ['hidden', 'response'])
    return dict(run=run.as_posix(), reference=reference.as_posix(), matched_config_fields=matched,
                exact_schedule_arrays=arrays, fit_records_equal=True,
                sources={str(seed): dict(steps=source_schedule.count(seed),
                                        endpoint_steps=source_schedule[::2].count(seed),
                                        continuous_steps=source_schedule[1::2].count(seed)) for seed in sorted(set(source_schedule))},
                source1_normalization_equal=(True if source1_scales is not None else None), loss_scales=scales,
                training_source_calls=read(run/'training_source_calls.json') if (run/'training_source_calls.json').exists() else None,
                information='Each training source has its own normalization access; the additional source changes available supervision.')


def run_record(run):
    summary = read(run/'metrics.summary.json')
    fields = ['wall_seconds', 'process_cpu_seconds', 'peak_allocated_bytes', 'sequence_forwards', 'token_forwards']
    return dict(run=run.as_posix(), status=read(run/'status.json'),
                resource_use={key: summary[key] for key in fields},
                code_snapshot_hash=read(run/'manifest.json')['code_snapshot_hash'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs='+')
    parser.add_argument('--training-runs', type=Path, nargs='+', required=True)
    parser.add_argument('--reference-training', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--training-only', action='store_true')
    args = parser.parse_args()
    assert args.bootstrap > 0 and not args.output.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checks = [training_check(run, args.reference_training) for run in args.training_runs]
    mixture = next(item for item in checks if set(item['sources']) == {'1', '3'})
    source3 = next(item for item in checks if set(item['sources']) == {'3'})
    assert all(mixture['loss_scales']['by_source']['3'][key] == source3['loss_scales'][key]
               for key in ['hidden', 'response'])
    if args.training_only:
        reference_config = read(args.reference_training/'config.resolved.json')
        configs = [read(run/'config.resolved.json') for run in args.training_runs]
        assert all(not config.get('resume_checkpoints') for config in [reference_config, *configs])
        write(args.output, dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
              matched_training=checks, source3_normalization_equal=True, no_resumed_optimizer=True,
              reference_variants=reference_config['variants'],
              initialization_code_reading=dict(reference_source=(args.reference_training/'source_snapshot/scripts/train_grammar_member_program.py').as_posix(),
                  reference_lines=[194, 195, 200], current_lines=[563, 566, 574],
                  observation='Each variant loads the same initial target before creating a fresh AdamW optimizer. Program has 512 independent updates; earlier gain and whole variants do not initialize program.'),
              run_records=[run_record(run) for run in args.training_runs]))
        print(json.dumps(dict(output=args.output.as_posix(), training_checks='PASS')))
        return
    assert args.runs
    analyses, panels, identities = {}, [], []
    required = {'initial', 'readout_initial', 'source1_program', 'source3_program', 'mixture_program', 'whole_reference'}
    for run in args.runs:
        config = read(run/'config.resolved.json')
        source = str(config['source_seed'])
        assert source not in analyses and config['target_seed'] == 2
        assert config['steps'] == 0 and not config['variants']
        result = family_analysis(run, args.bootstrap, focal='mixture_program')
        assert required <= result['summary'].keys()
        result['per_query'] = query_analysis(run, list(result['summary']), args.bootstrap, focal='mixture_program')
        result['inference'] = 'Paired sentence resampling within three fixed grammars, one fixed source and one fixed target; all methods and requests share each draw.'
        output = args.output.parent/f'SOURCE{source}_TRANSFER_ANALYSIS.json'
        write(output, result)
        analyses[source] = dict(path=output.as_posix(), summary=result['summary'], comparisons=result['comparisons'])
        panels.append(read(run/'panel.json'))
        identities.append(config['evaluation_checkpoints'])
        print(json.dumps(dict(stage='SOURCE_ANALYSIS', source=source, output=output.as_posix(),
                             primary={method: value['primary']['mean'] for method, value in result['summary'].items()})), flush=True)
    assert set(analyses) == {'1', '4', '5'}
    assert all(panel == panels[0] for panel in panels)
    assert all(identity == identities[0] for identity in identities)
    methods = list(analyses['1']['summary'])
    overview = {source: {method: analyses[source]['summary'][method]['primary'] for method in methods} for source in analyses}
    output = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), evidence_level='controlled_development',
                  primary_decision='Evaluate absolute error on source4 and source5 separately, with source1 retention; source differences remain descriptive.',
                  bootstrap=args.bootstrap, fixed_target=2, analyses=analyses, primary=overview,
                  heldout_sources_mean={method: float(np.mean([overview[source][method]['mean'] for source in ['4', '5']])) for method in methods},
                  inference='Source4/5 share target2 and the same sentences. Their mean is descriptive; no independent-source confidence interval is inferred.',
                  matched_training=checks, same_evaluation_panel=True, same_evaluation_checkpoint_paths=True,
                  run_records=[run_record(run) for run in [*args.training_runs, *args.runs]],
                  analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(args.output, output)
    print(json.dumps(dict(output=args.output.as_posix(), heldout_sources_mean=output['heldout_sources_mean'])), flush=True)


if __name__ == '__main__':
    main()
