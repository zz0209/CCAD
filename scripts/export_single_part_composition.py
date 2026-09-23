import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


METHODS = [('single_program', 'Single-part training'),
           ('mixed_program', 'Mixed-request training'),
           ('initial', 'Initial columns'),
           ('source_additive', 'Source-additive prediction'),
           ('readout_initial', 'Source-direction readout')]
COLUMNS = [('primary_binary4', 'response_nrmse'),
           ('strength4', 'response_nrmse'),
           ('primary_binary4', 'interaction_reconstruction_rmse')]


def identity(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path.resolve()), 'sha256': digest, 'bytes': path.stat().st_size}


def format_counts(values):
    unique = sorted(set(values))
    assert unique
    return str(unique[0]) if len(unique) == 1 else f'{unique[0]}--{unique[-1]}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='Root containing data/ and tables/ exports')
    args = parser.parse_args()
    results = json.loads(args.results.read_text(encoding='utf-8'))
    assert results['status'] == 'PASS', args.results
    data_path = args.output / 'data/single_part_composition.json'
    table_path = args.output / 'tables/single_part_composition.tex'
    assert not data_path.exists() and not table_path.exists(), args.output
    costs = {}
    for family in ['single', 'mixed']:
        per_run = []
        for pair in results['inputs']:
            run = pair[family]
            measured = run['costs']['source_measurement_cost.json']['content']
            program = measured['actual_calls']['program']
            per_run.append({'run': run['run'], 'target_seed': run['config']['target_seed'],
                            'distinct_fitting_pair_requests': measured['distinct_fitting_pair_requests'],
                            'actual_program_teacher_calls': sum(item['calls'] for item in program.values())})
        costs[family] = per_run
    count_text = {
        family: {key: format_counts([run[key] for run in rows]) for key in
                 ['distinct_fitting_pair_requests', 'actual_program_teacher_calls']}
        for family, rows in costs.items()}
    contrast = results['comparisons']['single_program_minus_mixed_program']['primary_binary4']
    lower, upper = contrast['ci95']
    caption = (
        r'\textbf{Composing requests after single-part supervision.} '
        'Binary and strength columns report response nRMSE. Interaction error is the '
        'unnormalized RMS difference between method and source interaction residuals; '
        'each residual subtracts that method\'s own single-part additive prediction. '
        'Values average over the fixed grammatical paradigms and target dictionaries. '
        f'The binary single-minus-mixed difference is {contrast["mean"]:.3f} '
        f'[{lower:.3f}, {upper:.3f}] with a paired 95\\% interval. '
        'Source-additive predicts source responses from measured single-part effects. '
        'Readout writes along the source decoder directions. '
        'Per target, single/mixed training uses '
        f'{count_text["single"]["distinct_fitting_pair_requests"]}/'
        f'{count_text["mixed"]["distinct_fitting_pair_requests"]} distinct fitting pair-request '
        'measurements and '
        f'{count_text["single"]["actual_program_teacher_calls"]}/'
        f'{count_text["mixed"]["actual_program_teacher_calls"]} actual program teacher calls.'
    )
    table = [r'\begin{table}[t]\centering\small', r'\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular}{lrrr}', r'\toprule',
             r'Method & Binary & Strength & Interaction \\', r'\midrule']
    for method, label in METHODS:
        values = [results['summary'][method][family][metric]['mean'] for family, metric in COLUMNS]
        assert all(math.isfinite(value) for value in values), method
        table.append(label + ' & ' + ' & '.join(f'{value:.3f}' for value in values) + r' \\')
    table += [r'\bottomrule', r'\end{tabular}', r'\caption{' + caption + '}',
              r'\label{tab:single_part_composition}', r'\end{table}']
    exported = {
        'written_at_utc': datetime.now(timezone.utc).isoformat(), 'results_identity': identity(args.results),
        'exporter_identity': identity(Path(__file__)), 'target_seeds': results['target_seeds'],
        'tasks': results['tasks'], 'samples_by_grammar': results['samples_by_grammar'],
        'families': results['families'], 'summary': results['summary'],
        'comparisons': results['comparisons'], 'noninferiority': results['noninferiority'],
        'per_request': results['per_request'], 'inference': results['inference'],
        'source_additive_role': results['source_additive_role'],
        'interaction_definition': results['interaction_definition'],
        'bootstrap': results['bootstrap'], 'bootstrap_seed': results['bootstrap_seed'],
        'cost_per_run': costs, 'cost_by_training_family': results['cost']['by_training_family'],
        'observed_run_totals': results['cost']['summed_observed_run_totals'],
        'original_assets': results['outputs'], 'caption_latex': caption,
        'table_methods': [method for method, _ in METHODS], 'table_columns': COLUMNS,
    }
    data_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text('\n'.join(table) + '\n', encoding='utf-8')
    exported['table_identity'] = identity(table_path)
    data_path.write_text(json.dumps(exported, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'data': identity(data_path), 'table': identity(table_path)}), flush=True)


if __name__ == '__main__':
    main()
