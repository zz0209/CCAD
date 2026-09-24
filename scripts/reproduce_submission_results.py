import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from intervention_prediction_scores import score_predictions


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def tabular(columns, header, rows):
    return '\n'.join([r'\begin{tabular}{' + columns + '}', r'\toprule',
                      header + r' \\', r'\midrule',
                      *[' & '.join(row) + r' \\' for row in rows],
                      r'\bottomrule', r'\end{tabular}']) + '\n'


def rebuild_tables(data):
    outputs = {}
    grammar = read_json(data / 'grammar_program_confirmation.json')
    names = [('initial', 'Initial columns'), ('gain', 'Calibrated columns'),
             ('whole', 'Complete-request training'), ('program', r'\textbf{Program training}'),
             ('readout_initial', 'Source-direction readout'),
             ('readout_program', 'Readout from trained dictionary')]
    rows = []
    for method, label in names:
        cells = grammar['summary'][method]
        lo, hi = cells['primary']['ci']
        rows.append([label, *[f"{cells[f]['mean']:.3f}" for f in ['full', 'parts', 'primary']],
                     f'[{lo:.3f}, {hi:.3f}]'])
    outputs['grammar_program_confirmation.tex'] = tabular('lrrrr',
        r'Execution & Full & Parts & New requests & New-request 95\% interval', rows)
    reuse = read_json(data / 'grammar_source_reuse.json')
    names = [('initial', 'Initial'), ('reuse_whole', 'Complete-request training'),
             ('reuse_program', r'\textbf{Part-program training}'),
             ('readout_initial', 'Source-direction readout')]
    rows = [[label, *[f"{reuse['summary'][method][f]['mean']:.3f}"
             for f in ['full', 'parts', 'primary']]] for method, label in names]
    outputs['grammar_source_reuse_compact.tex'] = tabular('lrrr', 'Frozen execution & Full & Parts & New', rows)

    fixed = read_json(data / 'human_program_adaptation.json')['results']
    names = [('geometry', 'Decoder geometry'), ('geometry_gain', 'Calibrated geometry'),
             ('native', 'Original member relation'), ('raw', 'Source-direction readout'),
             ('raw_reconstruction', 'Reconstruction readout'), ('whole', 'Complete-program fit'),
             ('random_parts', 'Random-member fit'), ('parts_relation', 'Part fit, fixed dictionary'),
             ('parts', 'Part fit, updated dictionary')]
    rows = [[label, *[f"{fixed[method][f]['response_nrmse']['mean']:.3f}"
             for f in ['all', 'parts', 'unseen_combinations']],
             f"{100 * fixed[method]['parts']['balanced_agreement']['mean']:.2f}"]
            for method, label in names]
    outputs['human_program_adaptation.tex'] = (
        r'\begin{tabular}{lrrrr}\toprule' + '\n' +
        r'& \multicolumn{3}{c}{Response nRMSE} & Part\\' + '\n' +
        r'Realization & All & Parts & Combinations & agreement\\\midrule' + '\n' +
        '\n'.join(' & '.join(row) + r' \\' for row in rows) + '\n' +
        r'\bottomrule\end{tabular}' + '\n')

    use = read_json(data / 'program_use_replication.json')['panels']['ADDITIONAL']['rows']
    rows = []
    for row in use:
        label = r'\textbf{' + row['label'] + '}' if row['method'] == 'input_program' else row['label']
        rows.append([label, '---' if row['response'] is None else f"{row['response']:.3f}",
                     *[f'{100 * row[key]:.2f}' for key in
                       ['full_accuracy', 'part_accuracy', 'part_worst_group', 'pronoun_name']]])
    outputs['program_use_main.tex'] = tabular('lrrrrr',
        'Method & Response error & Full accuracy & Part accuracy & Part WG & Pronoun--name', rows)

    program = read_json(data / 'program_confirmation.json')
    names = [('initial', 'native', 'Initial fixed relation'),
             ('head_mixed', 'mixed', 'Trained fixed relation'),
             ('input_initial', 'native_tangent_relation_8', 'Initial input-dependent relation'),
             ('tangent_gain', 'tangent_gain', 'Calibrated source columns'),
             ('tangent_mixed', 'tangent_mixed', 'Trained input-dependent relation'),
             ('raw_reconstruction', 'raw_reconstruction', 'Source-direction readout')]
    rows = []
    for human, infinitive, label in names:
        values = [program[study]['summary'][method][f]['nrmse']
                  for study, method in [('human', human), ('infinitive', infinitive)]
                  for f in ['endpoints', 'held_requests', 'member_subsets']]
        rows.append([label, *[f'{value:.3f}' for value in values]])
    extra = program['infinitive_readout_supplement']['summary']['raw_reconstruction_after_tangent_mixed']
    rows.append(['Readout from trained dictionary', '---', '---', '---',
                 *[f"{extra[f]['nrmse']:.3f}" for f in ['endpoints', 'held_requests', 'member_subsets']]])
    outputs['program_confirmation.tex'] = '\n'.join([
        r'\begin{tabular}{lrrrrrr}', r'\toprule',
        r'& \multicolumn{3}{c}{Human explanation} & \multicolumn{3}{c}{Grammatical explanation} \\',
        r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
        r'Execution & Endpoints & Participation & Members & Endpoints & Participation & Members \\',
        r'\midrule', *[' & '.join(row) + r' \\' for row in rows],
        r'\bottomrule', r'\end{tabular}']) + '\n'
    member = read_json(data / 'member_information.json')
    labels = [('initial', 'Initial program'), ('program', 'Trained member columns'),
              ('group_uniform', 'Uniform within groups'), ('source_norm_share', 'Source-contribution shares'),
              ('readout_initial', 'Source-direction readout'), ('readout_trained', 'Trained-dictionary readout')]
    rows = [[label, *[f"{member['summary'][method][family]['mean']:.3f}"
                     for family in ['member_subsets', 'participation', 'endpoints']]]
            for method, label in labels]
    outputs['member_information.tex'] = tabular('lrrr',
        'Execution & Member subsets & Participation & Endpoints', rows)
    return outputs


def recompute_grammar(data):
    index = read_json(data / 'raw/INDEX.json')
    checked = {}
    for study in index['studies']:
        reference = read_json(ROOT / study['analysis'])
        per_method = {}
        for record in study['runs']:
            panel = read_json(ROOT / record['panel'])
            queries = panel['query_order']
            groups = {task: [i for i, row in enumerate(panel['rows']) if row['task'] == task]
                      for task in reference['tasks']}
            families = reference['queries']
            with np.load(ROOT / record['responses'], allow_pickle=False) as values:
                for method in sorted(set(values.files) - {'source', 'none'}):
                    for family, names in families.items():
                        query_ids = [queries.index(name) for name in names]
                        scores = []
                        for ids in groups.values():
                            selection = np.ix_(query_ids, ids)
                            source = values['source'][selection]
                            clean = values['none'][selection]
                            target = values[method][selection]
                            energy = np.square(source - clean).sum()
                            if energy <= 0:
                                raise ValueError('A declared response family has no source energy')
                            scores.append(float(np.sqrt(np.square(target - source).sum() / energy)))
                        per_method.setdefault((method, family), []).append(float(np.mean(scores)))
        maximum = 0.0
        estimates = {}
        for (method, family), values in per_method.items():
            estimate = float(np.mean(values))
            expected = reference['summary'][method][family]['mean']
            difference = abs(estimate - expected)
            if difference > 1e-12:
                raise ValueError(f'Raw response recomputation differs: {study["name"]}/{method}/{family}')
            maximum = max(maximum, difference)
            estimates.setdefault(method, {})[family] = estimate
        checked[study['name']] = {'runs': len(study['runs']), 'maximum_absolute_difference': maximum,
                                  'estimates': estimates}
    return checked


def draw_figures(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
                         'mathtext.fontset': 'stix', 'font.size': 8, 'pdf.fonttype': 42})
    records = read_json(data / 'program_confirmation_figure.json')
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.65), sharey=True)
    labels = ['Initial columns', 'Calibrated columns', 'Program training', 'Source-direction readout']
    colors = ['#747474', '#276f64', '#77516f', '#111111']
    titles = ['Human explanation\nPythia-70M', 'Infinitive explanation\nPythia-70M', 'Agreement components\nGPT2Medium']
    for k, ax in enumerate(axes):
        cells = [row for row in records if row['study'] == k]
        for index, row in enumerate(cells):
            ax.plot([row['lower'], row['upper']], [index, index], color=colors[index], lw=.9)
            ax.plot(row['mean'], index, ['o', '^', 's', 'D'][index], color=colors[index], ms=4)
            ax.annotate(f"{row['mean']:.3f}", (row['mean'], index), xytext=(0, 6),
                        textcoords='offset points', ha='center', fontsize=7)
        ax.set_xscale('log')
        ax.set_xlim(min(row['lower'] for row in cells) * .72, max(row['upper'] for row in cells) * 1.25)
        ax.xaxis.set_major_locator(FixedLocator([.05, .1, .2, .5, 1, 2]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:g}'))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_title(titles[k], fontsize=9)
        ax.set_xlabel('Response nRMSE')
        ax.set_ylim(3.45, -.55)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', lw=.4, color='#dedede')
    axes[0].set_yticks(range(4), labels)
    fig.subplots_adjust(left=.255, right=.99, bottom=.20, top=.76, wspace=.34)
    fig.savefig(output / 'program_confirmation.pdf')
    plt.close(fig)

    full = read_json(data / 'full_part_checks.json')['human']
    example = full['examples'][0]
    display = read_json(data / 'full_part_display_example.json')
    if display['document_sha256'] != example['document_sha256']:
        raise ValueError('The saved illustrative document differs from the evaluated example')
    fig = plt.figure(figsize=(6.75, 2.62))
    fig.text(.01, .98, '(a) One biography, four feature deletions', va='top', fontsize=9)
    fig.text(.01, .83, '“' + display['excerpt'] + '”', fontsize=8.4, style='italic')
    for x, label in [(.01, 'Features deleted'), (.24, 'Source'), (.335, 'Geometry'), (.435, 'Member\nrelation')]:
        fig.text(x, .72, label, ha='left' if x < .1 else 'center', va='center', fontsize=8)
    for y, request, label in [(.56, 'full', 'All gender cues'), (.455, 'pronouns', 'Pronouns'),
                               (.35, 'names', 'Names'), (.245, 'associated_words', 'Associated words')]:
        fig.text(.01, y, label, fontsize=8)
        for x, value in [(.24, example['source'][request]), (.335, example['target']['geometry'][request]),
                         (.435, example['target']['native'][request])]:
            fig.text(x, y, 'Nurse' if value > 0 else 'Professor', ha='center', fontsize=8)
    fig.text(.01, .08, 'Edits act on SAE features; the input text stays fixed.', fontsize=8)
    fig.text(.52, .98, '(b) Part errors after the full-edit check passes', va='top', fontsize=9)
    ax = fig.add_axes([.705, .21, .235, .58])
    labels = ['Geometry', 'Geometry + gains', 'Member relation', 'Source-dir. readout']
    for index, method in enumerate(['geometry', 'geometry_gain', 'native', 'raw']):
        row = full['methods'][method]
        value = 100 * row['any_part_error']
        low, high = [100 * v for v in row['paired95']]
        color = '#216b57' if method == 'native' else '#78517b' if method == 'raw' else '#777777'
        ax.barh(3 - index, value, height=.48, color=color, alpha=.82)
        ax.errorbar(value, 3 - index, xerr=[[value - low], [high - value]], fmt='none', ecolor=color, capsize=2)
        ax.text(high + 1.3, 3 - index, f'{value:.1f}', va='center', fontsize=7.8)
    ax.set_yticks([3, 2, 1, 0], labels)
    ax.set_xlim(0, 35)
    ax.set_xlabel('At least one part wrong (%)')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    fig.text(.52, .035, 'Identical accepted cases; paired 95% intervals.', fontsize=8)
    fig.savefig(output / 'human_part_check.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT / 'paper/data')
    parser.add_argument('--figures', action='store_true')
    parser.add_argument('--raw-grammar', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    outputs = rebuild_tables(args.data)
    checked = []
    for name, contents in outputs.items():
        (args.output / name).write_text(contents, encoding='utf-8')
        original = (ROOT / 'paper/tables' / name).read_text(encoding='utf-8')
        if ''.join(contents.split()) != ''.join(original.split()):
            raise ValueError(f'Reconstructed table differs from manuscript table: {name}')
        checked.append(name)
    raw = args.data / 'intervention_prediction_example.npz'
    with np.load(raw, allow_pickle=False) as arrays:
        values = {key: arrays[key] for key in ['clean', 'source_full', 'target_full', 'source_parts', 'target_parts']}
        result = score_predictions(**values)
        result['method_names'] = arrays['method_names'].tolist()
        result['part_names'] = arrays['part_names'].tolist()
    (args.output / 'raw_prediction_recomputation.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    if args.figures:
        draw_figures(args.data, args.output)
    raw_grammar = recompute_grammar(args.data) if args.raw_grammar else None
    receipt = {'tables_recomputed_from_saved_numerical_exports': checked,
               'raw_prediction_cases': result['cases'], 'common_full_cases': result['common_full_cases'],
               'raw_input_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
               'raw_grammar_recomputation': raw_grammar,
               'scope': 'Tables use stored statistical estimates. The prediction example recomputes one target cohort from raw answers. Saved intervals are retained; no model training or bootstrap rerun is implied.'}
    (args.output / 'REPRODUCED.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
