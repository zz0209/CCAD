import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


ROOT = Path(__file__).resolve().parents[1]
METHODS = [('initial', 'Initial columns'), ('gain', 'Calibrated columns'),
           ('whole', 'Complete-request training'), ('program', 'Program training'),
           ('readout_initial', 'Source-direction readout'),
           ('readout_program', 'Readout from trained dictionary')]


def identity(path):
    return dict(path=path.as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    assert not args.receipt.exists()
    data = json.loads(args.analysis.read_text())
    assert [r['target_seed'] for r in data['inputs']] == [2, 3, 4, 5]
    assert data['bootstrap'] == 2000
    paper = ROOT/'paper'
    summary = data['summary']
    rows = []
    table = [r'\begin{tabular}{lrrrr}', r'\toprule',
        r'Execution & Full & Parts & New requests & New-request 95\% interval \\', r'\midrule']
    for method, label in METHODS:
        result = summary[method]
        values = [f"{result[family]['mean']:.3f}" for family in ['full', 'parts', 'primary']]
        lower, upper = result['primary']['ci']
        if method == 'program':
            label = r'\textbf{'+label+'}'
        table.append(label+' & '+' & '.join(values)+f' & [{lower:.3f}, {upper:.3f}]'+r' \\')
        for family, current in result.items():
            rows.append(dict(method=method, family=family, mean=current['mean'],
                lower=current['ci'][0], upper=current['ci'][1], by_seed=json.dumps(current['by_seed'])))
    table += [r'\bottomrule', r'\end{tabular}']
    table_path = paper/'tables/grammar_program_confirmation.tex'
    table_path.write_text('\n'.join(table)+'\n')
    macro_values = {f'GPT{label}{family.title()}': summary[method][family]['mean']
        for method, label in [('initial', 'Initial'), ('gain', 'Gain'), ('whole', 'Whole'),
                              ('program', 'Program'), ('readout_initial', 'Readout'),
                              ('readout_program', 'TrainedReadout')]
        for family in ['full', 'parts', 'primary']}
    for comparison, label in [('program_minus_whole', 'Whole'), ('program_minus_gain', 'Gain')]:
        cell = data['comparisons'][comparison]['primary']
        macro_values.update({f'GPTDelta{label}': cell['mean'], f'GPTDelta{label}Lower': cell['ci'][0],
                             f'GPTDelta{label}Upper': cell['ci'][1]})
    values_path = paper/'tables/grammar_program_values.tex'
    values_path.write_text('\n'.join('\\newcommand{\\'+key+'}{'+f'{value:.3f}'.replace('0.', '.')+'}'
        for key, value in macro_values.items())+'\n')
    data_path = paper/'data/grammar_program_confirmation.json'
    data_path.write_text(json.dumps(data, indent=2)+'\n')
    csv_path = paper/'data/grammar_program_confirmation.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    original = [json.loads((ROOT/f'artifacts/science_upgrade_20260919/ROUND04_{name}_ANALYSIS.json').read_text())
                for name in ['HUMAN', 'INFINITIVE']]
    choices = [('input_initial', 'native_tangent_relation_8', 'initial', 'Initial columns', '#747474', 'o'),
        ('tangent_gain', 'tangent_gain', 'gain', 'Calibrated columns', '#276f64', '^'),
        ('tangent_mixed', 'tangent_mixed', 'program', 'Program training', '#77516f', 's'),
        ('raw_reconstruction', 'raw_reconstruction', 'readout_initial', 'Source-direction readout', '#111111', 'D')]
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix', 'font.size': 8, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.65), sharey=True)
    titles = ['Human explanation\nPythia-70M', 'Infinitive explanation\nPythia-70M', 'Agreement components\nGPT2Medium']
    figure_rows = []
    for k, (ax, title) in enumerate(zip(axes, titles)):
        endpoints = []
        for i, choice in enumerate(choices):
            if k < 2:
                cell = original[k]['summary'][choice[k]]['held_requests']
                value, (lower, upper) = cell['nrmse'], cell['interval']
            else:
                cell = summary[choice[k]]['primary']
                value, (lower, upper) = cell['mean'], cell['ci']
            endpoints += [lower, upper]
            ax.plot([lower, upper], [i, i], color=choice[4], lw=.9)
            ax.plot(value, i, choice[5], ms=4, color=choice[4], mfc='white' if i == 3 else choice[4])
            ax.annotate(f'{value:.3f}', (value, i), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=7)
            figure_rows.append(dict(study=k, method=choice[k], mean=value, lower=lower, upper=upper))
        ax.set_xscale('log')
        ax.set_xlim(min(endpoints)*.72, max(endpoints)*1.25)
        ax.xaxis.set_major_locator(FixedLocator([.05, .1, .2, .5, 1, 2]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:g}'))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_title(title, fontsize=9, fontweight='normal')
        ax.set_xlabel('Response nRMSE')
        ax.set_ylim(3.45, -.55)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', lw=.4, color='#dedede')
    axes[0].set_yticks(range(4), [choice[3] for choice in choices])
    fig.subplots_adjust(left=.255, right=.99, bottom=.20, top=.76, wspace=.34)
    figure_path = paper/'figures/program_confirmation.pdf'
    fig.savefig(figure_path)
    fig.savefig(args.receipt.parent/'program_confirmation.png', dpi=230)
    plt.close(fig)
    figure_data = paper/'data/program_confirmation_figure.json'
    figure_data.write_text(json.dumps(figure_rows, indent=2)+'\n')
    outputs = [table_path, values_path, data_path, csv_path, figure_path, figure_data]
    provenance = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        sources=[identity(args.analysis), identity(args.analysis.parent/'EXECUTION_FREEZE.json')],
        generator=identity(Path(__file__)), outputs=[identity(path) for path in outputs])
    manifest_path = paper/'data/DATA_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['grammar_program_confirmation'] = provenance
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    figure_manifest_path = paper/'figures/FIGURE_MANIFEST.json'
    figure_manifest = json.loads(figure_manifest_path.read_text())
    figure_manifest['outputs'] = [r for r in figure_manifest['outputs'] if r['path'] != 'figures/program_confirmation.pdf']
    figure_manifest['outputs'].append(dict(identity(figure_path), path='figures/program_confirmation.pdf'))
    figure_manifest['program_confirmation_inputs'] = [identity(figure_data), identity(args.analysis)]
    figure_manifest_path.write_text(json.dumps(figure_manifest, indent=2)+'\n')
    index_path = paper/'EVIDENCE_INDEX.json'
    index = json.loads(index_path.read_text())
    claim_id = 'cross_model_member_program_confirmation'
    index['claims'] = [cell for cell in index['claims'] if cell['id'] != claim_id]
    index['claims'].append(dict(id=claim_id,
        paper='Main Section 3 and Figure 2; app:grammar_program_confirmation',
        result='The common member executor and part-program training preserve new requests in GPT2Medium with TopK sources, compared with complete-request training and source-column gains.',
        evidence=provenance['sources']+provenance['outputs'],
        statistics=data['inference'], current_manuscript=True, main_argument=True))
    for field in ['current_manuscript_claim_ids', 'main_argument_claim_ids']:
        if claim_id not in index[field]:
            index[field].append(claim_id)
    index['manuscript']['current_round_report'] = args.analysis.parent.as_posix()+'/REPORT.md'
    index['manuscript']['replication_scope'] = 'Same member-column program in Pythia-70M and GPT2Medium; fixed supplied explanations and tasks, independent target initializations and fresh confirmation contexts. Later classifier utility retains its development-cohort identity.'
    index['manuscript']['previous_index'] = args.analysis.parent.as_posix()+'/pre_integration/EVIDENCE_INDEX.json'
    for claim in index['claims']:
        for record in claim.get('evidence', []):
            path = Path(record.get('path', ''))
            if path.as_posix().startswith('paper/sections/') and (ROOT/path).is_file():
                record.update(bytes=(ROOT/path).stat().st_size,
                    sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
    index['manifest_identities'] = dict(data_manifest=identity(manifest_path),
        figure_manifest=identity(figure_manifest_path))
    index['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    index_path.write_text(json.dumps(index, indent=2)+'\n')
    args.receipt.write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(dict(outputs=[str(path) for path in outputs])))


if __name__ == '__main__':
    main()
