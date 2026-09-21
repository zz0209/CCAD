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
REUSE_METHODS = [('initial', 'Initial columns'), ('reuse_whole', 'Saved complete-request training'),
    ('reuse_program', 'Saved part-program training'), ('readout_initial', 'Source-direction readout'),
    ('reuse_readout', 'Readout from trained dictionary')]


def identity(path):
    return dict(path=path.as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def export_source_reuse(data, paper):
    edges = {(r['source_seed'], r['target_seed']) for r in data['inputs']}
    assert edges == {(s, t) for s in range(2, 6) for t in range(2, 6) if s != t}
    assert data['bootstrap'] == 2000
    summary, rows = data['summary'], []
    table = [r'\begin{tabular}{lrrrr}', r'\toprule',
        r'Execution & Full & Parts & New requests & New-request 95\% interval \\', r'\midrule']
    compact = [r'\begin{tabular}{lrrr}', r'\toprule', r'Frozen execution & Full & Parts & New \\', r'\midrule']
    for method, label in REUSE_METHODS:
        result = summary[method]
        values = [f"{result[family]['mean']:.3f}" for family in ['full', 'parts', 'primary']]
        lower, upper = result['primary']['ci']
        if method == 'reuse_program':
            label = r'\textbf{'+label+'}'
        table.append(label+' & '+' & '.join(values)+f' & [{lower:.3f}, {upper:.3f}]'+r' \\')
        if method != 'reuse_readout':
            compact_label = {'initial': 'Initial', 'reuse_whole': 'Complete-request training',
                'reuse_program': r'\textbf{Part-program training}', 'readout_initial': 'Source-direction readout'}[method]
            compact.append(compact_label+' & '+' & '.join(values)+r' \\')
        for family, cell in result.items():
            rows.append(dict(method=method, family=family, mean=cell['mean'], lower=cell['ci'][0],
                upper=cell['ci'][1], by_direction=json.dumps(cell['by_seed'])))
    table += [r'\bottomrule', r'\end{tabular}']
    compact += [r'\bottomrule', r'\end{tabular}']
    per_edge = [r'\begin{tabular}{rrrrrrr}', r'\toprule',
        r'Source & Target & Initial & Saved complete & Saved parts & Readout & Trained readout \\', r'\midrule']
    for source, target in sorted(edges):
        key = f's{source}_t{target}'
        cells = [f"{summary[method]['primary']['by_seed'][key]:.3f}" for method, _ in REUSE_METHODS]
        per_edge.append(f'{source} & {target} & '+' & '.join(cells)+r' \\')
    per_edge += [r'\bottomrule', r'\end{tabular}']
    values = {f'GPTReuse{label}{family.title()}': summary[method][family]['mean']
        for method, label in [('initial', 'Initial'), ('reuse_whole', 'Whole'), ('reuse_program', 'Program'),
            ('readout_initial', 'Readout'), ('reuse_readout', 'TrainedReadout')]
        for family in ['full', 'parts', 'primary']}
    for comparison, label in [('reuse_program_minus_initial', 'Initial'), ('reuse_program_minus_reuse_whole', 'Whole')]:
        cell = data['comparisons'][comparison]['primary']
        values.update({f'GPTReuseDelta{label}': cell['mean'], f'GPTReuseDelta{label}Lower': cell['ci'][0],
            f'GPTReuseDelta{label}Upper': cell['ci'][1]})
        differences = [node['summary']['reuse_program']['primary']-node['summary'][
            'initial' if label == 'Initial' else 'reuse_whole']['primary']
            for node in data['delete_one_shared_seed'].values()]
        values.update({f'GPTReuseNode{label}Lower': min(differences), f'GPTReuseNode{label}Upper': max(differences)})
    outputs = []
    for name, lines in [('grammar_source_reuse', table), ('grammar_source_reuse_compact', compact),
                        ('grammar_source_reuse_directions', per_edge)]:
        path = paper/'tables'/f'{name}.tex'
        path.write_text('\n'.join(lines)+'\n')
        outputs.append(path)
    path = paper/'tables/grammar_source_reuse_values.tex'
    path.write_text('\n'.join('\\newcommand{\\'+key+'}{'+f'{value:.3f}'.replace('0.', '.')+'}'
        for key, value in values.items())+'\n')
    outputs.append(path)
    path = paper/'data/grammar_source_reuse.json'
    path.write_text(json.dumps(data, indent=2)+'\n')
    outputs.append(path)
    path = paper/'data/grammar_source_reuse.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outputs.append(path)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--reuse-analysis', type=Path)
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
    extra_sources = []
    if args.reuse_analysis:
        reuse = json.loads(args.reuse_analysis.read_text())
        outputs.extend(export_source_reuse(reuse, paper))
        extra_sources = [identity(args.reuse_analysis), identity(args.reuse_analysis.parent/'EXECUTION_FREEZE.json')]
    provenance = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        sources=[identity(args.analysis), identity(args.analysis.parent/'EXECUTION_FREEZE.json')]+extra_sources,
        generator=identity(Path(__file__)), outputs=[identity(path) for path in outputs])
    manifest_path = paper/'data/DATA_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['grammar_program_confirmation'] = provenance
    if args.reuse_analysis:
        manifest['grammar_source_reuse'] = dict(sources=extra_sources, generator=identity(Path(__file__)),
            outputs=[identity(path) for path in outputs if 'grammar_source_reuse' in path.name])
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
    current_root = (args.reuse_analysis or args.analysis).parent
    if args.reuse_analysis:
        claim_id = 'frozen_execution_across_source_definitions'
        index['claims'] = [cell for cell in index['claims'] if cell['id'] != claim_id]
        index['claims'].append(dict(id=claim_id, paper='Main Section 3.5; app:grammar_source_reuse',
            result='Target parameters trained on source1 are reused with independently selected members from sources2–5 for the same three functions, with zero additional target updates.',
            evidence=extra_sources+[identity(path) for path in outputs if 'grammar_source_reuse' in path.name],
            statistics=reuse['inference'], current_manuscript=True, main_argument=True))
        for field in ['current_manuscript_claim_ids', 'main_argument_claim_ids']:
            if claim_id not in index[field]:
                index[field].append(claim_id)
    index['manuscript']['current_round_report'] = current_root.as_posix()+'/REPORT.md'
    index['manuscript']['replication_scope'] = 'Same member-column program in Pythia-70M and GPT2Medium; fixed supplied explanations and tasks, independent target initializations and fresh confirmation contexts. Later classifier utility retains its development-cohort identity.'
    if args.reuse_analysis:
        index['manuscript']['replication_scope'] += ' GPT2 source reuse keeps target parameters fixed across four additional source definitions of the same three functions; the shared four-seed network is a fixed inference cohort.'
    index['manuscript']['previous_index'] = current_root.as_posix()+'/pre_integration/EVIDENCE_INDEX.json'
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
