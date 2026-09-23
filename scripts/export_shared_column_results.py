from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/final_science_20260921_round04'
sys.path.insert(0, str(ROOT/'.aris/plot_runtime_v1'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def export_reuse(directory):
    receipt = directory/'PAPER_EXPORT.json'
    assert not receipt.exists(), receipt
    paths = [directory/'confirmation_analysis/CALIBRATION_REUSE.json',
             directory/'HUMAN_SHARED_RESPONSE.json', directory/'HUMAN_SHARED_UTILITY.json']
    grammar, response, utility = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    assert grammar['evidence_level'] == 'fresh_sentence_confirmation_fixed_dictionaries'
    assert response['fixed_targets'] and utility['fixed_targets']
    assert response['targets'] == utility['target_seeds'] == [4, 5]
    assert utility['classifier'] == 'retrained'
    assert grammar['responses']['bootstrap'] == response['bootstrap'] == utility['bootstrap'] == 2000
    paper = ROOT/'paper'
    outputs = []

    def save_table(name, columns, header, rows):
        path = paper/'tables'/f'{name}.tex'
        lines = [r'\begin{tabular}{'+columns+'}', r'\toprule', header+r' \\', r'\midrule']
        lines += [' & '.join(row)+r' \\' for row in rows]
        lines += [r'\bottomrule', r'\end{tabular}']
        path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
        outputs.append(path)

    methods = [('initial', 'Initial'), ('source_columns_gain', 'Shared gains'),
               ('source_columns_lr0001', 'Shared columns'), ('program', 'Target program'),
               ('readout_initial', 'Source readout')]
    rows = []
    for method, label in methods:
        values = [grammar['responses']['directions'][key]['primary'][method]['mean']
                  for key in ['s1_t4', 's1_t5', 's4_t5', 's5_t4']]
        overall = grammar['responses']['source_equal_primary'][method]
        rows.append([label]+[f'{v:.3f}' for v in values]+[
            f"{overall['mean']:.3f} [{overall['ci95'][0]:.3f},{overall['ci95'][1]:.3f}]"])
    save_table('calibration_reuse_grammar', 'lrrrrr',
        r'Method & $1\to4$ & $1\to5$ & $4\to5$ & $5\to4$ & Source mean [95\% CI]', rows)

    rows = []
    for source, result in grammar['mechanism']['by_source'].items():
        scores = result['overall_primary']
        rows.append([source]+[f"{scores[key]['rmse']:.3f}" for key in
            ['zero', 'training_target_margin_mean', 'hidden_common_error_prediction']])
    save_table('calibration_reuse_mechanism', 'lrrr',
        r'Source & Zero & Mean response error & $g^\top\mu$', rows)

    choices = [('source', 'Source explanation'), ('native', 'Fixed member relation'),
               ('input_tangent_budget', 'Initial'), ('input_shared_gain', 'Shared gains'),
               ('input_shared_columns', 'Shared columns'), ('input_program', 'Target program'),
               ('raw_reconstruction', 'Source readout')]
    rows = []
    for method, label in choices:
        scores = utility['results'][method]
        fidelity = response['methods'][method]['parts']['mean'] if method in response['methods'] else None
        rows.append([label, '---' if fidelity is None else f'{fidelity:.3f}',
            f"{100*scores['full']['profession']['mean']:.2f}",
            f"{100*scores['parts_mean']['profession']['mean']:.2f}",
            f"{100*scores['parts_mean']['worst_group']['mean']:.2f}",
            f"{100*utility['pronouns_minus_names'][method]['mean']['profession']['mean']:.2f}"])
    save_table('calibration_reuse_consumer', 'lrrrrr',
        r'Method & Part error & Full accuracy & Part accuracy & Part WG & Pronoun--name', rows)

    data_path = paper/'data/calibration_reuse.json'
    data_path.write_text(json.dumps(dict(grammar=grammar, human_response=response, human_utility=utility),
                                   indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    outputs.append(data_path)

    def identity(path):
        return dict(path=path.as_posix(), bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    provenance = dict(sources=[identity(path) for path in paths], generator=identity(Path(__file__)),
                      outputs=[identity(path) for path in outputs])
    manifest_path = paper/'data/DATA_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['calibration_reuse'] = provenance
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    index_path = paper/'EVIDENCE_INDEX.json'
    index = json.loads(index_path.read_text())
    claim_id = 'shared_calibration_complete_use'
    index['claims'] = [claim for claim in index['claims'] if claim['id'] != claim_id]
    index['claims'].append(dict(id=claim_id, paper='Appendix app:shared_columns',
        result='Frozen shared calibration across source and target dictionaries, common execution error prediction, and later classifier use of the same saved human correction.',
        evidence=provenance['sources']+provenance['outputs'], current_manuscript=True, main_argument=False,
        statistics=dict(grammar=grammar['responses']['resampling'], human=utility['inference'])))
    if claim_id not in index['current_manuscript_claim_ids']:
        index['current_manuscript_claim_ids'].append(claim_id)
    index['manifest_identities']['data_manifest'] = identity(manifest_path)
    index['manuscript']['current_round_report'] = (directory/'REPORT.md').as_posix()
    index['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    index_path.write_text(json.dumps(index, indent=2)+'\n')
    receipt.write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(dict(outputs=[path.as_posix() for path in outputs])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-directory', type=Path)
    args = parser.parse_args()
    if args.reuse_directory is not None:
        export_reuse(args.reuse_directory)
        return
    studies = [json.loads((OUT/f'SHARED_COLUMN_{s}_CONFIRMATION_ANALYSIS.json').read_text())
               for s in ['HUMAN', 'INFINITIVE']]
    assert all(s['freeze_verified'] for s in studies)
    choices = [('initial', 'native', 'Fixed relation', 0),
               ('input_initial', 'native_tangent_relation_8', 'Encoder columns', 0),
               ('raw_reconstruction', 'raw_reconstruction', 'Source readout', 0),
               ('source_columns_shared_gain', 'source_columns_shared_gain', 'Shared gains', 0),
               ('source_columns_shared', 'source_columns_shared', 'Shared columns', 0),
               ('tangent_gain', 'tangent_gain', 'Target gains', 512),
               ('tangent_mixed', 'tangent_mixed', 'Target program', 512)]
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'],
                         'mathtext.fontset': 'stix', 'font.size': 8, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), sharey=True)
    rows = []
    for k, (ax, study, title) in enumerate(zip(axes, studies, ['Human explanation', 'Grammatical explanation'])):
        upper = 0
        for i, choice in enumerate(choices):
            result = study['summary'][choice[k]]['held_requests']
            v, (lo, hi) = result['nrmse'], result['interval']
            color = '#236e60' if choice[2] == 'Shared columns' else '#505050'
            upper = max(upper, hi)
            ax.plot([lo, hi], [i, i], color=color, lw=1.1)
            ax.plot(v, i, 'D' if choice[2] == 'Shared columns' else 'o', color=color, ms=4)
            for j, value in enumerate(result['by_seed']):
                ax.plot(value, i+.15, marker=['o', 's', '^'][j], color=color, ms=2.3, alpha=.55)
            for family in ['endpoints', 'held_requests', 'member_subsets']:
                r = study['summary'][choice[k]][family]
                rows.append(dict(study=study['setting'], method=choice[k], label=choice[2],
                                 target_response_updates=choice[3], family=family,
                                 nrmse=r['nrmse'], lower=r['interval'][0], upper=r['interval'][1],
                                 by_seed=json.dumps(r['by_seed'])))
        ax.set_xlim(0, upper*1.1)
        ax.set_title(title, fontsize=9, fontweight='normal')
        ax.set_xlabel('Response error (nRMSE)')
        ax.axhline(4.5, lw=.5, color='#b5b5b5')
        ax.spines[['left', 'top', 'right']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color='#e1e1e1', lw=.4)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(choices)), [c[2] for c in choices])
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=.20, right=.985, bottom=.20, top=.87, wspace=.22)
    for ext in ['pdf', 'png']:
        fig.savefig(ROOT/f'paper/figures/shared_columns.{ext}', dpi=230)
    plt.close(fig)
    table = [r'\begin{tabular}{lrrrrrrr}', r'\toprule',
             r'& Target & \multicolumn{3}{c}{Human explanation} & \multicolumn{3}{c}{Grammar} \\',
             r'Execution & updates & Parts & Participation & Members & Parts & Participation & Members \\', r'\midrule']
    for choice in choices:
        vals = [studies[k]['summary'][choice[k]][f]['nrmse'] for k in range(2)
                for f in ['endpoints', 'held_requests', 'member_subsets']]
        table.append(choice[2]+' & '+str(choice[3])+' & '+' & '.join(f'{v:.3f}' for v in vals)+r' \\')
    table += [r'\bottomrule', r'\end{tabular}']
    (ROOT/'paper/tables/shared_columns.tex').write_text('\n'.join(table)+'\n')
    with (ROOT/'paper/data/shared_columns.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ROOT/'paper/data/shared_columns.json').write_text(json.dumps(dict(human=studies[0], infinitive=studies[1]), indent=2)+'\n')
    print('Exported shared-column response results with all controls and per-target values.')


if __name__ == '__main__':
    main()
