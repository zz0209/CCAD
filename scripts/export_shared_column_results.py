from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/final_science_20260921_round04'
sys.path.insert(0, str(ROOT/'.aris/plot_runtime_v1'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
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
