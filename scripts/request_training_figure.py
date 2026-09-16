"""Plot the matched request-family comparison from frozen program results.

The full nine-method/consumer display stays in the manuscript appendix.
This plot presents all four matched adaptation variants on identical axes.
No model execution or statistical re-estimation is performed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--paper', type=Path, default=Path('paper'))
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text(encoding='utf-8'))
    methods = ['whole', 'random_parts', 'parts_relation', 'parts']
    labels = ['Complete program', 'Random member subsets',
              'Named parts, fixed dictionary', 'Named parts, updated dictionary']
    families = ['parts', 'unseen_combinations']
    rows = []
    for method in methods:
        for family in families:
            cell = data['results'][method][family]['response_nrmse']
            rows.append(dict(method=method, request_family=family,
                             estimate=cell['mean'], lower=cell['ci95'][0], upper=cell['ci95'][1]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    with plt.rc_context({'font.family': 'Times New Roman', 'font.size': 8.5,
                         'mathtext.fontset': 'stix', 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.linewidth': .6}):
        fig, axes = plt.subplots(1, 2, figsize=(6.75, 1.95), sharey=True)
        fig.subplots_adjust(left=.33, right=.98, top=.82, bottom=.27, wspace=.20)
        upper = max(row['upper'] for row in rows) * 1.10
        for ax, family, title in zip(axes, families, ['(a) Named parts', '(b) Full and paired parts']):
            for i, method in enumerate(methods):
                row = next(row for row in rows if row['method'] == method and row['request_family'] == family)
                y = len(methods)-1-i
                color = '#216b57' if method == 'parts' else '#646464'
                marker = 's' if method == 'parts' else 'o'
                ax.errorbar(row['estimate'], y,
                            xerr=[[row['estimate']-row['lower']], [row['upper']-row['estimate']]],
                            fmt=marker, color=color, markersize=4.6, linewidth=.9, capsize=2)
            ax.set_xlim(0, upper)
            ax.set_ylim(-.45, 3.45)
            ax.set_yticks([3, 2, 1, 0], labels)
            ax.set_xticks([0, .1, .2, .3])
            ax.set_xlabel('Response nRMSE', labelpad=4)
            ax.set_title(title, fontsize=9, pad=8)
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.tick_params(axis='y', length=0)
            ax.grid(axis='x', color='.91', linewidth=.5)
            ax.set_axisbelow(True)
        for extension in ['pdf', 'svg', 'png']:
            fig.savefig(args.paper/f'figures/human_request_training.{extension}', dpi=240)
        plt.close(fig)
    with (args.paper/'data/human_request_training.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(source_path=str(args.analysis),
                    source_sha256=hashlib.sha256(args.analysis.read_bytes()).hexdigest(),
                    inference=data['inference'], methods=methods, families=families,
                    dimensions_inches=[6.75, 1.95], font='Times New Roman/STIX',
                    reference_display='figures/human_program_adaptation.pdf',
                    note='All matched adaptation variants. Full-program fitting includes full deletion; named-part fitting leaves every combination unfitted.',
                    values=rows)
    (args.paper/'data/human_request_training.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
