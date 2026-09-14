"""Paired effect intervals for equivalent-source learning and its translation."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    meta = json.loads((args.analysis / 'confirmation.json').read_text())
    path = args.analysis / 'cluster_outcomes.npz'
    with np.load(path) as z:
        outcomes, methods = z['outcomes'][:, 0, ..., 0], z['methods'].tolist()
    comparisons = [('source_views', 'source_original', 'Source: equivalent inputs vs. original'),
                   ('field_views', 'field_old', 'Field: new source vs. original source'),
                   ('field_views', 'assignment_views', 'Field vs. member assignment'),
                   ('field_views', 'direct_320', 'Field vs. direct gradient (320 batches)')]
    n = outcomes.shape[1]
    rng = np.random.default_rng(meta['spec']['bootstrap_seed'])
    draws = rng.integers(n, size=(meta['spec']['bootstrap_replicates'], n))
    cells = []
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    with plt.rc_context({'font.family': 'Times New Roman', 'font.size': 9, 'mathtext.fontset': 'stix',
                         'axes.linewidth': .6, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        fig, ax = plt.subplots(figsize=(7, 2.7))
        fig.subplots_adjust(left=.43, right=.98, bottom=.22, top=.87)
        ax.axvline(0, color='#777777', linewidth=.7, linestyle=':')
        for i, (a, b, label) in enumerate(comparisons):
            diff = outcomes[methods.index(a)] - outcomes[methods.index(b)]
            for op, index, offset, color, marker in [('both', None, -.19, '#222222', 'D'),
                    ('unit', 0, 0, '#286956', 'o'), ('tens', 1, .19, '#785481', 's')]:
                values = diff.mean((1, 2, 3)) if index is None else diff[:, index].mean((1, 2))
                mean = values.mean()*100
                lo, hi = np.quantile(values[draws].mean(1)*100, [.025, .975])
                ax.errorbar(mean, i+offset, xerr=np.array([[mean-lo], [hi-mean]]), fmt=marker,
                            color=color, markersize=4, linewidth=.8, capsize=2,
                            label={'both':'Both requests', 'unit':'Units', 'tens':'Tens'}[op] if i==0 else None)
                cells.append(dict(reference=a, comparator=b, operation=op,
                                  difference_points=mean, interval_points=[lo, hi]))
        ax.set(yticks=range(len(comparisons)), yticklabels=[c[2] for c in comparisons],
               xlabel='Difference in complete hybrid accuracy (percentage points)')
        ax.invert_yaxis()
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0, pad=9)
        ax.tick_params(axis='x', width=.6, length=3)
        ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False,
                  fontsize=8, columnspacing=1.1, handletextpad=.35)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for ext in ['pdf', 'svg', 'png']:
            fig.savefig(args.output.with_suffix('.'+ext), dpi=220, facecolor='white')
        plt.close(fig)
    args.output.with_suffix('.json').write_text(json.dumps(dict(cells=cells,
        data=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size_inches=[7, 2.7], fonts='Times New Roman and STIX',
        scope='Fresh question-pair cluster bootstrap conditional on the five fixed SAEs. Both-request field-versus-old and field-versus-assignment are primary; other comparisons and request breakdowns are prespecified diagnostics. All base failures retained.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
