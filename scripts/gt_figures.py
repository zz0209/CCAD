from pathlib import Path
import argparse
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np

from rc_figures import style
import ux_figures
from ux_figures import CIRC, GEMMA, GEMMA3, INK, LLAMA, READINGS, clean, save

sys.path.insert(0, str(Path(__file__).resolve().parent / 'circuits'))
from summarize_depth_referee import bounded  # noqa: E402

DESCRIPTION = 'Figures of the unit and edge test of attribution graphs, drawn from the saved runs and summaries.'
RUNS = Path('D:/CCAD_Storage/runs/circuits')
DR = {'gemma': 'CIRC_DR_GEMMA_CLT426K_v4', 'llama': 'CIRC_DR_LLAMA_CLT524K_v4', 'gemma3': 'CIRC_DR_G3_CLT262K_RES_v6'}
EDGES = RUNS / 'CIRC_EDGE_GEMMA_CLT426K_v1' / 'EDGE_REFEREE_ROWS.json'
TOY = RUNS / 'CIRC_TOY_REFEREE_FORMAL_v1' / 'TOY_REFEREE.json'
MODELS = [('gemma', 'Gemma-2', GEMMA), ('llama', 'Llama', LLAMA), ('gemma3', 'Gemma-3', GEMMA3)]
# Confirmation runs from a light sand (not produced) to a dark slate (produced), with luminance falling monotonically
# so that the scale reads in grayscale.
CONFIRM = LinearSegmentedColormap.from_list('confirm', ['#ead8b5', '#b9b39f', '#7e95a0', '#45667a', '#1d3446'])
EARLY, LATE = '#c9a66b', '#45667a'
ALWAYS = {'L0F7081', 'L1F3148', 'L1F3872'}


def dallas(ax):
    # The CLT's own graph on the Dallas prompt as an arc diagram: units by read layer, on the row of their token
    # ("Dallas" above, "is" below), colored by the bounded share of their claimed later writes that the model produces
    # from their input. The 16 edges with the largest claimed change are drawn as arcs below the "is" row (or as lines
    # from the "Dallas" row) and colored by the share of that change the model produces.
    graph = 'gemma-2-2b/gemma-fact-dallas-austin'
    units = json.loads((CIRC / 'EXAMPLE_DALLAS_UNITS.json').read_text())['units']
    rows = {9: .55, 10: 0.}
    sites = {}
    for u in units:
        sites.setdefault((u['layer'], u['position']), []).append(u)
    pos = {}
    for (l, p), us in sites.items():
        for k, u in enumerate(us):
            pos[u['unit']] = (l + (k - (len(us) - 1) / 2) * .42, rows[p])
    edges = [r for r in json.loads(EDGES.read_text())['rows'] if r['graph'] == graph
             and abs(r['claimed']) >= .1 * abs(r['v_pre']) and r['u'] in pos and r['v'] in pos]
    edges.sort(key=lambda r: -abs(r['claimed']))
    t = np.linspace(0, 1, 60)
    for r in edges[:16][::-1]:
        (x0, y0), (x1, y1) = pos[r['u']], pos[r['v']]
        c = float(np.clip(r['model'] / r['claimed'], 0, 1))
        if y0 == y1:
            xs, ys = x0 + (x1 - x0) * t, y0 - .055 * (x1 - x0) * np.sin(np.pi * t)
        else:
            xs, ys = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        ax.plot(xs, ys, color=CONFIRM(c), lw=.5 + .4 * min(abs(r['claimed']), 3), zorder=2, solid_capstyle='round')
    for u in units:
        x, y = pos[u['unit']]
        ax.scatter([x], [y], s=16, color=CONFIRM(u['bounded_u']), edgecolor=INK, lw=.5, zorder=3)
    for p, y in rows.items():
        ax.text(-1.2, y, {9: '"Dallas"', 10: '"is"'}[p], fontsize=7, ha='right', va='center')
    ax.annotate('always-active\nfeatures', xy=(.6, -.08), xytext=(3.2, -.62), fontsize=6.2, color='.3', ha='left',
                va='center', linespacing=1.0, arrowprops=dict(arrowstyle='-', color='.5', lw=.5))
    cax = ax.inset_axes([.62, .9, .3, .045])
    cb = plt.colorbar(plt.cm.ScalarMappable(cmap=CONFIRM), cax=cax, orientation='horizontal')
    cb.set_ticks([0, 1])
    cb.ax.tick_params(labelsize=6, length=1.5, width=.4, pad=1)
    cb.outline.set_linewidth(.4)
    ax.text(.6, .925, 'share produced', transform=ax.transAxes, fontsize=6.3, ha='right', va='center')
    ax.set_xlim(-1.6, 25.5)
    ax.set_ylim(-1.45, 1.0)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Read layer', fontsize=7.2)
    clean(ax)


def unit_shares(m):
    rows = json.loads((RUNS / DR[m] / 'DEPTH_REFEREE_ROWS.json').read_text())['rows']
    rows = [r for r in rows if len(r['claim_energy']) > 1 and sum(r['claim_energy'][1:]) > 0]
    early = [bounded(r, 'removed_u') for r in rows if r['layer'] <= 1]
    late = [bounded(r, 'removed_u') for r in rows if r['layer'] >= 2]
    return early, late


def units_panel(ax):
    # Per-unit bounded share of the claimed later writes produced from the unit's input, for CLT units read at layers
    # 0 and 1 and later, as boxes (quartiles, whiskers at the 10th and 90th percentiles) over jittered points.
    rng = np.random.default_rng(0)
    for k, (m, name, _) in enumerate(MODELS):
        for j, (vals, color) in enumerate(zip(unit_shares(m), [EARLY, LATE])):
            x = k + (j - .5) * .42
            v = np.array(vals)
            ax.scatter(x + rng.uniform(-.13, .13, len(v)), v, s=1.6, color=color, alpha=.45, lw=0, zorder=2)
            q10, q25, q50, q75, q90 = np.quantile(v, [.1, .25, .5, .75, .9])
            ax.add_patch(plt.Rectangle((x - .14, q25), .28, max(q75 - q25, .004), facecolor='none', edgecolor=INK,
                                       lw=.6, zorder=3))
            ax.plot([x - .14, x + .14], [q50, q50], color=INK, lw=1.2, zorder=4)
            ax.plot([x, x], [q10, q25], color=INK, lw=.5, zorder=3)
            ax.plot([x, x], [q75, q90], color=INK, lw=.5, zorder=3)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([n for _, n, _ in MODELS], fontsize=6.8)
    ax.set_xlim(-.6, len(MODELS) - .4)
    ax.set_ylim(-.03, 1.17)
    ax.set_yticks([0, .5, 1])
    ax.set_yticklabels(['0', '.5', '1'])
    ax.set_ylabel('Bounded share $S$ of later writes', fontsize=7.2)
    handles = [Line2D([], [], ls='', marker='s', ms=4, mfc=EARLY, mec=EARLY, label='Read at layers 0-1'),
               Line2D([], [], ls='', marker='s', ms=4, mfc=LATE, mec=LATE, label='Read later')]
    ax.legend(handles=handles, frameon=False, fontsize=6.1, loc='upper center', handletextpad=.2, borderaxespad=.0,
              ncol=2, columnspacing=.8, bbox_to_anchor=(.5, 1.03))
    clean(ax)


def edges_panel(ax):
    # Edges leaving CLT units read at layers 0 and 1, by the read layer of the downstream unit: the median share of
    # the claimed change of its pre-activation that the model produces when the upstream unit's input is removed, with
    # graph-bootstrap intervals.
    d = json.loads((CIRC / 'EDGE_DISTANCE_v2.json').read_text())
    bands = ['to layers 2-3', 'to layers 4-12', 'to layers 13+']
    for k, (m, name, color) in enumerate(MODELS):
        key = {'gemma': 'gemma-clt', 'llama': 'llama-clt', 'gemma3': 'g3-clt-res'}[m]
        xs, ys, lo, hi = [], [], [], []
        for j, b in enumerate(bands):
            if 'median' in d[key][b]:
                med, l, h = d[key][b]['median']
                xs.append(j + (k - 1) * .08)
                ys.append(med)
                lo.append(l)
                hi.append(h)
        ax.plot(xs, ys, color=color, lw=.9, marker='o', ms=3.2, label=name, zorder=3)
        ax.vlines(xs, lo, hi, color=color, lw=.7, zorder=2)
    ax.axhline(0, color='.6', lw=.5, ls=':')
    ax.set_xticks(range(3))
    ax.set_xticklabels(['2-3', '4-12', '13+'], fontsize=6.8)
    ax.set_xlabel('Read layer of the downstream unit', fontsize=7.2)
    ax.set_ylim(-.1, .8)
    ax.set_yticks([0, .25, .5, .75])
    ax.set_yticklabels(['0', '.25', '.5', '.75'])
    ax.set_ylabel('Claimed change produced', fontsize=7.2)
    ax.legend(frameon=False, fontsize=6.1, loc='upper center', handletextpad=.2, borderaxespad=.1,
              bbox_to_anchor=(.55, 1.0))
    clean(ax)


def overview():
    fig = plt.figure(figsize=(6.3, 2.35))
    a = fig.add_axes([.065, .2, .355, .7])
    b = fig.add_axes([.49, .2, .22, .7])
    c = fig.add_axes([.785, .2, .205, .7])
    dallas(a)
    units_panel(b)
    edges_panel(c)
    fig.text(.005, .985, '(a) The CLT graph for Dallas$\\to$Austin', fontsize=8, va='top')
    fig.text(.425, .985, '(b) Units of all graphs', fontsize=8, va='top')
    fig.text(.725, .985, '(c) Edges from layers 0-1', fontsize=8, va='top')
    save(fig, 'gt_overview')


def toy():
    # Known circuits, formal run on held-out seeds: the share of the claimed later writes of the CLT features of each
    # circuit that removing their input takes out of the model (filled) and the bounded share S (open), as medians
    # with quartiles over features and inputs.
    s = json.loads(TOY.read_text())['summary']
    rows = [('A two-step', 'Two-step circuit, collapsed'), ('B amplified', 'Amplified over layers'),
            ('F erased copy', 'Read from a surviving copy'), ('E parallel', 'Same feature at every layer'),
            ('D redundant', 'Read from an unused copy'), ('C constant', 'Constant write')]
    fig, ax = plt.subplots(figsize=(3.05, 1.75))
    for k, (key, label) in enumerate(rows):
        y = len(rows) - 1 - k
        for cond, dy, face in [('removed_u', .12, INK), ('bounded_u', -.12, 'white')]:
            med, q25, q75 = s[key][cond]
            ax.plot([q25, q75], [y + dy, y + dy], color=INK, lw=.7)
            ax.scatter([med], [y + dy], s=13, facecolor=face, edgecolor=INK, lw=.7, zorder=3)
    ax.axhspan(-.5, 1.5, color='#f1ebe0', lw=0, zorder=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for _, label in rows][::-1], fontsize=6.8)
    ax.set_xlim(-.1, 1.1)
    ax.set_xticks([0, .5, 1])
    ax.set_xticklabels(['0', '.5', '1'])
    ax.axvline(0, color='.6', lw=.5, ls=':')
    ax.axvline(1, color='.6', lw=.5, ls=':')
    ax.set_xlabel('Claimed later writes the model removes', fontsize=7.2)
    handles = [Line2D([], [], ls='', marker='o', ms=3.6, mfc=INK, mec=INK, label='Share removed'),
               Line2D([], [], ls='', marker='o', ms=3.6, mfc='white', mec=INK, label='Bounded share $S$')]
    ax.legend(handles=handles, frameon=False, fontsize=6.1, loc='lower right', handletextpad=.1, borderaxespad=.1)
    clean(ax)
    fig.subplots_adjust(left=.43, right=.98, top=.97, bottom=.22)
    save(fig, 'gt_toy')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--figures', nargs='+', default=['overview', 'toy'])
    parser.add_argument('--preview', type=Path, default=None, help='also write PNG previews to this directory')
    args = parser.parse_args()
    ux_figures.PREVIEW = args.preview
    style()
    for name in args.figures:
        globals()[name]()


if __name__ == '__main__':
    main()
