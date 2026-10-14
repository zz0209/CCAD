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
DR = {'gemma': 'CIRC_DR_GEMMA_CLT426K_TOK_v1', 'llama': 'CIRC_DR_LLAMA_CLT524K_TOK_v1',
      'gemma3': 'CIRC_DR_G3_CLT262K_RES_TOK_v1'}
EDGES = RUNS / 'CIRC_EDGE_GEMMA_CLT426K_v1' / 'EDGE_REFEREE_ROWS.json'
TOY = RUNS / 'CIRC_TOY_REFEREE_FORMAL_v3' / 'TOY_REFEREE.json'
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
    # Known circuits, final formal run on held-out seeds: the share of the claimed later writes of the CLT features of
    # each circuit that removing their input takes out of the model (filled) and the bounded share S (open), as
    # medians with quartiles over features and inputs.
    s = json.loads(TOY.read_text())['summary']
    rows = [('A two-step', 'Two-step circuit, collapsed'), ('B amplified', 'Amplified over layers'),
            ('F erased copy', 'Read from a surviving copy'), ('E parallel', 'Same feature at every layer'),
            ('G both copies', 'Both stored copies read'), ('D redundant', 'Read from an unused copy'),
            ('C constant', 'Constant write')]
    fig, ax = plt.subplots(figsize=(3.05, 1.95))
    for k, (key, label) in enumerate(rows):
        y = len(rows) - 1 - k
        for cond, dy, face in [('removed_u', .12, INK), ('bounded_u', -.12, 'white')]:
            med, q25, q75 = s[key][cond][:3]
            ax.plot([q25, min(q75, 1.1)], [y + dy, y + dy], color=INK, lw=.7)
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


def profile():
    # Intervention profiles: for CLT units read at layers 0-1 (outside the always-active features) and later in
    # Gemma-2 and Llama, and for the CLT features of known circuits of the toy model (second formal run), the median
    # share of the claimed later writes removed when the unit's input is removed, when every feature active at the site
    # is switched off, and the share of the writes removed by the counterfactual input that restoring the unit's read
    # projection brings back, with quartiles.
    toy = json.loads((RUNS / 'CIRC_TOY_REFEREE_FORMAL_v3' / 'TOY_REFEREE.json').read_text())['summary']
    real = json.loads((CIRC / 'SUFFICIENCY.json').read_text())
    rows = [('Toy: amplified', toy['B amplified'], 'toy'), ('Toy: every layer', toy['E parallel'], 'toy'),
            ('Toy: two-step', toy['A two-step'], 'toy'), ('Toy: both copies read', toy['G both copies'], 'toy'),
            ('Toy: copy unused later', toy['D redundant'], 'toy'),
            ('Gemma-2, read later', real['gemma']['later'], LATE), ('Llama, read later', real['llama']['later'], LATE),
            ('Gemma-2, read at 0-1', real['gemma']['early_other'], EARLY),
            ('Llama, read at 0-1', real['llama']['early_other'], EARLY)]
    cols = [('removed_u', 'Input\nremoved'), ('removed_joint', 'All site\nfeatures off'),
            ('sufficiency', 'Read restored\nafter replacement')]
    fig, axes = plt.subplots(1, 3, figsize=(3.1, 2.05), sharey=True)
    for ax, (key, title) in zip(axes, cols):
        for k, (label, d, kind) in enumerate(rows):
            y = len(rows) - 1 - k
            med, q25, q75 = d[key][:3]
            color = INK if kind == 'toy' else kind
            face = 'white' if kind == 'toy' else kind
            ax.plot([max(q25, -.1), min(q75, 1.1)], [y, y], color=color, lw=.7)
            ax.scatter([float(np.clip(med, -.1, 1.1))], [y], s=12, facecolor=face, edgecolor=color, lw=.7, zorder=3)
        ax.axhspan(-.5, 3.5, color='#f1ebe0', lw=0, zorder=0)
        ax.axvline(0, color='.6', lw=.5, ls=':')
        ax.set_xlim(-.12, 1.12)
        ax.set_xticks([0, .5, 1])
        ax.set_xticklabels(['0', '.5', '1'], fontsize=6.2)
        ax.set_title(title, fontsize=6.6, pad=2, linespacing=1.0)
        clean(ax)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[0] for r in rows][::-1], fontsize=6.4)
    axes[1].set_xlabel('Median share with quartiles', fontsize=6.8)
    fig.subplots_adjust(left=.36, right=.98, top=.86, bottom=.17, wspace=.14)
    save(fig, 'gt_profile')


def method():
    # Schematic of the unit test: a CLT unit read at layer l and position p claims writes c_k into the MLP outputs at p;
    # removing its input changes the model's MLP outputs by dm_k, whose projection on c_k is the part of the claim the
    # model removes. A small inset shows one layer's projection.
    fig = plt.figure(figsize=(3.1, 1.5))
    ax = fig.add_axes([0, 0, .66, 1])
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 5)
    ax.axis('off')
    xr = 1.45
    ax.plot([xr, xr], [.25, 4.35], color='.6', lw=1.4, zorder=1)
    ax.text(xr - .15, 4.45, 'residual stream at $p$', ha='left', va='bottom', fontsize=7.2, color='.35')
    ys = [.95, 2.2, 3.45]
    for y, lab in zip(ys, ['$l$', '$l{+}1$', '$l{+}2$']):
        ax.add_patch(plt.Rectangle((1.85, y - .32), 1.55, .64, facecolor='white', edgecolor=INK, lw=.6, zorder=2))
        ax.text(2.62, y, 'MLP ' + lab, ha='center', va='center', fontsize=7.2, zorder=3)
        ax.annotate('', xy=(1.85, y - .12), xytext=(xr, y - .45), arrowprops=dict(arrowstyle='-|>', lw=.5, color='.45'))
        # Model's actual change of this MLP output (solid) above the claimed write (dashed).
        ax.annotate('', xy=(4.75, y + .14), xytext=(3.4, y + .14), arrowprops=dict(arrowstyle='-|>', lw=.8, color=INK))
        ax.text(4.85, y + .14, '$\\Delta m_k$', fontsize=7.6, va='center')
    ax.text(2.62, 3.85, '$\\vdots$', ha='center', fontsize=8.5)
    # The unit reads the input of MLP l and claims writes into MLP l and every later MLP output.
    ax.scatter([.6], [.95], s=70, color=CONFIRM(.15), edgecolor=INK, lw=.6, zorder=4)
    ax.annotate('', xy=(.75, .88), xytext=(xr - .02, .55), arrowprops=dict(arrowstyle='-|>', lw=.6, color=INK))
    ax.text(.6, 1.35, 'unit', ha='center', fontsize=7.2)
    ax.text(.15, .12, 'reads $x$', fontsize=7.0, color='.3')
    for y in ys:
        ax.annotate('', xy=(4.75, y - .2), xytext=(.78, 1.0), arrowprops=dict(
            arrowstyle='-|>', lw=.75, ls=(0, (3, 2)), color=LATE, connectionstyle='arc3,rad=-.15'))
        ax.text(4.85, y - .22, '$c_k$', fontsize=7.6, va='center', color=LATE)
    ax.plot([1.05, 1.22], [.6, .77], color=EARLY, lw=1.5)
    ax.plot([1.05, 1.22], [.77, .6], color=EARLY, lw=1.5)
    ax.text(.05, 2.0, 'remove\ninput', fontsize=7.2, color='.25', linespacing=1.0)
    # Inset: the share of a claimed write that the model removes is the projection of -dm_k on c_k.
    ix = fig.add_axes([.67, .1, .32, .8])
    ix.set_xlim(-.1, 1.3)
    ix.set_ylim(-.35, 1.05)
    ix.axis('off')
    ix.annotate('', xy=(1.0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', lw=1.1, color=LATE))
    ix.text(1.04, .02, '$c_k$', fontsize=8, color=LATE, va='center')
    ix.annotate('', xy=(.6, .62), xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', lw=1.1, color=INK))
    ix.text(.02, .55, '$-\\Delta m_k$', fontsize=8)
    ix.plot([.6, .6], [.62, 0], color='.5', lw=.6, ls=':')
    ix.plot([0, .6], [-.08, -.08], color=GEMMA, lw=1.8)
    ix.text(.3, -.26, '$\\pi_k$', fontsize=8, color=GEMMA, ha='center')
    ix.text(-.05, .95, 'at layer $k$', fontsize=7.2, color='.35')
    save(fig, 'gt_method')


def dose():
    # Dose-response: CLT units binned by the share of their own-layer write that removing their input takes out, against
    # the median share of their later writes taken out, for units read at layers 0-1 and later, per model.
    d = json.loads((CIRC / 'DOSE_RESPONSE.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(3.1, 1.45), sharey=True)
    for ax, (m, name, _) in zip(axes, MODELS):
        for group, color, marker in [('early', EARLY, 'o'), ('later', LATE, 's')]:
            pts = [b for b in d[m][group] if b['units'] >= 5]
            xs = [b['own'] for b in pts]
            ax.plot(xs, [b['later'][0] for b in pts], color=color, lw=.9, marker=marker, ms=2.8, zorder=3)
            ax.vlines(xs, [b['later'][1] for b in pts], [b['later'][2] for b in pts], color=color, lw=.5, alpha=.7)
        ax.set_xlim(-.05, 1.15)
        ax.set_xticks([0, .5, 1])
        ax.set_xticklabels(['0', '.5', '1'], fontsize=6.2)
        ax.set_ylim(-.1, 1.1)
        ax.set_title(name, fontsize=7, pad=2)
        clean(ax)
    axes[0].set_yticks([0, .5, 1])
    axes[0].set_yticklabels(['0', '.5', '1'])
    axes[0].set_ylabel('Later writes removed', fontsize=6.8)
    axes[1].set_xlabel('Own-layer write removed (dose)', fontsize=6.8)
    handles = [Line2D([], [], color=EARLY, marker='o', ms=2.8, lw=.9, label='Read at layers 0-1'),
               Line2D([], [], color=LATE, marker='s', ms=2.8, lw=.9, label='Read later')]
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, fontsize=6.1, bbox_to_anchor=(.55, 1.02),
               handletextpad=.3, columnspacing=1.)
    fig.subplots_adjust(left=.15, right=.98, top=.76, bottom=.27, wspace=.12)
    save(fig, 'gt_dose')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--figures', nargs='+', default=['overview', 'toy', 'method', 'dose', 'profile'])
    parser.add_argument('--preview', type=Path, default=None, help='also write PNG previews to this directory')
    args = parser.parse_args()
    ux_figures.PREVIEW = args.preview
    style()
    for name in args.figures:
        globals()[name]()


if __name__ == '__main__':
    main()
