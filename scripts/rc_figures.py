from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import blended_transform_factory
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts' / 'containment'
FIG = ROOT / 'paper' / 'figures'
DESCRIPTION = 'Data figures of the reader-coverage manuscript, drawn from saved containment analyses.'

# Error increases from paper white to a dark iron oxide, with luminance falling monotonically for grayscale reading.
ERROR = LinearSegmentedColormap.from_list('error', ['#fbf8f1', '#ead9b4', '#cf9a5a', '#a2512d', '#5c1f14'])
LATER = ['composer_surgeon_o0', 'composer_surgeon_o1', 'model_software_engineer_o0', 'model_software_engineer_o1']
MASSES = ['mass_pronoun', 'mass_female', 'mass_male']
MARGINS = ['margin_distractor_agreement_relational_noun', 'margin_distractor_agreement_relative_clause',
           'margin_irregular_plural_subject_verb_agreement_1', 'margin_regular_plural_subject_verb_agreement_2']


def style():
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    font_manager.fontManager.addfont('C:/Windows/Fonts/timesi.ttf')
    font_manager.fontManager.addfont('C:/Windows/Fonts/timesbd.ttf')
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 8, 'mathtext.fontset': 'stix',
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': .5})


def gemma_panel(run):
    pt = json.loads((ART / 'EXP2_GEMMA_PAPER.json').read_text())['runs'][run]['point']
    rows = [('hungarian_decoder', ('Hungarian matching', '', ''), None),
            ('natural_sup_hungarian', ('Generic text', 'Next token', 'Random'), 5),
            ('part_sup_hungarian', ('Biographies', 'Classifier', 'Parts'), 0),
            ('e4c_subset_sup_hungarian', ('Biographies', 'Classifier', 'Parts, random'), 0),
            ('e8a_random_sup_hungarian', ('Biographies', 'Classifier', 'Random'), 0),
            ('lw_subset_state_sup_hungarian', ('Biographies', 'Layer-22 states', 'Parts, random'), 4),
            ('lw_random_state_sup_hungarian', ('Biographies', 'Layer-22 states', 'Random'), 4)]
    cols = ['Original classifier', 'Held-out classifiers', 'Biography next token', 'Word-class mass',
            'Layer-22 states', 'Generic text']
    get = lambda m, r: pt[m][f'parts|{r}']
    values = np.array([[get(m, 'original'), np.mean([get(m, r) for r in LATER + ['gender']]), get(m, 'bio_next_token'),
                        np.mean([get(m, r) for r in MASSES]), get(m, 'state'), get(m, 'fineweb_next_token')]
                       for m, _, _ in rows])
    return [r[1] for r in rows], cols, values, [r[2] for r in rows]


def grammar_panel(family):
    held = json.loads((ART / f'GRAMMAR_READERS_{family}.json').read_text())['point']
    own = json.loads((ART / f'C8_{family}_RC.json').read_text())['point']
    if family == 'GPT2':
        own = dict(own, **{k: v for k, v in json.loads((ART / 'C7_GPT2_MATCHED4.json').read_text())['point'].items()})
    own_key = dict(hungarian_activation='hungarian_activation', membership_natural='membership_natural',
                   matched_natural4='matched_natural@merged' if family == 'GPT2' else 'matched_natural',
                   membership_parts='membership_parts', rc_distparts='membership_distparts@merged')
    rows = [('hungarian_activation', ('Hungarian matching', '', ''), None),
            ('membership_natural', ('Generic text', 'Next token', 'Random'), 3),
            ('membership_parts', ('BLiMP pairs', 'Margin', 'Parts'), 0),
            ('rc_distparts', ('BLiMP pairs', 'Next token', 'Parts'), 0)]
    cols = ['Own paradigms', 'Held-out margins', 'Held-out next token', 'Generic text']
    values = np.array([[own[own_key[m]]['parts'], np.mean([held[m]['parts'][r] for r in MARGINS]),
                        held[m]['parts']['heldout_next_token'], held[m]['parts']['fineweb_next_token']]
                       for m, _, _ in rows])
    return [r[1] for r in rows], cols, values, [r[2] for r in rows]


def heat(ax, title, cols, values, fits):
    shown = np.minimum(values, 1.)
    ax.imshow(shown, cmap=ERROR, vmin=0., vmax=1., aspect='auto')
    best = values.argmin(0)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            text = f'{v:.2f}'.lstrip('0') if v < 1 else f'{v:.2f}'
            ax.text(j, i, text, ha='center', va='center', fontsize=7, color='white' if shown[i, j] > .52 else '#1e1e1e',
                    fontweight='bold' if best[j] == i else 'normal')
        if fits[i] is not None:
            j = fits[i]
            ax.add_patch(Polygon([(j - .5, i - .5), (j - .2, i - .5), (j - .5, i - .2)], closed=True, color='#1e1e1e', lw=0))
    for j, i in enumerate(best):
        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec='#1e1e1e', lw=.9))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=7, rotation=50, ha='left', rotation_mode='anchor')
    ax.xaxis.tick_top()
    ax.set_yticks([])
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    # The rotated headers take about .8 inch above the cells; the panel title sits above them.
    height_in = ax.get_position().height * ax.figure.get_figheight()
    ax.text(0, 1 + .84 / height_in, title, transform=ax.transAxes, fontsize=8.2, ha='left', va='bottom')


def objective_labels(ax, rows, width_in):
    # Each fitting objective is described by the inputs it sees, the reader it matches and the requests it is trained on.
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    xs = [-2.04 / width_in, -1.36 / width_in, -.63 / width_in]
    for x, head in zip(xs, ['Inputs', 'Reader', 'Requests']):
        ax.text(x, -.72, head, transform=tr, fontsize=7, fontstyle='italic', ha='left', va='bottom', color='.35')
    for i, (inputs, reader, requests) in enumerate(rows):
        for x, t in zip(xs, [inputs, reader, requests]):
            ax.text(x, i, t, transform=tr, fontsize=7.2, ha='left', va='center')


def reader_matrix():
    fig = plt.figure(figsize=(6.3, 4.7))
    left, gap, width = .34, .025, .29
    w_in = width * 6.3
    top_h, bottom_h = 7 * .19 / 4.7, 4 * .19 / 4.7
    top_y, bottom_y = .495, .045
    panels = [((left, top_y, width, top_h), '(a) Gemma-2-2B SHIFT, 65k target',
               gemma_panel('CSM_EXP2_GEMMA2B_readers_w65k_v1_20261011')),
              ((left + width + gap, top_y, width, top_h), '(b) Gemma-2-2B SHIFT, 16k lowest-$L_0$ target',
               gemma_panel('CSM_EXP2_GEMMA2B_readers_l0low16k_v2_20261011')),
              ((left, bottom_y, width, bottom_h), '(c) GPT-2 Medium grammar', grammar_panel('GPT2')),
              ((left + width + gap, bottom_y, width, bottom_h), '(d) Qwen3-1.7B grammar', grammar_panel('QWEN'))]
    for k, (pos, title, (rows, cols, values, fits)) in enumerate(panels):
        ax = fig.add_axes(pos)
        heat(ax, title, cols, values, fits)
        if k % 2 == 0:
            objective_labels(ax, rows, w_in)
    save(fig, 'rc_reader_matrix')


PRONOUN = '#4f6f8f'
WOMEN = '#9a5b7a'
STATE = '#1d6b63'
CLASSIFIER = '#b0543f'
MATCH = '#7d7d7d'


def box(ax, x, y, w, h, text, ec='#3a3a3a', fc='white', size=7.2, style='normal'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.012', ec=ec, fc=fc, lw=.7))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=size, fontstyle=style, color='#1e1e1e')


def arrow(ax, a, b, color='#3a3a3a', lw=.7, style='-|>'):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=7, color=color, lw=lw, shrinkA=0, shrinkB=0))


def overview_schematic(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    src = [(.06, y) for y in [.70, .63, .56, .49]] + [(.06, y) for y in [.37, .30, .23]]
    colors = [PRONOUN] * 4 + [WOMEN] * 3
    tgt = [(.22, .73 - .075 * k) for k in range(8)]
    links = [(0, 0), (0, 1), (1, 1), (2, 2), (2, 3), (3, 3), (3, 6), (4, 4), (4, 5), (5, 5), (6, 6), (6, 7), (1, 4)]
    for i, j in links:
        ax.plot([src[i][0], tgt[j][0]], [src[i][1], tgt[j][1]], color=colors[i], lw=.8, alpha=.75, zorder=1)
    for (x, y), c in zip(src, colors):
        ax.scatter(x, y, s=24, color=c, zorder=2, lw=0)
    for x, y in tgt:
        ax.scatter(x, y, s=24, facecolor='white', edgecolor='#3a3a3a', lw=.7, zorder=2)
    ax.text(.06, .77, 'Source\nSAE', ha='center', va='bottom', fontsize=7.2)
    ax.text(.22, .77, 'New\nSAE', ha='center', va='bottom', fontsize=7.2)
    ax.text(.0, .595, 'pronouns', ha='left', va='center', fontsize=6.8, color=PRONOUN, rotation=90)
    ax.text(.0, .30, 'women', ha='left', va='center', fontsize=6.8, color=WOMEN, rotation=90)
    ax.text(.14, .08, 'fitted membership $A$', ha='center', va='center', fontsize=7.2)
    # The edited model and its consumers, with the reader each fit matches drawn in its color.
    box(ax, .31, .54, .21, .13, 'edit through\nnew features')
    box(ax, .31, .35, .21, .13, 'frozen model,\nlayers 0 to 22')
    box(ax, .31, .16, .21, .13, 'layer-22 states', ec=STATE)
    ax.text(.415, .135, 'read by the state fit', ha='center', va='top', fontsize=6.6, color=STATE)
    ax.text(.415, .715, 'request $q$, parts or\nrandom subsets', ha='center', va='bottom', fontsize=6.6, color='.25')
    arrow(ax, (.26, .605), (.31, .605))
    arrow(ax, (.415, .54), (.415, .48))
    arrow(ax, (.415, .35), (.415, .29))
    readers = [('original classifier', .74, CLASSIFIER), ('later classifiers', .57, '#3a3a3a'),
               ('gender probe', .40, '#3a3a3a'), ('next token', .23, '#3a3a3a')]
    for name, y, ec in readers:
        box(ax, .67, y - .055, .25, .11, name, ec=ec, size=6.9)
        arrow(ax, (.52, .225), (.67, y))
    ax.text(.795, .80, 'read by the classifier fit', ha='center', va='bottom', fontsize=6.6, color=CLASSIFIER)
    ax.text(.62, .04, 'inputs are biographies or generic text', ha='center', va='bottom', fontsize=6.6, color='.25')


def overview_uses(ax):
    pt = json.loads((ART / 'EXP2_GEMMA_PAPER.json').read_text())['runs'][
        'CSM_EXP2_GEMMA2B_readers_l0low16k_v2_20261011']['point']
    uses = [('Original\nclassifier', lambda m: pt[m]['parts|original']),
            ('Held-out\nclassifiers', lambda m: np.mean([pt[m][f'parts|{r}'] for r in LATER + ['gender']])),
            ('Biography\nnext token', lambda m: pt[m]['parts|bio_next_token']),
            ('Generic-text\nnext token', lambda m: pt[m]['parts|fineweb_next_token'])]
    series = [('hungarian_decoder', 'Hungarian matching', MATCH, 's', ':'),
              ('part_sup_hungarian', 'Classifier fit', CLASSIFIER, 'o', '-'),
              ('lw_subset_state_sup_hungarian', 'State fit', STATE, 'D', '-')]
    xs = np.arange(len(uses))
    for m, label, color, marker, ls in series:
        ys = [f(m) for _, f in uses]
        ax.plot(xs, ys, color=color, marker=marker, ms=4, lw=.9, ls=ls, label=label, mec='white', mew=.4)
    ax.axhline(1, color='.55', lw=.6, ls='--')
    ax.text(len(uses) - 1.02, 1.03, 'no edit', ha='right', va='bottom', fontsize=6.8, color='.4')
    ax.set_xticks(xs)
    ax.set_xticklabels([u for u, _ in uses], fontsize=6.9)
    ax.set_ylim(0, 1.65)
    ax.set_xlim(-.3, len(uses) - .7)
    ax.set_yticks([0, .5, 1, 1.5])
    ax.set_ylabel('Part-deletion error (nRMSE)', fontsize=7.4)
    ax.tick_params(labelsize=7, length=2.5, width=.5)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=6.9, loc='upper left', handlelength=1.8)


def overview():
    fig = plt.figure(figsize=(6.3, 2.25))
    a = fig.add_axes([.0, .02, .56, .96])
    b = fig.add_axes([.64, .2, .35, .74])
    overview_schematic(a)
    overview_uses(b)
    fig.text(.005, .985, '(a) A correspondence and the consumers of the edited model', fontsize=8, va='top')
    fig.text(.585, .985, '(b) Gemma-2-2B SHIFT, 16k lowest-$L_0$ target', fontsize=8, va='top')
    save(fig, 'rc_overview')


def save(fig, name):
    for ext in ['pdf', 'svg']:
        fig.savefig(FIG / f'{name}.{ext}')
    if PREVIEW:
        fig.savefig(PREVIEW / f'{name}.png', dpi=220)
    plt.close(fig)


PREVIEW = None


def main():
    global PREVIEW
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--figures', nargs='+', default=['reader_matrix'])
    parser.add_argument('--preview', type=Path, default=None, help='also write PNG previews to this directory')
    args = parser.parse_args()
    PREVIEW = args.preview
    style()
    for name in args.figures:
        globals()[name]()


if __name__ == '__main__':
    main()
