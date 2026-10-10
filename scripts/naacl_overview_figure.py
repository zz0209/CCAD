from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts' / 'naacl_sprint_20261010'
FIG = ROOT / 'paper' / 'figures'

PART = '#1d6b63'
WRONG = '#a5463f'
INK = '#222222'
MUTED = '#6b6b6b'
GROUP = {'pronouns': '#3f6fa3', 'names': '#c07a2c', 'associated': '#7a5c99'}


def answer(v):
    return 'Nurse' if v else 'Professor'


def arrow(ax, a, b, color=INK, lw=.7, ls='-'):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=6, color=color, lw=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


def main():
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    font_manager.fontManager.addfont('C:/Windows/Fonts/timesi.ttf')
    font_manager.fontManager.addfont('C:/Windows/Fonts/timesbd.ttf')
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 8, 'mathtext.fontset': 'stix',
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    ex = json.loads((ART / 'e1_matching' / 'FIGURE1_EXAMPLE.json').read_text())
    fig = plt.figure(figsize=(6.75, 2.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 6.75)
    ax.set_ylim(0, 2.3)
    ax.axis('off')

    ax.text(.06, 2.2, '(a) One held-out biography, edited in a retrained SAE', fontsize=8.4, va='top')
    ax.text(.06, 1.99, '\u201cShe facilitates workshops \u2026 and teaches in both the',
            fontsize=7.3, style='italic', va='top', color=INK)
    ax.text(.06, 1.84, 'undergraduate and graduate programs \u2026\u201d', fontsize=7.3, style='italic', va='top', color=INK)
    ax.text(2.62, 1.92, 'Unedited answer: Nurse\nTrue profession: Professor', fontsize=7.0, va='center', color=MUTED,
            linespacing=1.3)
    cols = [(1.42, 'Original\nSAE'), (2.03, 'Complete\nsupervision'), (2.66, 'Hungarian\nmatching'), (3.29, 'Part\nsupervision')]
    for x, label in cols:
        ax.text(x, 1.50, label, ha='center', va='center', fontsize=7.2, color=PART if 'Part' in label else INK,
                linespacing=1.15)
    ax.plot([.06, 3.62], [1.33, 1.33], color='.55', lw=.5)
    rows = [('full', 'All gender cues'), ('pronouns', 'Pronouns'), ('names', 'Names'), ('associated_words', 'Associated words')]
    methods = ['source', 'whole', 'hungarian_decoder', 'parts']
    for r, (q, label) in enumerate(rows):
        y = 1.18 - .27 * r
        ax.text(.06, y, label, fontsize=7.5, va='center')
        src = ex['answers']['source'][q]
        for (x, _), m in zip(cols, methods):
            a = ex['answers'][m][q]
            bad = m != 'source' and a != src
            ax.text(x, y, answer(a), ha='center', va='center', fontsize=7.5,
                    color=WRONG if bad else (PART if m == 'parts' else INK))
            if bad:
                ax.add_patch(FancyBboxPatch((x - .28, y - .08), .56, .16, boxstyle='round,pad=0.01,rounding_size=0.04',
                                            fill=False, ec=WRONG, lw=.6))
    ax.text(.06, .16, 'Each cell gives the classifier answer after deleting that part of the explanation.',
            fontsize=7.2, color=INK, va='center')
    ax.plot([3.78, 3.78], [.1, 2.18], color='.85', lw=.6)

    ax.text(3.92, 2.2, '(b) Part-supervised correspondence', fontsize=8.4, va='top')
    ax.text(4.55, 1.92, 'Source SAE', ha='center', fontsize=7.2)
    ax.text(5.40, 1.92, 'New SAE', ha='center', fontsize=7.2)
    src = [('pronouns', 1.70), ('pronouns', 1.55), ('names', 1.33), ('names', 1.18), ('associated', .96), ('associated', .81)]
    tgt = [1.74, 1.58, 1.42, 1.26, 1.10, .94, .78]
    links = [(0, 0, .9), (0, 1, .4), (1, 1, .7), (2, 2, .8), (2, 3, .5), (3, 3, .6), (3, 1, .3), (4, 4, .8),
             (4, 5, .5), (5, 5, .7), (5, 6, .4), (1, 4, .25)]
    for i, j, w in links:
        ax.plot([4.60, 5.35], [src[i][1], tgt[j]], color=GROUP[src[i][0]], lw=.4 + 1.1 * w, alpha=.55,
                solid_capstyle='round')
    for name, y in src:
        ax.add_patch(Circle((4.55, y), .05, color=GROUP[name], zorder=3))
    for y in tgt:
        ax.add_patch(Circle((5.40, y), .046, fc='white', ec='.35', lw=.6, zorder=3))
    for name, y, label in [('pronouns', 1.625, 'pronouns'), ('names', 1.255, 'names'), ('associated', .885, 'associated\nwords')]:
        ax.text(4.44, y, label, ha='right', va='center', fontsize=7.4, color=GROUP[name], linespacing=1.1)
    ax.text(4.97, .56, 'many-to-many membership', ha='center', fontsize=7.4, color=INK)

    ax.add_patch(FancyBboxPatch((5.66, 1.42), 1.0, .40, boxstyle='round,pad=0.02,rounding_size=0.05', fc='#eef4f3',
                                ec=PART, lw=.6))
    ax.text(6.16, 1.62, 'delete one part\nthrough new features', ha='center', va='center', fontsize=7.4, linespacing=1.15)
    ax.add_patch(FancyBboxPatch((5.66, .86), 1.0, .34, boxstyle='round,pad=0.02,rounding_size=0.05', fc='white',
                                ec='.45', lw=.6))
    ax.text(6.16, 1.03, 'frozen language model', ha='center', va='center', fontsize=7.4)
    arrow(ax, (5.47, 1.42), (5.66, 1.58), color='.35')
    arrow(ax, (6.16, 1.42), (6.16, 1.20), color='.35')
    ax.text(6.16, .43, 'match the original effect\nof every part', ha='center', va='center', fontsize=7.4, color=PART,
            linespacing=1.15)
    arrow(ax, (6.16, .86), (6.16, .60), color=PART)
    arrow(ax, (5.62, .36), (5.08, .36), color=PART, ls='--', lw=.6)
    ax.text(5.35, .25, 'update membership', ha='center', fontsize=7.2, color=PART)
    for ext in ['pdf', 'svg']:
        fig.savefig(FIG / f'naacl_overview.{ext}')


if __name__ == '__main__':
    main()
