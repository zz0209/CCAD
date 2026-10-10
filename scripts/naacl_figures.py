from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts' / 'naacl_sprint_20261010'
FIG = ROOT / 'paper' / 'figures'
DESCRIPTION = 'Data figures of the NAACL manuscript, drawn from saved sprint analyses.'

PART = '#1d6b63'
COMPLETE = '#8a5a9e'
MATCH = '#8d8d8d'
READOUT = '#b5832e'
OTHER = '#4d6f9a'


def style():
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    font_manager.fontManager.addfont('C:/Windows/Fonts/timesi.ttf')
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 8, 'mathtext.fontset': 'stix',
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': .6,
                         'xtick.major.width': .6, 'ytick.major.width': .6, 'xtick.major.size': 2.5,
                         'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False})


def complete_part():
    data = json.loads((ART / 'e1_matching' / 'E1_SHIFT_R59PANEL.json').read_text())
    s, ci = data['summary'], data['intervals']
    rows = [('hungarian_decoder', 'Hungarian, decoder', MATCH, 'o'),
            ('hungarian_activation', 'Hungarian, activation', MATCH, 's'),
            ('geometry', 'Top-2 decoder', MATCH, 'D'),
            ('activation_top2', 'Top-2 activation', MATCH, '^'),
            ('geometry_gain', 'Calibrated decoder', MATCH, 'v'),
            ('native', 'Natural-text fit', OTHER, 'P'),
            ('whole', 'Complete supervision', COMPLETE, 'X'),
            ('parts', 'Part supervision', PART, 'o'),
            ('raw_reconstruction', 'Re-applied source edit', READOUT, 'h')]
    fig, ax = plt.subplots(figsize=(3.25, 2.75))
    ax.plot([0, .66], [0, .66], color='.78', lw=.6, zorder=0)
    labels = {'hungarian_decoder': (.262, .492, 'left'), 'hungarian_activation': (.222, .441, 'right'),
              'geometry': (.300, .640, 'left'), 'activation_top2': (.222, .560, 'right'),
              'geometry_gain': (.336, .352, 'center'), 'native': (.272, .262, 'left'),
              'whole': (.108, .318, 'left'), 'parts': (.158, .072, 'left'),
              'raw_reconstruction': (.098, .195, 'center')}
    for key, label, color, marker in rows:
        x, y = s[key]['full_nrmse'], s[key]['parts_nrmse']
        (xl, xh), (yl, yh) = ci[key]['full'], ci[key]['parts']
        ax.errorbar(x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]], fmt=marker, ms=4.4,
                    color=color, mec='white' if key == 'parts' else color, mew=.5, elinewidth=.6, capsize=0)
        tx, ty, ha = labels[key]
        ax.text(tx, ty, label, fontsize=7.4, ha=ha, va='center', color=color if color != MATCH else '.25')
    ax.set_xlim(0, .42)
    ax.set_ylim(0, .66)
    ax.set_xticks([0, .1, .2, .3, .4])
    ax.set_yticks([0, .2, .4, .6])
    ax.set_xlabel('Complete-deletion error (nRMSE)')
    ax.set_ylabel('Part-deletion error (nRMSE)')
    fig.tight_layout(pad=.3)
    (x0, y0), (x1, y1) = ax.transData.transform((.1, .1)), ax.transData.transform((.3, .3))
    angle = np.degrees(np.arctan2(y1 - y0, x1 - x0))
    ax.text(.222, .205, 'equal error', color='.55', rotation=angle, fontsize=6.6, ha='center', va='center',
            rotation_mode='anchor')
    for ext in ['pdf', 'svg']:
        fig.savefig(FIG / f'naacl_complete_part.{ext}')


def public():
    data = json.loads((ART / 'e4_scale' / 'E4A_RESULTS.json').read_text())
    xs, series = [], {'raw_reconstruction': [], 'tangent_full': [], 'tangent_mixed': []}
    widths = []
    for run in data['per_run'].values():
        xs.append(float(np.mean(run['target']['source_activation_retained'])))
        widths.append(run['target']['dict_size'])
        for key in series:
            series[key].append(run['methods'][key]['parts']['nrmse'])
    xs = np.array(xs)
    fig, ax = plt.subplots(figsize=(3.25, 2.25))
    spec = [('raw_reconstruction', 'Re-applied source edit', READOUT, 'h'),
            ('tangent_full', 'Complete supervision', COMPLETE, 'X'),
            ('tangent_mixed', 'Part supervision', PART, 'o')]
    rng = np.random.default_rng(20261010)
    grid = np.linspace(xs.min(), xs.max(), 40)
    for key, label, color, marker in spec:
        y = np.array(series[key])
        hollow = key == 'tangent_full'
        ax.scatter(xs, y, s=18 if hollow else 15, marker=marker, facecolor='none' if hollow else color,
                   edgecolor='white' if key == 'tangent_mixed' else color, linewidth=.8 if hollow else .4, label=label,
                   zorder=4 if hollow else 3)
        slope, intercept = np.polyfit(xs, y, 1)
        ax.plot(grid, slope * grid + intercept, color=color, lw=.7, alpha=.8)
        # 95% band of the fitted line from resampling the 16 public targets.
        fits = []
        for _ in range(2000):
            idx = rng.integers(0, len(xs), len(xs))
            b = np.polyfit(xs[idx], y[idx], 1)
            fits.append(b[0] * grid + b[1])
        lo, hi = np.quantile(np.array(fits), [.025, .975], axis=0)
        ax.fill_between(grid, lo, hi, color=color, alpha=.12, lw=0)
    ax.set_xlabel('Source activation retained by the target reconstruction')
    ax.set_ylabel('Part-deletion error (nRMSE)')
    ax.set_ylim(0, .65)
    ax.legend(frameon=False, fontsize=7, loc='upper right', handletextpad=.3, borderaxespad=.2)
    fig.tight_layout(pad=.3)
    for ext in ['pdf', 'svg']:
        fig.savefig(FIG / f'naacl_public.{ext}')
    corr = {key: float(np.corrcoef(xs, series[key])[0, 1]) for key in series}
    (ART / 'e4_scale' / 'PUBLIC_FIGURE_DATA.json').write_text(json.dumps(dict(
        retained=xs.tolist(), widths=widths, part_nrmse=series, pearson_with_retention=corr), indent=1))
    print('pearson with retention', corr)


GEMMA_GROUPS = [('Complete', lambda r: r == 'complete'), ('Parts', lambda r: r in ('pronouns', 'female_descriptions')),
                 ('Layer\nranges', lambda r: r.startswith('band_') and r.count('_') == 1),
                 ('Range\nwithin part', lambda r: r.startswith('band_') and r.count('_') > 1),
                 ('Random\nhalves', lambda r: r.startswith('members_'))]


def gemma_groups(run, rng, draws):
    data = np.load(run / 'responses.npz')
    names = list(data['request_names'])
    src = data['source']
    strata = data['eval_true'] * 2 + data['eval_spur']
    w = np.zeros((draws, len(strata)))
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        picks = rng.choice(idx, size=(draws, len(idx)))
        for d in range(draws):
            np.add.at(w[d], picks[d], 1.)
    out = {}
    for method in [k for k in data.files if k not in ('request_names', 'eval_true', 'eval_spur', 'eval_base', 'source')]:
        key = method.rsplit('_s', 1)[0] if '_sup_' in method else method
        for g, test in GEMMA_GROUPS:
            ix = [i for i, r in enumerate(names) if test(r)]
            err, eff = ((data[method][ix] - src[ix]) ** 2).sum(0), (src[ix] ** 2).sum(0)
            point = np.sqrt(err.sum() / eff.sum())
            boot = np.sqrt((w * err).sum(1) / (w * eff).sum(1))
            out.setdefault(key, {}).setdefault(g, []).append((point, boot))
    return {k: {g: (float(np.mean([p for p, _ in v])), np.mean([b for _, b in v], axis=0)) for g, v in gs.items()}
            for k, gs in out.items()}


def gemma(runs):
    assert runs, 'pass --gemma-runs and --gemma-titles'
    rng = np.random.default_rng(20261010)
    spec = [('hungarian_decoder', 'Hungarian matching', MATCH, 'D'), ('readout', 'Re-applied source edit', READOUT, 'h'),
            ('complete_sup_hungarian', 'Complete supervision', COMPLETE, 'X'),
            ('part_sup_hungarian', 'Part supervision', PART, 'o'),
            ('subset_sup_hungarian', 'Part and subset supervision', OTHER, 's')]
    fig, axes = plt.subplots(1, len(runs), figsize=(6.75, 2.3), sharey=True)
    record, top, legend = {}, 0., {}
    for ax, (title, run) in zip(np.atleast_1d(axes), runs):
        res = gemma_groups(run, rng, 2000)
        record[title] = {k: {g: dict(point=p, interval=np.quantile(b, [.025, .975]).tolist()) for g, (p, b) in v.items()}
                         for k, v in res.items()}
        xs = np.arange(len(GEMMA_GROUPS))
        width = .14
        for i, (key, label, color, marker) in enumerate(spec):
            if key not in res:
                continue
            for x, (g, _) in zip(xs, GEMMA_GROUPS):
                p, b = res[key][g]
                lo, hi = np.quantile(b, [.025, .975])
                top = max(top, hi)
                h = ax.errorbar(x + (i - 2) * width, p, yerr=[[p - lo], [hi - p]], fmt=marker, ms=4.2, color=color,
                                mec='white' if key == 'part_sup_hungarian' else color, mew=.4, elinewidth=.9,
                                capsize=1.4, capthick=.7)
                legend[label] = h
        ax.set_xticks(xs)
        ax.set_xticklabels([g for g, _ in GEMMA_GROUPS], fontsize=7)
        ax.set_title(title, fontsize=8)
    np.atleast_1d(axes)[0].set_ylim(0, top * 1.05)
    np.atleast_1d(axes)[0].set_ylabel('Response error (nRMSE)')
    order = [label for _, label, _, _ in spec if label in legend]
    fig.legend([legend[k] for k in order], order, loc='upper center', ncol=len(order), frameon=False, fontsize=7,
               bbox_to_anchor=(.5, 1.03))
    fig.tight_layout(pad=.3, rect=(0, 0, 1, .93))
    for ext in ['pdf', 'svg']:
        fig.savefig(FIG / f'naacl_gemma.{ext}')
    (ART / 'e4_scale' / 'GEMMA_FIGURE_DATA.json').write_text(json.dumps(record, indent=1))


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('figures', nargs='*', default=['complete_part', 'public'])
    parser.add_argument('--gemma-runs', nargs='*', type=Path, default=[])
    parser.add_argument('--gemma-titles', nargs='*', default=[])
    args = parser.parse_args()
    style()
    for name in args.figures:
        if name == 'gemma':
            gemma(list(zip(args.gemma_titles, args.gemma_runs)))
        else:
            globals()[name]()


if __name__ == '__main__':
    main()
