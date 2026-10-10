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
RANDOM = '#b0543f'


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
            ('random_parts', 'Random halves', RANDOM, '^'),
            ('parts', 'Part supervision', PART, 'o'),
            ('raw_reconstruction', 'Re-applied\nsource edit', READOUT, 'h')]
    fig, ax = plt.subplots(figsize=(3.25, 2.75))
    ax.plot([0, .66], [0, .66], color='.78', lw=.6, zorder=0)
    labels = {'hungarian_decoder': (.262, .492, 'left'), 'hungarian_activation': (.222, .441, 'right'),
              'geometry': (.300, .640, 'left'), 'activation_top2': (.222, .560, 'right'),
              'geometry_gain': (.336, .352, 'center'), 'native': (.272, .262, 'left'),
              'whole': (.108, .318, 'left'), 'random_parts': (.134, .249, 'right'), 'parts': (.150, .082, 'left'),
              'raw_reconstruction': (.118, .168, 'right')}
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


def public_panel(ax, legend_size=7):
    data = json.loads((ART / 'e4_scale' / 'E4A_RESULTS.json').read_text())
    xs, series = [], {'raw_reconstruction': [], 'tangent_full': [], 'tangent_mixed': []}
    widths = []
    for run in data['per_run'].values():
        xs.append(float(np.mean(run['target']['source_activation_retained'])))
        widths.append(run['target']['dict_size'])
        for key in series:
            series[key].append(run['methods'][key]['parts']['nrmse'])
    xs = np.array(xs)
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
    ax.legend(frameon=False, fontsize=legend_size, loc='upper right', handletextpad=.3, borderaxespad=.2)
    return xs, widths, series


def public():
    fig, ax = plt.subplots(figsize=(3.25, 2.25))
    xs, widths, series = public_panel(ax)
    fig.tight_layout(pad=.3)
    for ext in ['pdf', 'svg']:
        fig.savefig(FIG / f'naacl_public.{ext}')
    corr = {key: float(np.corrcoef(xs, series[key])[0, 1]) for key in series}
    (ART / 'e4_scale' / 'PUBLIC_FIGURE_DATA.json').write_text(json.dumps(dict(
        retained=xs.tolist(), widths=widths, part_nrmse=series, pearson_with_retention=corr), indent=1))
    print('pearson with retention', corr)


SETTINGS = [('SHIFT, Pythia-70M', 'SHIFT, Pythia-70M', 'o', False), ('Infinitive, Pythia-70M', 'Infinitive, Pythia-70M', 's', False),
            ('Grammar, GPT-2 Medium', 'Grammar, GPT-2 Medium', 'D', False), ('Infinitive, SAEBench', 'Infinitive, SAEBench', 's', True),
            ('Grammar, Qwen3-1.7B', 'Grammar, Qwen-Scope', 'D', True), ('SHIFT, Gemma-2-2B', 'SHIFT, Gemma Scope', 'o', True)]
INK = '#33424d'


def ratio_panel(ax, x, ratio, marker, filled, label, size=17):
    ax.scatter(x, ratio, s=size, marker=marker, facecolor=INK if filled else 'white', edgecolor=INK, linewidth=.7,
               label=label, zorder=3)


def alternatives():
    # (a) Matching against the learned correspondence over every target, by atom stability of the explained features.
    # (b) Part error on SAEBench against the retention of the source activation by the target reconstruction.
    data = json.loads((ART / 'e4_scale' / 'ATOM_STABILITY.json').read_text())
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.25, 3.75))
    for key, label, marker, filled in SETTINGS:
        rows = [r for r in data['rows'] if r['setting'] == key]
        ratio_panel(ax_a, [r['hungarian_cos'] for r in rows], [r['errors']['hungarian'] / r['errors']['part'] for r in rows],
                    marker, filled, label)
    ax_a.axhline(1, color='.7', lw=.6, zorder=0)
    ax_a.set_yscale('log')
    ax_a.set_yticks([1, 2, 5, 10, 20, 50])
    ax_a.set_yticklabels(['1', '2', '5', '10', '20', '50'])
    ax_a.set_ylim(.8, 60)
    ax_a.set_xlabel('Decoder cosine of Hungarian matches')
    ax_a.set_ylabel('Hungarian error / learned error')
    ax_a.legend(frameon=False, fontsize=6.6, ncol=2, loc='lower center', bbox_to_anchor=(.5, 1.0), handletextpad=.2,
                columnspacing=1.2, borderaxespad=.1, labelspacing=.25)
    ax_a.text(-.19, 1.02, '(a)', transform=ax_a.transAxes, fontsize=8, va='bottom')
    public_panel(ax_b, legend_size=6.6)
    ax_b.text(-.19, 1.02, '(b)', transform=ax_b.transAxes, fontsize=8, va='bottom')
    fig.tight_layout(pad=.3, h_pad=.8)
    for ext in ['pdf', 'svg']:
        fig.savefig(FIG / f'naacl_alternatives.{ext}')


GRID = [('lowest', '16k', 'final_l0low16k'), ('lower', '16k', 'grid_w16k_lower'), ('higher', '16k', 'grid_w16k_higher'),
        ('highest', '16k', 'grid_w16k_highest'), ('lowest', '65k', 'grid_w65k_lowest'), ('closest\nto 100', '65k', 'final_w65k'),
        ('highest', '65k', 'grid_w65k_highest')]


def grid():
    # Part error of each method on seven Gemma Scope targets, ordered by width and by L0 relative to the source.
    runs = {}
    for name in ['E4C_FINAL.json', 'E10_GRID.json']:
        runs.update(json.loads((ART / 'e4_scale' / name).read_text())['runs'])
    spec = [('hungarian_decoder', 'Hungarian matching', MATCH, 'D'), ('readout', 'Re-applied source edit', READOUT, 'h'),
            ('complete_sup_hungarian', 'Complete supervision', COMPLETE, 'X'),
            ('part_sup_hungarian', 'Part supervision', PART, 'o'),
            ('subset_sup_hungarian', 'Parts and random subsets', OTHER, 's')]
    fig, ax = plt.subplots(figsize=(6.75, 2.1))
    record, legend = {}, {}
    for x, (l0, width, key) in enumerate(GRID):
        run = next(v for k, v in runs.items() if key in k)
        record[f'{width} {l0}'] = {}
        for i, (method, label, color, marker) in enumerate(spec):
            p, (lo, hi) = run['point'][method]['parts'], run['intervals'][method]['parts']
            record[f'{width} {l0}'][method] = dict(point=p, interval=[lo, hi])
            h = ax.errorbar(x + (i - (len(spec) - 1) / 2) * .14, p, yerr=[[p - lo], [hi - p]], fmt=marker, ms=4.2,
                            color=color, mec='white' if method == 'part_sup_hungarian' else color, mew=.4, elinewidth=.9,
                            capsize=1.4, capthick=.7)
            legend[label] = h
    ax.axvline(3.5, color='.8', lw=.6)
    ax.set_xticks(range(len(GRID)))
    ax.set_xticklabels([l0 for l0, _, _ in GRID], fontsize=7)
    ax.text(1.5, -.115, 'Width 16k, L0 relative to the source', ha='center', fontsize=7.4, transform=ax.get_xaxis_transform())
    ax.text(5.0, -.115, 'Width 65k', ha='center', fontsize=7.4, transform=ax.get_xaxis_transform())
    ax.set_ylabel('Part-deletion error (nRMSE)')
    ax.set_ylim(0, None)
    fig.legend([legend[label] for _, label, _, _ in spec], [label for _, label, _, _ in spec], loc='upper center',
               ncol=len(spec), frameon=False, fontsize=7, bbox_to_anchor=(.5, 1.02))
    fig.tight_layout(pad=.3, rect=(0, 0, 1, .9))
    for ext in ['pdf', 'svg']:
        fig.savefig(FIG / f'naacl_grid.{ext}')
    (ART / 'e4_scale' / 'GRID_FIGURE_DATA.json').write_text(json.dumps(record, indent=1))


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


def gemma(runs, extra_runs):
    assert runs, 'pass --gemma-runs and --gemma-titles'
    rng = np.random.default_rng(20261010)
    spec = [('hungarian_decoder', 'Hungarian matching', MATCH, 'D'), ('readout', 'Re-applied source edit', READOUT, 'h'),
            ('complete_sup_hungarian', 'Complete supervision', COMPLETE, 'X'),
            ('random_sup_hungarian', 'Random subsets', RANDOM, '^'),
            ('part_sup_hungarian', 'Part supervision', PART, 'o'),
            ('subset_sup_hungarian', 'Parts and random subsets', OTHER, 's')]
    fig, axes = plt.subplots(1, len(runs), figsize=(6.75, 2.3), sharey=True)
    record, top, legend = {}, 0., {}
    for ax, (title, run), extra in zip(np.atleast_1d(axes), runs, extra_runs or [None] * len(runs)):
        res = gemma_groups(run, rng, 2000)
        # Methods trained only in the other run of the same dictionary are added; shared methods keep this run.
        if extra is not None:
            for key, value in gemma_groups(extra, rng, 2000).items():
                res.setdefault(key, value)
        record[title] = {k: {g: dict(point=p, interval=np.quantile(b, [.025, .975]).tolist()) for g, (p, b) in v.items()}
                         for k, v in res.items()}
        xs = np.arange(len(GEMMA_GROUPS))
        width = .13
        for i, (key, label, color, marker) in enumerate(spec):
            if key not in res:
                continue
            for x, (g, _) in zip(xs, GEMMA_GROUPS):
                p, b = res[key][g]
                lo, hi = np.quantile(b, [.025, .975])
                top = max(top, hi)
                h = ax.errorbar(x + (i - (len(spec) - 1) / 2) * width, p, yerr=[[p - lo], [hi - p]], fmt=marker, ms=4.2, color=color,
                                mec='white' if key == 'part_sup_hungarian' else color, mew=.4, elinewidth=.9,
                                capsize=1.4, capthick=.7)
                legend[label] = h
        ax.set_xticks(xs)
        ax.set_xticklabels([g for g, _ in GEMMA_GROUPS], fontsize=7)
        ax.set_title(title, fontsize=8)
    np.atleast_1d(axes)[0].set_ylim(0, top * 1.05)
    np.atleast_1d(axes)[0].set_ylabel('Response error (nRMSE)')
    order = [label for _, label, _, _ in spec if label in legend]
    rows = 1 if len(order) <= 5 else 2
    fig.legend([legend[k] for k in order], order, loc='upper center', ncol=-(-len(order) // rows), frameon=False,
               fontsize=7, bbox_to_anchor=(.5, 1.03))
    fig.tight_layout(pad=.3, rect=(0, 0, 1, .93 if rows == 1 else .86))
    for ext in ['pdf', 'svg']:
        fig.savefig(FIG / f'naacl_gemma.{ext}')
    (ART / 'e4_scale' / 'GEMMA_FIGURE_DATA.json').write_text(json.dumps(record, indent=1))


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('figures', nargs='*', default=['complete_part', 'public'])
    parser.add_argument('--gemma-runs', nargs='*', type=Path, default=[])
    parser.add_argument('--gemma-titles', nargs='*', default=[])
    parser.add_argument('--gemma-extra-runs', nargs='*', type=Path, default=[])
    args = parser.parse_args()
    style()
    for name in args.figures:
        if name == 'gemma':
            gemma(list(zip(args.gemma_titles, args.gemma_runs)), args.gemma_extra_runs)
        else:
            globals()[name]()


if __name__ == '__main__':
    main()
