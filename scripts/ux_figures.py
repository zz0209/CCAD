from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from rc_figures import MATCH, STATE, style

ROOT = Path(__file__).resolve().parents[1]
CIRC = ROOT / 'artifacts' / 'circuits'
ART = ROOT / 'artifacts' / 'containment'
FIG = ROOT / 'paper' / 'figures'
DESCRIPTION = 'Figures of the dictionary-change manuscript, drawn from CIRCUIT_SUMMARY.json and saved SAE analyses.'

INK = '#1e1e1e'
OWN = '#3b4a5a'
LATER = '#a2512d'
GEMMA = '#2f5d7c'
LLAMA = '#b8873a'
GEMMA3 = '#4f7a4a'
PREVIEW = None


def clean(ax):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=6.8, length=2.2, width=.5)


def save(fig, name):
    for ext in ['pdf', 'svg']:
        fig.savefig(FIG / f'{name}.{ext}')
    if PREVIEW:
        fig.savefig(PREVIEW / f'{name}.png', dpi=220)
    plt.close(fig)


def supports(ax, s):
    # Write supports of one supernode deletion. Bar heights are the median shares of the squared write norm per layer,
    # with the CLT candidates' later-layer share spread evenly over the later layers drawn.
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    later = s['oracle']['gemma_clt426k']['candidate_later_share']['median']
    cols = [.47, .565, .66, .755, .85, .945]
    names = ['$l$', '$l{+}1$', '$l{+}2$', '$l{+}3$', '$\\cdots$', '$L$']
    rows = [('Published deletion\nin the PLT', .70, [1, 0, 0, 0, 0, 0], None),
            ('Whole CLT features', .42, [1 - later] + [later / 5] * 5, None),
            ('Layer-$l$ slices of\nCLT features', .14, [1, 0, 0, 0, 0, 0], '////')]
    h, w = .2, .055
    for label, y, shares, hatch in rows:
        ax.text(.0, y + .06, label, ha='left', va='center', fontsize=7.1, linespacing=1.05)
        ax.plot([cols[0] - .05, cols[-1] + .04], [y, y], color='.6', lw=.5)
        for x, v in zip(cols, shares):
            if v == 0:
                continue
            c = OWN if x == cols[0] else LATER
            ax.add_patch(Rectangle((x - w / 2, y), w, h * v, fc='white' if hatch else c, ec=c, lw=.6, hatch=hatch))
    for x, n in zip(cols, names):
        ax.text(x, .055, n, ha='center', va='center', fontsize=7)
    ax.text((cols[0] + cols[-1]) / 2, -.03, 'layer whose MLP output is written', ha='center', va='center',
            fontsize=6.8, color='.3')
    ax.text((cols[1] + cols[-1]) / 2, .42 + h * later / 5 + .03, f'{later:.0%} of the write lands in later layers',
            ha='center', va='bottom', fontsize=6.6, color=LATER)
    ax.text(cols[3], .14 + .1, 'not an edit of\nwhole features', ha='center', va='center', fontsize=6.6, color='.3')


def overview_errors(ax, s):
    # Effect error of published explanations after a change of dictionary, by kind of change.
    o, f, r, g3 = s['oracle'], s['fits'], s['reverse'], s['oracle_g3']
    targets = json.loads((ART / 'GEMMA_TARGETS.json').read_text())['rows']
    tok = lambda m: [t['methods'][m]['token'] for t in targets]
    held = lambda fam, m: json.loads((ART / f'UJ_{fam}_READERS.json').read_text())['point'][m]['parts']['E2']
    plt_whole = [o[f'gemma_plt_r{k}']['nrmse_whole']['median'] for k in range(1, 5)]
    # Each row: label, then values (or ranges) for matched features, fitted correspondence, best reconstruction with
    # whole target features and with layer-l slices.
    rows = [('Same family', None),
            ('Gemma-2 PLT, other $L_0$', dict(match=f['gemma_plt_r2']['one_to_one']['median'],
                                              fit=f['gemma_plt_r2']['fitted']['median'], whole=plt_whole)),
            ('Gemma-2 CLT 426k$\\to$2.5M', dict(whole=s['clt_pair']['nrmse_whole']['median'],
                                                slice=s['clt_pair']['nrmse_slices']['median'])),
            ('Gemma-3 PLT, other $L_0$', dict(whole=g3['g3_plt_big']['nrmse_whole']['median'])),
            ('Retrained SAE', None),
            ('SHIFT, Gemma-2', dict(match=tok('hungarian'), fit=tok('state'))),
            ('Grammar, GPT-2 M', dict(match=held('GPT2', 'hungarian_activation'), fit=held('GPT2', 'u_distjoint'))),
            ('Grammar, Qwen3-1.7B', dict(match=held('QWEN', 'hungarian_activation'), fit=held('QWEN', 'u_distjoint'))),
            ('PLT$\\to$CLT', None),
            ('Gemma-2, 426k', dict(match=s['match']['gemma_clt426k']['one_to_one']['median'],
                                 fit=[f['gemma_clt426k_whole']['fitted']['median'], f['gemma_clt426k_slices']['fitted']['median']],
                                 whole=o['gemma_clt426k']['nrmse_whole_best']['median'],
                                 slice=o['gemma_clt426k']['nrmse_slice']['median'])),
            ('Gemma-2, 2.5M', dict(whole=o['gemma_clt2500k']['nrmse_whole_best']['median'],
                                 slice=o['gemma_clt2500k']['nrmse_slice']['median'])),
            ('Llama, 524k', dict(whole=o['llama_clt524k']['nrmse_whole_best']['median'],
                                 slice=o['llama_clt524k']['nrmse_slice']['median'])),
            ('Gemma-3, 262k', dict(whole=g3['g3_clt262k']['nrmse_whole_best']['median'],
                                   slice=g3['g3_clt262k']['nrmse_slice']['median'])),
            ('CLT$\\to$PLT', None),
            ('Gemma-2', dict(whole=r['gemma']['nrmse_layerwise']['median'])),
            ('Llama', dict(whole=r['llama']['nrmse_layerwise']['median'])),
            ('Gemma-3', dict(whole=r['gemma3']['nrmse_layerwise']['median']))]
    marks = dict(match=('Matched features', MATCH, 's', MATCH), fit=('Fitted\ncorrespondence', STATE, 'D', STATE),
                 whole=('Best edit of\nwhole features', INK, 'o', INK), slice=('Best edit of\nlayer-$l$ slices', LATER, '^', 'white'))
    y = 0
    ticks, labels = [], []
    for label, vals in rows:
        y -= 1
        if vals is None:
            ax.text(-.02, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=7.1,
                    fontstyle='italic', color='.25')
            continue
        ticks.append(y)
        labels.append(label)
        for k, (name, color, marker, face) in marks.items():
            if k not in vals:
                continue
            v = np.atleast_1d(vals[k])
            if len(v) > 1:
                ax.plot([v.min(), v.max()], [y, y], color=color, lw=1.1, alpha=.55, solid_capstyle='butt')
            ax.scatter([np.median(v)], [y], s=20, marker=marker, color=face, edgecolor=color, lw=.7, zorder=3)
    ax.axvline(1, color='.55', lw=.6, ls='--')
    ax.text(.97, -1, 'no edit', ha='right', va='center', fontsize=6.6, color='.4')
    ax.set_xscale('log')
    ax.set_xlim(.1, 1.25)
    ax.set_xticks([.1, .2, .5, 1])
    ax.set_xticklabels(['.1', '.2', '.5', '1'])
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6.9)
    ax.set_ylim(y - .6, -.4)
    ax.set_xlabel('Effect error of the moved explanation (nRMSE)', fontsize=7.2)
    ax.grid(axis='x', color='.9', lw=.4)
    ax.set_axisbelow(True)
    clean(ax)
    ax.tick_params(axis='y', length=0)
    handles = [Line2D([], [], ls='', marker=m, ms=4.2, mfc=fc, mec=c, mew=.7, label=n) for n, c, m, fc in marks.values()]
    ax.legend(handles=handles, frameon=False, fontsize=6.8, loc='upper left', handletextpad=.3, borderaxespad=.1,
              bbox_to_anchor=(1.02, 1.0), labelspacing=.9)


def transfer():
    s = json.loads((CIRC / 'CIRCUIT_SUMMARY.json').read_text())
    fig = plt.figure(figsize=(6.3, 2.55))
    a = fig.add_axes([.0, .07, .38, .83])
    b = fig.add_axes([.56, .14, .255, .80])
    supports(a, s)
    overview_errors(b, s)
    fig.text(.005, .985, '(a) Where a supernode deletion writes', fontsize=8, va='top')
    fig.text(.41, .985, '(b) Explanations after a change of dictionary', fontsize=8, va='top')
    save(fig, 'ux_transfer')


READINGS = Path('D:/CCAD_Storage/runs/circuits/EXAMPLE_READINGS_GEMMA_v1/READINGS.json')


def depth_map(ax):
    # The Dallas graph on token-by-layer axes: published PLT supernode features and the CLT graph's top units, a CLT
    # unit filled when a single PLT feature at its layer and token reproduces its write (residual below .4).
    g = [r for r in json.loads(READINGS.read_text(encoding='utf-8'))['readings'] if r['graph'].endswith('dallas-austin')][0]
    toks = ['BOS' if t == '<bos>' else (t.strip() or '_') for t in g['tokens']]
    names = {'capital': 'capital', 'state': 'state', 'Texas': 'Texas',
             'preposition followed by place name': 'place name', 'capital cities / say a capital city': 'say a capital'}
    for s in g['supernodes']:
        xs = [x['position'] for x in s['sites']]
        ys = [x['layer'] for x in s['sites']]
        ax.scatter(xs, ys, s=22, marker='o', facecolor='none', edgecolor=INK, lw=.7, zorder=3)
        # Labels of supernodes on the token before the last go to their left, clear of the last token's column.
        left = max(xs) == len(toks) - 2
        ax.text(max(xs) + (-.45 if left else .55), max(ys) + .8, names.get(s['name'], s['name']), fontsize=6.3,
                va='center', ha='right' if left else 'left', color='.25')
    # Features of the 426k CLT active at nearly every token (CIRC_FREQ_GEMMA_CLT426K_v1).
    always = {(0, 7081), (1, 3148), (1, 3872)}
    for u in g['clt_units']:
        shared = u['plt_residual'] < .4
        dense = (u['layer'], u['index']) in always
        size = 14 + 10 * min(abs(u['answer_drop']), 4)
        ax.scatter([u['position'] + .18], [u['layer']], s=size, marker='^', zorder=4, lw=.6,
                   facecolor=GEMMA if shared else ('.75' if dense else 'white'), edgecolor=GEMMA if shared else LATER)
    ax.set_xticks(range(len(toks)))
    ax.set_xticklabels(toks, rotation=60, fontsize=6.2, ha='right')
    ax.set_ylim(-1, 27)
    ax.set_xlim(-.5, len(toks) + 1.9)
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_ylabel('Layer', fontsize=7.2)
    clean(ax)
    handles = [Line2D([], [], ls='', marker='o', ms=4, mfc='none', mec=INK, label='PLT supernode feature'),
               Line2D([], [], ls='', marker='^', ms=4.5, mfc=GEMMA, mec=GEMMA, label='CLT unit, shared'),
               Line2D([], [], ls='', marker='^', ms=4.5, mfc='white', mec=LATER, label='CLT unit, CLT only'),
               Line2D([], [], ls='', marker='^', ms=4.5, mfc='.75', mec=LATER, label='CLT only, always active')]
    ax.legend(handles=handles, frameon=False, fontsize=6.1, loc='upper left', handletextpad=.2, borderaxespad=.1)


NATURAL = Path('D:/CCAD_Storage/runs/circuits')


def natural_map(ax, graph):
    # One natural-text prompt on token-by-layer axes: the PLT's own top units and the CLT's own top units, a CLT unit
    # filled when PLT features at its layer and token reproduce its write, gray when it is one of the always-active
    # features, sized by the effect of deleting its writes with the later MLP outputs clamped.
    toks = json.loads((NATURAL / 'natural' / 'NATURAL_TOKENS_gemma-2-2b.json').read_text(encoding='utf-8'))['tokens'][graph]
    toks = ['BOS' if t == '<bos>' else (t.strip() or '_') for t in toks]
    reg = json.loads((NATURAL / 'GRAPH_UNITS_GEMMA_CLT426K_PLT_NATURAL.json').read_text())['graphs'][graph]
    for fs in reg['variants'][0]['supernodes'].values():
        for l, p, _ in fs:
            ax.scatter([p - .18], [l], s=16, marker='o', facecolor='none', edgecolor=INK, lw=.6, zorder=3)
    rev = {r['unit']: r for r in json.loads((NATURAL / 'CIRC_NAT_REVERSE_GEMMA_CLT426K_v1' / 'REVERSE_ROWS.json').read_text())['rows']
           if r['graph'] == graph and r['features'] == 1}
    dr = {r['unit']: r for r in json.loads((NATURAL / 'CIRC_NAT_DR_GEMMA_CLT426K_v1' / 'DEPTH_REFEREE_ROWS.json').read_text())['rows']
          if r['graph'] == graph}
    always = {'L0F7081', 'L1F3148', 'L1F3872'}
    for unit, r in rev.items():
        l, rest = unit[1:].split('F')
        f, p = rest.split('P')
        own = [res for k, res, _ in r['residual_by_offset'] if k == 0]
        shared = bool(own) and own[0] < .4
        dense = f'L{l}F{f}' in always
        eff = dr[unit]['energy_w_clamped'] ** .5 if unit in dr else 0.
        ax.scatter([int(p) + .18], [int(l)], s=14 + 30 * min(eff, 2), marker='^', zorder=4, lw=.6,
                   facecolor=GEMMA if shared else ('.75' if dense else 'white'), edgecolor=GEMMA if shared else LATER)
    ax.set_xticks(range(len(toks)))
    ax.set_xticklabels(toks, rotation=60, fontsize=6.2, ha='right')
    ax.set_ylim(-1, 27)
    ax.set_xlim(-.5, len(toks) - .3)
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    clean(ax)


NAT_REVERSE = {'gemma': 'CIRC_NAT_REVERSE_GEMMA_CLT426K_v1', 'llama': 'CIRC_NAT_REVERSE_LLAMA_CLT524K_v1',
               'gemma3': 'CIRC_NAT_REVERSE_G3_CLT262K_v1'}


def natural_shared(m, threshold=.4):
    # Shares of the CLT's own top units on the natural-text prompts that PLT features reproduce at their own layer.
    early, later = [], []
    for r in json.loads((NATURAL / NAT_REVERSE[m] / 'REVERSE_ROWS.json').read_text())['rows']:
        own = [res for k, res, _ in r['residual_by_offset'] if k == 0] if r['features'] == 1 else []
        if own:
            (early if int(r['unit'][1:].split('F')[0]) <= 1 else later).append(own[0] < threshold)
    return float(np.mean(early)), float(np.mean(later))


def shared_panel(ax, s):
    models = [('gemma', 'Gemma-2'), ('llama', 'Llama'), ('gemma3', 'Gemma-3')]
    x = np.arange(len(models))
    early = [s['shared_by_depth'][m]['shared_early'] for m, _ in models]
    later = [s['shared_by_depth'][m]['shared_later'] for m, _ in models]
    ax.bar(x - .19, early, width=.36, color=LATER, lw=0, label='Read at layers 0-1')
    ax.bar(x + .19, later, width=.36, color=GEMMA, lw=0, label='Read later')
    nat = [natural_shared(m) for m, _ in models]
    ax.scatter(x - .19, [v[0] for v in nat], marker='D', s=9, color=INK, zorder=4, lw=0, label='Natural text')
    ax.scatter(x + .19, [v[1] for v in nat], marker='D', s=9, color=INK, zorder=4, lw=0)
    for k, (m, _) in enumerate(models):
        ax.text(k - .19, max(early[k], nat[k][0]) + .03, f"{s['shared_by_depth'][m]['early_share']:.0%}\nof units",
                ha='center', va='bottom', fontsize=5.8, color=LATER)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in models], fontsize=6.6)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, .5, 1])
    ax.set_yticklabels(['0', '.5', '1'])
    ax.set_ylabel('Share with a PLT counterpart', fontsize=7.2)
    ax.legend(frameon=False, fontsize=6.1, loc='upper left', handletextpad=.3, borderaxespad=.1)
    clean(ax)


def referee_panel(ax, s):
    rw = s['read_write']
    groups = [('plt_own', lambda l: True, 'PLT units', '.55', 'o'),
              ('clt_own', lambda l: l >= 2, 'CLT units, later', GEMMA, 's'),
              ('clt_own', lambda l: l <= 1, 'CLT units, layers 0-1', LATER, '^')]
    for key, keep, label, color, marker in groups:
        pts = np.array([p for m in rw for p in rw[m].get(key, {}).get('points', []) if keep(p[2])])
        ax.scatter(pts[:, 0], np.maximum(pts[:, 1], 1e-3), s=6, marker=marker, color=color, alpha=.6, lw=0, label=label)
    lim = (1e-2, 15)
    ax.plot(lim, lim, color='.4', lw=.6, ls=':')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(.25, 15)
    ax.set_ylim(1e-2, 15)
    ax.set_xticks([.3, 1, 3, 10])
    ax.set_xticklabels(['.3', '1', '3', '10'])
    ax.set_yticks([.01, .1, 1, 10])
    ax.set_yticklabels(['.01', '.1', '1', '10'])
    ax.set_xlabel('Effect of deleting what it writes', fontsize=7.2)
    ax.set_ylabel('Effect of removing what it reads', fontsize=7.2)
    ax.legend(frameon=False, fontsize=6.1, loc='upper left', handletextpad=.1, borderaxespad=.1, markerscale=1.6)
    clean(ax)


def confirm_panel(ax):
    # CLT units binned by how well PLT features at the same layer and token reproduce their own-layer write, against
    # the median fraction of their later writes that the model removes when the unit's read is removed.
    edges = [0, .25, .5, .75, 1.0001]
    mids = [.125, .375, .625, .875]
    # Solid lines with quartiles for the published prompts, dashed lines for the natural-text prompts.
    for source, dashed in [('AGREEMENT_VS_REFEREE.json', False), ('NAT_AGREEMENT_VS_REFEREE.json', True)]:
        a = json.loads((CIRC / source).read_text())
        for k, (m, name, color, marker) in enumerate([('gemma', 'Gemma-2', GEMMA, 'o'), ('llama', 'Llama', LLAMA, 's'),
                                                        ('gemma3', 'Gemma-3', GEMMA3, '^')]):
            pts = a[m]['points']
            agree = np.clip(np.array([p['agree_own'] for p in pts]), 0, 1)
            conf = np.array([p['confirm_u'] for p in pts])
            xs, ys, lo, hi = [], [], [], []
            for j in range(4):
                sel = conf[(agree >= edges[j]) & (agree < edges[j + 1])]
                if len(sel) >= 8:
                    xs.append(mids[j] + (k - 1) * .025 + (.012 if dashed else 0))
                    ys.append(np.median(sel))
                    lo.append(np.quantile(sel, .25))
                    hi.append(np.quantile(sel, .75))
            if dashed:
                ax.plot(xs, ys, color=color, lw=.8, ls='--', marker=marker, ms=3, mfc='white', zorder=3)
            else:
                ax.plot(xs, ys, color=color, lw=.9, marker=marker, ms=3.4, label=name, zorder=3)
                ax.vlines(xs, lo, hi, color=color, lw=.6, alpha=.7)
    ax.plot([], [], color='.4', lw=.8, ls='--', marker='o', ms=3, mfc='white', label='Natural text')
    ax.axhline(0, color='.6', lw=.5, ls=':')
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .5, 1])
    ax.set_xticklabels(['0', '.5', '1'])
    ax.set_ylim(-.15, 1.05)
    ax.set_yticks([0, .5, 1])
    ax.set_yticklabels(['0', '.5', '1'])
    ax.set_xlabel('Agreement with the PLT', fontsize=7.2)
    ax.set_ylabel('Later writes the model removes', fontsize=7.2)
    ax.legend(frameon=False, fontsize=6.1, loc='upper left', handletextpad=.2, borderaxespad=.1)
    clean(ax)


def overview():
    s = json.loads((CIRC / 'CIRCUIT_SUMMARY.json').read_text())
    fig = plt.figure(figsize=(6.3, 2.45))
    a = fig.add_axes([.055, .25, .33, .66])
    b = fig.add_axes([.46, .25, .2, .66])
    c = fig.add_axes([.75, .25, .24, .66])
    depth_map(a)
    shared_panel(b, s)
    confirm_panel(c)
    fig.text(.005, .985, '(a) Dallas$\\to$Austin in a PLT and a CLT', fontsize=8, va='top')
    fig.text(.40, .985, '(b) Shared concepts', fontsize=8, va='top')
    fig.text(.69, .985, '(c) Agreement and the model', fontsize=8, va='top')
    save(fig, 'ux_overview')


def mechanism():
    s = json.loads((CIRC / 'CIRCUIT_SUMMARY.json').read_text())
    o, r = {**s['oracle'], **s['oracle_g3']}, s['reverse']
    targets = [('gemma_clt426k', 'Gemma-2 CLT 426k', GEMMA), ('gemma_clt2500k', 'Gemma-2 CLT 2.5M', '#6f93ad'),
               ('llama_clt524k', 'Llama CLT 524k', LLAMA), ('g3_clt262k', 'Gemma-3 CLT 262k', GEMMA3)]
    ticks = ['G2\n426k', 'G2\n2.5M', 'Llama\n524k', 'G3\n262k']
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.1), gridspec_kw=dict(width_ratios=[.9, .95, 1.65, .75]))
    # (a) Later-layer share of the squared write norm of the CLT features active at each supernode site.
    ax = axes[0]
    for k, (key, label, color) in enumerate(targets):
        v = np.array([p[2] for p in o[key]['pairs']])
        jitter = np.random.default_rng(k).uniform(-.18, .18, len(v))
        ax.scatter(k + jitter, v, s=3, color=color, alpha=.45, lw=0)
        ax.plot([k - .28, k + .28], [np.median(v)] * 2, color=INK, lw=1.)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(ticks, fontsize=6.4)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .5, 1])
    ax.set_yticklabels(['0', '.5', '1'])
    ax.set_ylabel('Later-layer share', fontsize=7.2)
    ax.set_title('(a) CLT write support', fontsize=7.6, pad=3, loc='left')
    clean(ax)
    # (b) Residual at the edited layer against the residual of whole features over all layers, per supernode.
    ax = axes[1]
    for key, label, color in [targets[0], targets[2], targets[3]]:
        p = np.array(o[key]['pairs'])
        ax.scatter(p[:, 0], p[:, 1], s=4, color=color, alpha=.6, lw=0, label=label)
    ax.plot([0, 1], [0, 1], color='.5', lw=.6, ls=':')
    rnd = o['gemma_clt426k']['site_residual_random']['median']
    ax.axhline(rnd, color='.55', lw=.6, ls='--')
    ax.text(.02, rnd + .015, 'random direction', fontsize=6.3, color='.4', va='bottom')
    ax.set_xlim(0, 1)
    ax.set_ylim(.4, 1.05)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(['0', '.25', '.5', '.75', '1'])
    ax.set_yticks([.4, .6, .8, 1])
    ax.set_yticklabels(['.4', '.6', '.8', '1'])
    ax.set_xlabel('Edited-layer residual', fontsize=7.2)
    ax.set_ylabel('All-layers residual', fontsize=7.2)
    ax.set_title('(b) Site residual', fontsize=7.6, pad=3, loc='left')
    clean(ax)
    # (c) Effect error of the reconstructed deletion as layer-l slices, whole features and whole features with frozen
    # MLP outputs.
    ax = axes[2]
    kinds = [('nrmse_slice', 'Slices'), ('nrmse_whole', 'Same\namounts'), ('nrmse_whole_frozen', 'Same,\nfrozen'),
             ('nrmse_whole_best', 'Best\nwhole'), ('nrmse_response', 'With\nresponses')]
    for k, (key, label, color) in enumerate(targets):
        for j, (m, _) in enumerate(kinds):
            q = o[key][m]
            x = j + (k - 1.5) * .18
            ax.plot([x, x], [q['q25'], q['q75']], color=color, lw=1.)
            ax.scatter([x], [q['median']], s=12, color=color, zorder=3, lw=0, label=label if j == 0 else None)
    ax.axhline(1, color='.55', lw=.6, ls='--')
    ax.set_yscale('log')
    ax.set_ylim(.2, 10)
    ax.set_yticks([.2, .5, 1, 2, 5, 10])
    ax.set_yticklabels(['.2', '.5', '1', '2', '5', '10'])
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels([k[1] for k in kinds], fontsize=6.0)
    ax.set_ylabel('Effect error (nRMSE)', fontsize=7.2)
    ax.set_title('(c) Reconstructed deletion', fontsize=7.6, pad=3, loc='left')
    clean(ax)
    # (d) Converse direction: residual of CLT graph units in the PLT, norm-weighted over writes.
    ax = axes[3]
    for k, (key, label, color) in enumerate([('gemma', 'Gemma', GEMMA), ('llama', 'Llama', LLAMA),
                                              ('gemma3', 'Gemma-3', GEMMA3)]):
        for j, m in enumerate(['residual_own', 'residual_later']):
            ax.bar(j + (k - 1) * .27, r[key][m], width=.25, color=color, lw=0, label=label if j == 0 else None)
    ax.axhline(r['gemma']['residual_random']['median'], color='.55', lw=.6, ls='--')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Own\nlayer', 'Later\nlayers'], fontsize=6.6)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, .5, 1])
    ax.set_yticklabels(['0', '.5', '1'])
    ax.set_ylabel('Residual in the PLT', fontsize=7.2)
    ax.set_title('(d) Converse', fontsize=7.6, pad=3, loc='left')
    clean(ax)
    handles = [Line2D([], [], ls='', marker='o', ms=4, mfc=c, mec=c, label=n) for _, n, c in targets]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, fontsize=6.8, handletextpad=.2,
               columnspacing=1.4, bbox_to_anchor=(.5, 1.0))
    fig.subplots_adjust(left=.06, right=.995, top=.80, bottom=.19, wspace=.42)
    save(fig, 'ux_mechanism')


def main():
    global PREVIEW
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--figures', nargs='+', default=['overview', 'mechanism', 'transfer'])
    parser.add_argument('--preview', type=Path, default=None, help='also write PNG previews to this directory')
    args = parser.parse_args()
    PREVIEW = args.preview
    style()
    for name in args.figures:
        globals()[name]()


if __name__ == '__main__':
    main()
