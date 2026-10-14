from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]
CIRC = ROOT / 'artifacts' / 'circuits'
TAB = ROOT / 'paper' / 'tables'
DESCRIPTION = 'LaTeX tables of the dictionary-change manuscript, written from CIRCUIT_SUMMARY.json and the oracle rows.'


def fmt(v, digits=2):
    s = f'{v:.{digits}f}'
    if float(s) == 0:
        s = s.lstrip('-')
    neg = s.startswith('-')
    s = s.lstrip('-')
    s = s[1:] if s.startswith('0.') else s
    return '$-$' + s if neg else s


def q(d):
    return f"{fmt(d['median'])} [{fmt(d['q25'])}, {fmt(d['q75'])}]"


def graphs(runs):
    # Kept published graphs with their prompts and numbers of supernodes, from the rows of the oracle runs.
    lines = []
    for model, run in [('Gemma-2-2B', 'CIRC_P1E_ORACLE_GEMMA_PLT_TO_CLT426K_v7'),
                       ('Llama-3.2-1B', 'CIRC_P1E_ORACLE_LLAMA_PLT_TO_CLT524K_v2')]:
        rows = json.loads((runs / run / 'ORACLE_ROWS.json').read_text())['rows']
        seen = {}
        for r in rows:
            seen.setdefault(r['graph'], [r['prompt'], 0])[1] += 1
        lines.append(f'\\multicolumn{{3}}{{l}}{{\\textit{{{model}}}, {len(seen)} graphs, {len(rows)} supernodes}}\\\\')
        for g, (prompt, n) in seen.items():
            slug = g.split('/')[1].replace('_', '\\_')
            text = prompt.replace('\\', '\\textbackslash{}').replace('&', '\\&').replace('%', '\\%').replace('$', '\\$')
            text = text.replace('#', '\\#').replace('_', '\\_').replace('"', "''")
            lines.append(f'\\texttt{{{slug}}} & {text} & {n}\\\\')
    body = '\n'.join(lines)
    (TAB / 'ux_graphs.tex').write_text('\\begin{tabular}{@{}llr@{}}\n\\toprule\nGraph & Prompt & Supernodes\\\\\n'
                                      '\\midrule\n' + body + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')


def targets(s):
    sim = s['similarity']
    o = s['oracle']
    rows = [('Gemma Scope PLT, $L_0$ rank 1', 'gemma_plt_r1', sim['gemma']['gemmascope-l0-rank:1']),
            ('Gemma Scope PLT, $L_0$ rank 2', 'gemma_plt_r2', sim['gemma']['gemmascope-l0-rank:2']),
            ('Gemma Scope PLT, $L_0$ rank 3', 'gemma_plt_r3', sim['gemma']['gemmascope-l0-rank:3']),
            ('Gemma Scope PLT, $L_0$ rank 4', 'gemma_plt_r4', sim['gemma']['gemmascope-l0-rank:4']),
            ('Gemma CLT 426k', 'gemma_clt426k', sim['gemma']['mntss/clt-gemma-2-2b-426k']),
            ('Gemma CLT 2.5M', 'gemma_clt2500k', sim['gemma']['mntss/clt-gemma-2-2b-2.5M']),
            ('Llama CLT 524k', 'llama_clt524k', sim['llama']['mntss/clt-llama-3.2-1b-524k'])]
    g3 = s['oracle_g3']
    rows += [('Gemma-3 PLT, larger $L_0$', 'g3_plt_big', None), ('Gemma-3 CLT 262k', 'g3_clt262k', None)]
    lines = []
    for label, key, sm in rows:
        if key == 'g3_plt_big':
            lines.append('\\midrule')
        r = o[key] if key in o else g3[key]
        cross = r.get('site_residual_whole') is not None
        m = lambda k: fmt(r[k]['median']) if cross else '--'
        # For a PLT the slice, whole and best whole-feature deletions coincide.
        best = r['nrmse_whole_best'] if cross else r['nrmse_whole']
        same = r['answers']['whole_best' if cross else 'whole']
        lines.append(' & '.join([label, str(r['supernodes']), fmt(sm['median']) if sm else '--', fmt(r['site_residual']['median']),
                                 fmt(r['site_residual_random']['median']), m('site_residual_whole'),
                                 m('candidate_later_share'), m('nrmse_slice'), m('nrmse_whole'), fmt(best['median']),
                                 fmt(r['nrmse_response']['median']) if 'nrmse_response' in r else '--',
                                 fmt(r['nrmse_whole_frozen']['median']), fmt(same['same_new']),
                                 fmt(same['keeps_base'])]) + '\\\\')
    head = ('Target & $n$ & Corr. & Site & Rand. & All & Later & Slices & Same amt. & Best & Resp. & Frozen & New & Old\\\\')
    (TAB / 'ux_targets.tex').write_text('\\begin{tabular}{@{}lrrrrrrrrrrrrr@{}}\n\\toprule\n' + head + '\n\\midrule\n' +
                                       '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')


def coverage(s):
    # Both conditions on the same circuit explanations: effect error (and the share of answer-changing supernodes
    # that keep the new answer) on the published prompts of the graphs with prompt variants.
    c = s['coverage']
    rows = [('PLT family', c['gemma_plt_r2']['covered']),
            ('CLT, whole', c['gemma_clt426k_whole']['covered']),
            ('CLT, slices', c['gemma_clt426k_slices']['covered'])]
    pct = lambda v: f'{100 * v:.0f}\\%'
    lines = []
    for label, r in rows:
        sn = r['same_new']
        lines.append(' & '.join([label, f"{fmt(r['matched']['median'])} ({pct(sn['matched'])})",
                                 f"{fmt(r['generic_fit']['median'])} ({pct(sn['generic'])})",
                                 f"{fmt(r['covered_fit']['median'])} ({pct(sn['covered'])})",
                                 fmt(r['best']['median'])]) + '\\\\')
    head = 'Target & Matched & Generic & Covering & Best\\\\'
    (TAB / 'ux_coverage.tex').write_text('\\begin{tabular}{@{}lllll@{}}\n\\toprule\n' + head + '\n\\midrule\n' +
                                         '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')


def referee(s):
    # Median ratio of the effect of deleting what a unit writes to the effect of removing what it reads, among units
    # whose write deletion has a substantial effect, with the share of units above ten and the number of units.
    rw = s['read_write']
    cols = [('plt_published', 'all'), ('plt_own', 'all'), ('clt_own', 'later'), ('clt_own', 'early')]
    lines = []
    for m, name in [('gemma', 'Gemma-2'), ('llama', 'Llama'), ('gemma3', 'Gemma-3')]:
        cells = []
        for key, band in cols:
            if key not in rw[m]:
                cells.append('--')
                continue
            v = rw[m][key][band]
            cells.append(f"{v['ratio']:.1f} ({100 * v['over_10']:.0f}\\%)")
        lines.append(' & '.join([name] + cells) + '\\\\')
    head = ('Model & PLT publ. & PLT own & CLT, 2+ & CLT, 0--1\\\\')
    (TAB / 'ux_referee.tex').write_text('\\begin{tabular}{@{}lllll@{}}\n\\toprule\n' + head + '\n\\midrule\n' +
                                        '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')


def confirm():
    # Medians over CLT units of the fraction of their claimed writes that the model removes: from their own layer when
    # their read is removed, and from later layers when their read, the PLT units of the prompt, or random PLT
    # features are removed.
    a = json.loads((CIRC / 'AGREEMENT_VS_REFEREE.json').read_text())
    labels = [('early', '0--1'), ('later_shared', '2+, shared'), ('later_other', '2+, CLT only')]
    lines = []
    for m, name in [('gemma', 'Gemma-2'), ('llama', 'Llama'), ('gemma3', 'Gemma-3')]:
        for k, (g, label) in enumerate(labels):
            v = a[m]['groups'][g]
            cells = [name if k == 0 else '', label, str(v['units'])] + [fmt(v[c]) for c in ('own_u', 'confirm_u', 'later_s', 'later_r')]
            lines.append(' & '.join(cells) + '\\\\')
        if m != 'gemma3':
            lines.append('\\midrule')
    head = ('& & & \\multicolumn{1}{c}{Own} & \\multicolumn{3}{c}{Later layers}\\\\\n'
            '\\cmidrule(lr){4-4}\\cmidrule(lr){5-7}\n'
            'Model & Read at & $n$ & Input & Input & PLT & Rand.\\\\')
    (TAB / 'ux_confirm.tex').write_text('\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\n' + head + '\n\\midrule\n' +
                                        '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')


def bands():
    # Removed fractions by read layer with graph-bootstrap intervals, from summarize_depth_referee.py.
    s = json.loads((CIRC / 'DEPTH_REFEREE_SUMMARY_v5.json').read_text())
    ci = lambda v: f'{fmt(v[0])} [{fmt(v[1])}, {fmt(v[2])}]'
    runs = REMOVAL_RUNS
    lines = []
    for run, name in runs:
        d = s[run]
        first = True
        for band, label in [('layers 0-1', 'CLT, 0--1'), ('layers 2-12', 'CLT, 2--12'), ('layers 13+', 'CLT, 13+')]:
            if band not in d:
                continue
            v = d[band]
            lines.append(' & '.join([name if first else '', label, str(v['units']), ci(v['own_layer_u']), ci(v['later_u']),
                                     ci(v['later_a']), ci(v['later_ra'])]) + '\\\\')
            first = False
        for band, label in [('PLT layers 0-4', 'PLT, 0--4'), ('PLT layers 13+', 'PLT, 13+')]:
            v = d[band]
            lines.append(' & '.join(['', label, str(v['units']), ci(v['own_layer']), '', '', '']) + '\\\\')
        if name != 'Gemma-3':
            lines.append('\\midrule')
    head = ('Model & Units & $n$ & Own layer & Own read & PLT units & Random\\\\')
    (TAB / 'ux_bands.tex').write_text('\\begin{tabular}{@{}llrllll@{}}\n\\toprule\n' + head + '\n\\midrule\n' +
                                      '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')


def natural(runs):
    # The unit test on the natural-text prompts against the published prompts, one column per model.
    nat = json.loads((CIRC / 'NAT_AGREEMENT_VS_REFEREE.json').read_text())
    nat01 = json.loads((CIRC / 'NAT_AGREEMENT_VS_REFEREE_EFFECT01.json').read_text())
    pub = json.loads((CIRC / 'AGREEMENT_VS_REFEREE.json').read_text())
    early = json.loads((CIRC / 'NAT_EARLY_UNITS.json').read_text())
    plt = json.loads((CIRC / 'NAT_PLT_SIDE.json').read_text())
    plt_pub = json.loads((CIRC / 'PLT_SIDE.json').read_text())
    models = [('gemma', 'GEMMA_CLT426K'), ('llama', 'LLAMA_CLT524K'), ('gemma3', 'G3_CLT262K')]
    ci = lambda v: f'{fmt(v[0])} [{fmt(v[1])}, {fmt(v[2])}]'

    def shares(tag):
        rows = json.loads((runs / f'CIRC_NAT_REVERSE_{tag}_v1' / 'REVERSE_ROWS.json').read_text())['rows']
        units = [(int(r['unit'][1:].split('F')[0]), [res for k, res, _ in r['residual_by_offset'] if k == 0])
                 for r in rows if r['features'] == 1]
        units = [(l, own[0]) for l, own in units if own]
        n_early = sum(1 for l, _ in units if l <= 1)
        later = [res < .4 for l, res in units if l >= 2]
        return n_early / len(units), sum(later) / len(later)

    sh = {m: shares(tag) for m, tag in models}
    lines = [
        ('CLT units read at layers 0--1', [f'{100 * sh[m][0]:.0f}\\%' for m, _ in models]),
        ('Later CLT units shared', [f'{100 * sh[m][1]:.0f}\\%' for m, _ in models]),
        ('Shared minus other, own input', [ci(nat[m]['robust_later']['confirm_u']['shared_minus_other']) for m, _ in models]),
        ('\\quad with effect above .1', [ci(nat01[m]['robust_later']['confirm_u']['shared_minus_other']) for m, _ in models]),
        ('\\quad published prompts', [ci(pub[m]['robust_later']['confirm_u']['shared_minus_other']) for m, _ in models]),
        ('Partial Spearman, own input', [ci(nat[m]['robust_later']['confirm_u']['partial_spearman_layer_norm']) for m, _ in models]),
        ('Early units, own input', [fmt(nat[m]['groups']['early']['confirm_u']) for m, _ in models]),
        ('Early stand-ins', [f"{early[f'CIRC_NAT_DR_{tag}_v1']['collapsed']}/{early[f'CIRC_NAT_DR_{tag}_v1']['tested']}"
                                   for m, tag in models]),
        ('PLT side, partial Spearman', [ci(plt[m]['all']['partial_spearman_layer_norm']) for m, _ in models]),
        ('\\quad published prompts', [ci(plt_pub[m]['all']['partial_spearman_layer_norm']) for m, _ in models]),
    ]
    body = '\n'.join(' & '.join([label] + cells) + '\\\\' for label, cells in lines)
    (TAB / 'ux_natural.tex').write_text('\\begin{tabular}{@{}llll@{}}\n\\toprule\n & Gemma-2 & Llama & Gemma-3\\\\\n'
                                        '\\midrule\n' + body + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')


def edges(runs):
    # The edge test: per model and dictionary, the edges that enter, the median confirmed share overall and for edges
    # leaving units read at layers 0-1 or later, and the share of the claimed change carried by edges from layers 0-1.
    e = json.loads((CIRC / 'EDGE_SUMMARY_v2.json').read_text())
    ci = lambda v: f'{fmt(v[0])} [{fmt(v[1])}, {fmt(v[2])}]'
    lines = []
    # Gemma-3's CLT reads the MLP input; its second row carries the removal to the residual stream.
    for m, name, kinds in [('gemma', 'Gemma-2', [('clt', 'CLT', 'CIRC_EDGE_GEMMA_CLT426K_v1'), ('plt', 'PLT', 'CIRC_EDGE_GEMMA_PLT_v1')]),
                           ('llama', 'Llama', [('clt', 'CLT', 'CIRC_EDGE_LLAMA_CLT524K_v1'), ('plt', 'PLT', 'CIRC_EDGE_LLAMA_PLT_v1')]),
                           ('g3', 'Gemma-3', [('clt-res', 'CLT', 'CIRC_EDGE_G3_CLT262K_RES_v3'),
                                              ('clt', 'CLT, MLP in', 'CIRC_EDGE_G3_CLT262K_v1'), ('plt', 'PLT', 'CIRC_EDGE_G3_PLT_v1')])]:
        for k, (kind, label, run) in enumerate(kinds):
            rows = [r for r in json.loads((runs / run / 'EDGE_REFEREE_ROWS.json').read_text())['rows']
                    if abs(r['claimed']) >= .1 * abs(r['v_pre']) and abs(r['claimed']) > 0]
            ratio = sorted(r['model'] / r['claimed'] for r in rows)
            med = ratio[len(ratio) // 2] if len(ratio) % 2 else (ratio[len(ratio) // 2 - 1] + ratio[len(ratio) // 2]) / 2
            early = sum(abs(r['claimed']) for r in rows if r['u_layer'] <= 1) / sum(abs(r['claimed']) for r in rows)
            d = e[f'{m}-{kind}']
            lines.append(' & '.join([name if k == 0 else '', label, str(len(rows)), fmt(med), f'{100 * early:.0f}\\%',
                                     ci(d['from layers 0-1']['median']), ci(d['from layers 2+']['median'])]) + '\\\\')
        if m != 'g3':
            lines.append('\\midrule')
    head = 'Model & Dict. & Edges & Confirmed & From 0--1 & Leaving 0--1 & Leaving 2+\\\\'
    (TAB / 'ux_edges.tex').write_text('\\begin{tabular}{@{}llrrrll@{}}\n\\toprule\n' + head + '\n\\midrule\n' +
                                      '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')


REMOVAL_RUNS = [('CIRC_DR_GEMMA_CLT426K_TOK_v1', 'Gemma-2'), ('CIRC_DR_LLAMA_CLT524K_TOK_v1', 'Llama'),
                ('CIRC_DR_G3_CLT262K_RES_TOK_v1', 'Gemma-3')]


def removals():
    # Main table: medians over the CLT's own top units read at layers 0-1 and later of the share of their claimed writes
    # that each removal takes out, from summarize_depth_referee.py.
    s = json.loads((CIRC / 'DEPTH_REFEREE_SUMMARY_v5.json').read_text())
    rows = [('Own layer, input removed', 'own_layer_u'), ('Later, input removed', 'later_u'),
            ('\\quad at all active positions', 'later_u_all'), ('\\quad projection to 0', 'later_u_zero'),
            ('\\quad projection to mean', 'later_u_mean'), ('\\quad bounded share $S$', 'bounded_u'),
            ('Later, token replaced', 'later_u_tok'), ('PLT units minus random', 'later_a_minus_ra')]
    lines = []
    for label, key in rows:
        cells = [label]
        for run, _ in REMOVAL_RUNS:
            for band in ('layers 0-1', 'layers 2+'):
                cells.append(fmt(s[run][band][key][0]))
        lines.append(' & '.join(cells) + '\\\\')
    units = ['Units'] + [str(s[run][band]['units']) for run, _ in REMOVAL_RUNS for band in ('layers 0-1', 'layers 2+')]
    head = (' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{name}}}' for _, name in REMOVAL_RUNS) + '\\\\\n'
            '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n'
            'Read at layer & 0--1 & 2+ & 0--1 & 2+ & 0--1 & 2+\\\\')
    (TAB / 'gt_removals.tex').write_text('\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n' + head + '\n\\midrule\n' +
                                         '\n'.join(lines) + '\n\\midrule\n' + ' & '.join(units) + '\\\\' +
                                         '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')
    # Appendix: the same medians with 95% graph-bootstrap intervals.
    ci = lambda v: f'{fmt(v[0])} [{fmt(v[1])}, {fmt(v[2])}]'
    lines = []
    for run, name in REMOVAL_RUNS:
        for k, (band, label) in enumerate([('layers 0-1', '0--1'), ('layers 2+', '2+')]):
            v = s[run][band]
            lines.append(' & '.join([name if k == 0 else '', label] + [ci(v[key]) for key in
                                     ('later_u_all', 'later_u_zero', 'later_u_mean', 'bounded_u', 'later_u_tok')]) + '\\\\')
    head = 'Model & Read at & All positions & Projection to 0 & Projection to mean & Bounded share & Token replaced\\\\'
    (TAB / 'gt_alternatives.tex').write_text('\\begin{tabular}{@{}lllllll@{}}\n\\toprule\n' + head + '\n\\midrule\n' +
                                             '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')


def toy(runs):
    # The formal run of the toy validation: per circuit, features and rows that enter, and medians with quartiles of
    # the removed share (input at threshold, read projection to zero), the bounded share and the PLT-unit removal.
    t = json.loads((runs / 'CIRC_TOY_REFEREE_FORMAL_v1' / 'TOY_REFEREE.json').read_text())['summary']
    q = lambda v: f'{fmt(v[0])} [{fmt(v[1])}, {fmt(v[2])}]'
    rows = [('A two-step', 'Two-step, collapsed'), ('B amplified', 'Amplified'), ('F erased copy', 'Surviving copy'),
            ('E parallel', 'Every layer'), ('D redundant', 'Unused copy'), ('C constant', 'Constant')]
    lines = [' & '.join([label, str(t[k]['features']), str(t[k]['rows']), q(t[k]['removed_u']), q(t[k]['removed_u_zero']),
                         q(t[k]['bounded_u']), q(t[k]['removed_s'])]) + '\\\\' for k, label in rows]
    head = 'Circuit & Feat. & Rows & Input removed & Projection to 0 & Bounded $S$ & PLT later units\\\\'
    (TAB / 'gt_toy.tex').write_text('\\begin{tabular}{@{}lrrllll@{}}\n\\toprule\n' + head + '\n\\midrule\n' +
                                    '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')


def toy_counterfactual(runs):
    # The second formal run of the toy validation, with the redundant circuit and the counterfactual input: per
    # circuit, medians with quartiles of the input removal, the joint removal of all features active at the site, the
    # counterfactual input, and the share of the latter that restoring the unit's read projection or the span of all
    # active read directions brings back.
    t = json.loads((runs / 'CIRC_TOY_REFEREE_FORMAL_v3' / 'TOY_REFEREE.json').read_text())['summary']
    q = lambda v: f'{fmt(v[0])} [{fmt(v[1])}, {fmt(v[2])}]' if v else '--'
    rows = [('A two-step', 'Two-step, collapsed'), ('B amplified', 'Amplified'), ('F erased copy', 'Surviving copy'),
            ('E parallel', 'Every layer'), ('G both copies', 'Both copies read'), ('D redundant', 'Unused copy'),
            ('C constant', 'Constant')]
    lines = [' & '.join([label, str(t[k]['features']), q(t[k]['removed_u']), q(t[k]['removed_joint']),
                         q(t[k]['removed_tok']), q(t[k]['sufficiency']), q(t[k]['sufficiency_joint'])]) + '\\\\'
             for k, label in rows]
    head = 'Circuit & Feat. & Input removed & All site features off & Counterfactual & Read restored & Span restored\\\\'
    (TAB / 'gt_toy_counterfactual.tex').write_text('\\begin{tabular}{@{}lrlllll@{}}\n\\toprule\n' + head +
                                                   '\n\\midrule\n' + '\n'.join(lines) +
                                                   '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')


def sufficiency():
    # Joint removal and sufficiency in the real models: CLT units read at layers 0-1 outside the always-active features
    # and units read later, medians with quartiles.
    s = json.loads((CIRC / 'SUFFICIENCY.json').read_text())
    q = lambda v: f'{fmt(v[0])} [{fmt(v[1])}, {fmt(v[2])}]'
    lines = []
    for m, name in [('gemma', 'Gemma-2'), ('llama', 'Llama'), ('gemma3', 'Gemma-3')]:
        for k, (group, label) in enumerate([('early_other', '0--1'), ('later', '2+')]):
            v = s[m][group]
            lines.append(' & '.join([name if k == 0 else '', label, f"{v['tested']}/{v['units']}", q(v['removed_u']),
                                     q(v['removed_joint']), q(v['removed_tok']), q(v['sufficiency']),
                                     q(v['sufficiency_joint'])]) + '\\\\')
    head = ('Model & Read at & Units & Input removed & All site features off & Token replaced & Read restored & '
            'Span restored\\\\')
    (TAB / 'gt_sufficiency.tex').write_text('\\begin{tabular}{@{}llrlllll@{}}\n\\toprule\n' + head + '\n\\midrule\n' +
                                            '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')


def filter_prediction():
    # v14: intervention prediction error of graphs filtered by the unit test against the unfiltered graph, a random
    # filter of the same size, the layer filter and the PLT graph, per model, for resampling (primary) and input removal
    # (second): medians over held-out prompts, the best baseline in bold, and the paired difference to it with its
    # bootstrap interval. The third row of a model is the median change of the answer logit by the first ten units of
    # each list under resampling (coverage).
    s = json.loads((CIRC / 'FILTER_PREDICTION_SUMMARY.json').read_text())['models']
    lists = ['test', 'all', 'random', 'layer', 'plt']
    lines = []
    for m, name in [('llama', 'Llama'), ('gemma', 'Gemma-2'), ('gemma3', 'Gemma-3')]:
        if m not in s:
            continue
        if lines:
            lines.append('\\midrule')
        for k, (kind, label) in enumerate([('rs', 'Resample'), ('rm', 'Remove')]):
            d = s[m][kind]
            cells = [f"\\textbf{{{fmt(d['median'][b])}}}" if b == d['best'] else fmt(d['median'][b]) for b in lists]
            ci = f"{fmt(d['diff'])} [{fmt(d['diff_ci'][0])}, {fmt(d['diff_ci'][1])}]"
            lines.append(' & '.join([name if k == 0 else '', label] + cells + [ci]) + '\\\\')
        c = s[m]['coverage']['median']
        lines.append(' & '.join(['', 'First ten'] + [fmt(c[b]) for b in lists] + ['']) + '\\\\')
    head = ('& & \\multicolumn{5}{c}{Node list} & \\\\\n\\cmidrule(lr){3-7}\n'
            'Model & & Test & All & Random & Layers 2+ & PLT & Test$-$best\\\\')
    (TAB / 'gt_filter.tex').write_text('\\begin{tabular}{@{}llrrrrrl@{}}\n\\toprule\n' + head + '\n\\midrule\n' +
                                       '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')


def utility():
    # Predicting unconfirmed CLT units on the natural-text prompts from a model fitted on the published prompts.
    rows = [('UTILITY_GEMMA.json', 'Gemma-2 426k'), ('UTILITY_GEMMA25.json', 'Gemma-2 2.5M'), ('UTILITY_LLAMA.json', 'Llama'),
            ('UTILITY_G3_RES.json', 'Gemma-3')]
    ci = lambda v: f'{fmt(v[0], 3)} [{fmt(v[1], 3)}, {fmt(v[2], 3)}]'
    lines = []
    for name, label in rows:
        u = json.loads((CIRC / name).read_text())
        lines.append(' & '.join([label, str(u['test_units']), f"{100 * u['test_unconfirmed']:.0f}\\%",
                                 fmt(u['single_auroc']['layer']), fmt(u['single_auroc']['agree']),
                                 fmt(u['auroc_baselines'], 3), ci(u['auroc_gain_from_agreement']),
                                 fmt(u['precision_top20pct_baselines'])]) + '\\\\')
    head = 'Dictionary & Units & Unconf. & Layer & Agreement & Baselines & Gain from agreement & Prec.@20\\%\\\\'
    (TAB / 'gt_utility.tex').write_text('\\begin{tabular}{@{}lrrrrrlr@{}}\n\\toprule\n' + head + '\n\\midrule\n' +
                                        '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')


def completeness():
    # E1: replacement and completeness scores of the CLT's and the PLT's own graphs on the same prompts, medians with the
    # paired difference and its bootstrap interval over prompts, and the median number of active features.
    e = json.loads((CIRC / 'E1_GRAPH_SCORES.json').read_text())
    # Gemma Scope 2 at its other sparsity: the CLT and the PLT with more active features per token.
    e['gemma3_dense'] = json.loads((CIRC / 'E1_GRAPH_SCORES_G3_DENSE.json').read_text())['dense_pair']
    ci = lambda v: f'{fmt(v["diff"])} [{fmt(v["diff_ci"][0])}, {fmt(v["diff_ci"][1])}]'
    k = lambda v: f'{v / 1000:.1f}k'
    lines = []
    for m, name in [('gemma', 'Gemma-2'), ('llama', 'Llama'), ('gemma3', 'Gemma-3'), ('gemma3_dense', 'Gemma-3 dense')]:
        d = e[m]
        lines.append(' & '.join([name, k(d['features']['clt']), k(d['features']['plt']), fmt(d['replacement']['clt']),
                                 fmt(d['replacement']['plt']), ci(d['replacement'])]) + '\\\\')
    head = ('& \\multicolumn{2}{c}{Features} & \\multicolumn{3}{c}{Replacement score}\\\\\n'
            '\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\n'
            'Model & CLT & PLT & CLT & PLT & CLT$-$PLT\\\\')
    (TAB / 'gt_completeness.tex').write_text('\\begin{tabular}{@{}lrrrrl@{}}\n\\toprule\n' + head + '\n\\midrule\n' +
                                             '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', type=Path, default=Path('D:/CCAD_Storage/runs/circuits'))
    args = parser.parse_args()
    s = json.loads((CIRC / 'CIRCUIT_SUMMARY.json').read_text())
    removals()
    toy(args.runs)
    toy_counterfactual(args.runs)
    sufficiency()
    filter_prediction()
    utility()
    completeness()
    graphs(args.runs)
    targets(s)
    coverage(s)
    referee(s)
    confirm()
    bands()
    natural(args.runs)
    edges(args.runs)


if __name__ == '__main__':
    main()
