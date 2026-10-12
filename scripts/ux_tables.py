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
    s = json.loads((CIRC / 'DEPTH_REFEREE_SUMMARY.json').read_text())
    ci = lambda v: f'{fmt(v[0])} [{fmt(v[1])}, {fmt(v[2])}]'
    runs = [('CIRC_DR_GEMMA_CLT426K_v2', 'Gemma-2'), ('CIRC_DR_LLAMA_CLT524K_v2', 'Llama'), ('CIRC_DR_G3_CLT262K_v2', 'Gemma-3')]
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
        ('Early units collapsed', [f"{early[f'CIRC_NAT_DR_{tag}_v1']['collapsed']}/{early[f'CIRC_NAT_DR_{tag}_v1']['tested']}"
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
    e = json.loads((CIRC / 'EDGE_SUMMARY.json').read_text())
    ci = lambda v: f'{fmt(v[0])} [{fmt(v[1])}, {fmt(v[2])}]'
    lines = []
    for m, name, tags in [('gemma', 'Gemma-2', [('clt', 'GEMMA_CLT426K'), ('plt', 'GEMMA_PLT')]),
                          ('llama', 'Llama', [('clt', 'LLAMA_CLT524K'), ('plt', 'LLAMA_PLT')]),
                          ('g3', 'Gemma-3', [('clt', 'G3_CLT262K'), ('plt', 'G3_PLT')])]:
        for k, (kind, tag) in enumerate(tags):
            rows = [r for r in json.loads((runs / f'CIRC_EDGE_{tag}_v1' / 'EDGE_REFEREE_ROWS.json').read_text())['rows']
                    if abs(r['claimed']) >= .1 * abs(r['v_pre']) and abs(r['claimed']) > 0]
            ratio = sorted(r['model'] / r['claimed'] for r in rows)
            med = ratio[len(ratio) // 2] if len(ratio) % 2 else (ratio[len(ratio) // 2 - 1] + ratio[len(ratio) // 2]) / 2
            early = sum(abs(r['claimed']) for r in rows if r['u_layer'] <= 1) / sum(abs(r['claimed']) for r in rows)
            d = e[f'{m if m != "g3" else "g3"}-{kind}']
            lines.append(' & '.join([name if k == 0 else '', kind.upper(), str(len(rows)), fmt(med), f'{100 * early:.0f}\\%',
                                     ci(d['from layers 0-1']['median']), ci(d['from layers 2+']['median'])]) + '\\\\')
        if m != 'g3':
            lines.append('\\midrule')
    head = 'Model & Dict. & Edges & Confirmed & From 0--1 & Leaving 0--1 & Leaving 2+\\\\'
    (TAB / 'ux_edges.tex').write_text('\\begin{tabular}{@{}llrrrll@{}}\n\\toprule\n' + head + '\n\\midrule\n' +
                                      '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', type=Path, default=Path('D:/CCAD_Storage/runs/circuits'))
    args = parser.parse_args()
    s = json.loads((CIRC / 'CIRCUIT_SUMMARY.json').read_text())
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
