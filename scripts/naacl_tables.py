from pathlib import Path
import argparse
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts' / 'naacl_sprint_20261010'
TAB = ROOT / 'paper' / 'tables'
DESCRIPTION = 'LaTeX tables of the NAACL manuscript, written from saved sprint analyses.'


def f3(x):
    return f'{x:.3f}'.replace('0.', '.', 1) if x >= 0 else '$-$' + f'{-x:.3f}'.replace('0.', '.', 1)


def ci(lo, hi):
    return f'[{f3(lo)}, {f3(hi)}]'


def settings():
    rows = [
        r'Setting & Model & Explanation & Source SAE & Target SAEs & Correspondence & Fit / held-out \\',
        r'\midrule',
        r'SHIFT & Pythia-70M & 55 features in 3 annotated parts at 11 sites & ReLU 32k, published & TopK 8k, 4 retrained & fixed; membership, 3 early dictionaries & 503 / 1,735 biographies \\',
        r'Infinitive & Pythia-70M & 4 features in predicate and object parts & ReLU 32k, published & TopK 8k, 4 retrained; 16 SAEBench TopK & input-dependent; dictionary & 64 / 96 contexts \\',
        r'Grammar & GPT-2 Medium & 3 functions of 64 features & TopK 8k & TopK 8k, 4 retrained & input-dependent; dictionary & 192 / 384 minimal pairs \\',
        r'Grammar & Qwen3-1.7B & 3 functions of 64 features & Qwen-Scope 32k & Qwen-Scope 32k at other $L_0$ & input-dependent; dictionary & 192 / 384 minimal pairs \\',
        r'SHIFT & Gemma-2-2B & 43 features in 2 annotated parts at 23 sites & Gemma Scope 16k, published & Gemma Scope 65k; Gemma Scope 16k at lowest $L_0$ & fixed; membership only & 1,024 / 1,736 biographies \\',
    ]
    col = r'>{\raggedright\arraybackslash}p'
    body = '\n'.join([r'\begin{tabular}{@{}ll' + f'{col}{{2.7cm}}{col}{{1.8cm}}{col}{{2.7cm}}{col}{{2.2cm}}{col}{{1.9cm}}' + '@{}}',
                      r'\toprule', *rows, r'\bottomrule', r'\end{tabular}'])
    (TAB / 'naacl_settings.tex').write_text(body + '\n')


def delta(c):
    return f"{f3(c['point'])} {ci(*c['interval'])}"


def supervision(entries):
    lines = [r'\begin{tabular}{lccccccc}', r'\toprule',
             r' & \multicolumn{5}{c}{Part-deletion error} & \multicolumn{2}{c}{Part $-$ complete supervision} \\',
             r'\cmidrule(lr){2-6}\cmidrule(lr){7-8}',
             r'Setting & Hungarian & Untrained & Re-applied$^\dagger$ & Complete & Part & Part error & Complete error \\',
             r'\midrule']
    for e in entries:
        errors = [e['hungarian'], e['natural'], e['readout'], e['complete'], e['part']]
        best = min(float(x) for x in errors if x != '--')
        errors = [r'\textbf{' + x + '}' if x != '--' and float(x) == best else x for x in errors]
        cells = errors + [e['delta_part'], e['delta_complete']]
        label = r'\begin{tabular}[t]{@{}l@{}}' + e['label'] + r'\\[-1pt]{\scriptsize ' + e['targets'] + r'}\end{tabular}'
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\[5pt]')
    lines += [r'\bottomrule', r'\end{tabular}']
    (TAB / 'naacl_supervision.tex').write_text('\n'.join(lines) + '\n')


def shift_entry():
    d = json.loads((ART / 'e1_matching' / 'E1_SHIFT_R59PANEL.json').read_text())
    s, c = d['summary'], d['contrasts']['parts-whole']
    return dict(label='SHIFT, Pythia-70M', targets='4 retrained', form='fixed', hungarian=f3(s['hungarian_decoder']['parts_nrmse']),
                natural=f3(s['native']['parts_nrmse']), readout=f3(s['raw_reconstruction']['parts_nrmse']),
                complete=f3(s['whole']['parts_nrmse']), part=f3(s['parts']['parts_nrmse']),
                delta_part=delta(c['parts']), delta_complete=delta(c['full']))


def supervision_entries():
    d = json.loads((ART / 'e2_supervision' / 'SUPERVISION.json').read_text())['settings']
    labels = dict(infinitive=('Infinitive, Pythia-70M', '4 retrained'), infinitive_public=('Infinitive, Pythia-70M', '16 SAEBench'),
                  gpt2=('Grammar, GPT-2 Medium', '4 retrained'), qwen=('Grammar, Qwen3-1.7B', '2 Qwen-Scope'))
    out = []
    for key, (label, targets) in labels.items():
        p, c = d[key]['point'], d[key]['contrasts']['part-complete']
        untrained = p.get('natural', p.get('untrained'))
        out.append(dict(label=label, targets=targets, form=r'input$^\dagger$', hungarian=f3(p['hungarian_decoder']['parts']),
                        natural=f3(untrained['parts']) if untrained else '--', readout=f3(p['readout']['parts']),
                        complete=f3(p['complete']['parts']), part=f3(p['part']['parts']),
                        delta_part=delta(c['parts']), delta_complete=delta(c['complete'])))
    return out


def use():
    d = json.loads((ART / 'e3_shift' / 'E3_SHIFT_USE.json').read_text())
    e1 = json.loads((ART / 'e1_matching' / 'E1_SHIFT_R59PANEL.json').read_text())['summary']
    rows = [('none', 'Unedited', False), ('source', 'Original explanation', False),
            ('raw_reconstruction', 'Re-applied edit$^\\dagger$', True),
            ('hungarian_decoder', 'Hungarian, decoder', True), ('hungarian_activation', 'Hungarian, activation', True),
            ('geometry', 'Top-2 decoder', True), ('geometry_gain', 'Calibrated decoder', True),
            ('whole', 'Complete sup.', True), ('parts', 'Part sup.', True)]
    def head(a, b):
        return r'\begin{tabular}[b]{@{}c@{}}' + a + r'\\' + b + r'\end{tabular}'
    lines = [r'\begin{tabular}{lcccc}', r'\toprule',
             'Method & ' + ' & '.join([head('Part', 'error'), head('Part', 'acc.'), head('Part', 'WG'), head('Pron.', r'$-$name')])
             + r' \\', r'\midrule']
    for key, label, err in rows:
        r, pn = d['results'][key], d['pronouns_minus_names'][key]['mean']['profession']['mean']
        e = f3(e1[key]['parts_nrmse']) if err else '--'
        signed = f"{100 * pn:.1f}" if pn >= 0 else f"$-${100 * -pn:.1f}"
        cells = [e, f"{100 * r['parts_mean']['profession']['mean']:.1f}", f"{100 * r['parts_mean']['worst_group']['mean']:.1f}",
                 signed if key != 'none' else '--']
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
        if key == 'raw_reconstruction':
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    (TAB / 'naacl_use.tex').write_text('\n'.join(lines) + '\n')


def member_information():
    s = json.loads((ROOT / 'paper' / 'data' / 'member_information.json').read_text())['summary']
    rows = [('initial', 'Initial'), ('program', 'Trained'), ('group_uniform', 'Uniform split'),
            ('source_norm_share', 'Contribution split'), ('readout_initial', r'Re-applied edit$^\dagger$'),
            ('readout_trained', r'Re-applied, trained$^\dagger$')]
    lines = [r'\begin{tabular}{lccc}', r'\toprule', r'Columns & Subsets & Strengths & Endpoints \\', r'\midrule']
    for key, label in rows:
        lines.append(label + ' & ' + ' & '.join(f3(s[key][f]['mean']) for f in ['member_subsets', 'participation', 'endpoints'])
                     + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (TAB / 'naacl_member_information.tex').write_text('\n'.join(lines) + '\n')


def single_part():
    d = json.loads((ROOT / 'paper' / 'data' / 'single_part_composition.json').read_text())
    s = d['summary']
    rows = [('single_program', 'Single parts'), ('mixed_program', 'Parts, full and strengths'), ('initial', 'Initial columns'),
            ('source_additive', 'Sum of source parts'), ('readout_initial', r'Re-applied edit$^\dagger$')]
    lines = [r'\begin{tabular}{lcc}', r'\toprule', r'Training requests & Unions & Strengths \\', r'\midrule']
    for key, label in rows:
        lines.append(label + ' & ' + ' & '.join(f3(s[key][f]['response_nrmse']['mean']) for f in ['primary_binary4', 'strength4'])
                     + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (TAB / 'naacl_single_part.tex').write_text('\n'.join(lines) + '\n')


def full_results():
    # Every executor on complete, part and new requests in each setting, the numbers behind Table 2.
    sup = json.loads((ART / 'e2_supervision' / 'SUPERVISION.json').read_text())['settings']
    e1 = json.loads((ART / 'e1_matching' / 'E1_SHIFT_R59PANEL.json').read_text())['summary']
    names = [('hungarian_decoder', 'Hungarian, decoder'), ('hungarian_activation', 'Hungarian, activation'),
             ('natural', 'Natural-text fit'), ('untrained', 'Untrained columns'), ('readout', r'Re-applied edit$^\dagger$'),
             ('gain', 'Gains only, part sup.'), ('complete', 'Complete sup.'), ('part', 'Part sup.')]
    shift_keys = dict(hungarian_decoder='hungarian_decoder', hungarian_activation='hungarian_activation', natural='native',
                      readout='raw_reconstruction', complete='whole', part='parts')
    blocks = [('SHIFT, Pythia-70M, 4 retrained', {k: (e1[v]['full_nrmse'], e1[v]['parts_nrmse'], e1[v]['unions_nrmse'])
                                                   for k, v in shift_keys.items()})]
    labels = dict(infinitive='Infinitive, Pythia-70M, 4 retrained', infinitive_public='Infinitive, Pythia-70M, 16 SAEBench',
                  gpt2='Grammar, GPT-2 Medium, 4 retrained', qwen='Grammar, Qwen3-1.7B, 2 Qwen-Scope')
    for key, label in labels.items():
        p = sup[key]['point']
        blocks.append((label, {k: (p[k]['complete'], p[k]['parts'], p[k]['new']) for k, _ in names if k in p}))
    lines = [r'\begin{tabular}{lccc}', r'\toprule', r'Method & Complete & Parts & New \\']
    for label, rows in blocks:
        lines += [r'\midrule', r'\multicolumn{4}{l}{\textit{' + label + '}}' + r' \\']
        for k, name in names:
            if k in rows:
                lines.append(name + ' & ' + ' & '.join(f3(x) for x in rows[k]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (TAB / 'naacl_full_results.tex').write_text('\n'.join(lines) + '\n')


def shift_parts():
    d = json.loads((ART / 'e1_matching' / 'E1_SHIFT_R59PANEL.json').read_text())
    rows = [('hungarian_decoder', 'Hungarian, decoder'), ('hungarian_activation', 'Hungarian, activation'),
            ('geometry', 'Top-2 decoder'), ('activation_top2', 'Top-2 activation'),
            ('geometry_gain', 'Calibrated decoder'), ('native', 'Natural-text fit'),
            ('raw_reconstruction', r'Re-applied edit$^\dagger$'), ('whole', 'Complete sup.'),
            ('parts_relation', 'Part sup., membership only'), ('parts', 'Part sup.')]
    lines = [r'\begin{tabular}{lccccc}', r'\toprule',
             r' & \multicolumn{3}{c}{Part error} & \multicolumn{2}{c}{Answer agreement} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-6}', r'Method & Pronouns & Names & Assoc. & Complete & Parts \\', r'\midrule']
    for key, label in rows:
        pp = np.mean([t['methods'][key]['parts']['per_query_nrmse'] for t in d['per_target'].values()], axis=0)
        s = d['summary'][key]
        lines.append(label + ' & ' + ' & '.join(f3(x) for x in pp) + f" & {100 * s['full_balanced_agreement']:.1f} & "
                     f"{100 * s['parts_balanced_agreement']:.1f}" + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (TAB / 'naacl_shift_parts.tex').write_text('\n'.join(lines) + '\n')


def dictionary_quality():
    d = json.loads((ART / 'e2_supervision' / 'DICTIONARY_QUALITY.json').read_text())['settings']
    lines = [r'\begin{tabular}{lcccc}', r'\toprule', r'Setting & Before & Complete sup. & Part sup. & Largest drop \\',
             r'\midrule']
    for name, v in d.items():
        lines.append(name + ' & ' + ' & '.join(f"{100 * v[k]:.2f}" for k in ['before', 'complete', 'part', 'largest_drop'])
                     + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (TAB / 'naacl_dictionary_quality.tex').write_text('\n'.join(lines) + '\n')


def use_intervals():
    d = json.loads((ART / 'e3_shift' / 'E3_SHIFT_USE.json').read_text())
    rows = [('source', 'Original explanation'), ('raw_reconstruction', r'Re-applied edit$^\dagger$'),
            ('hungarian_decoder', 'Hungarian, decoder'), ('hungarian_activation', 'Hungarian, activation'),
            ('whole', 'Complete sup.'), ('parts', 'Part sup.')]

    def cell(v):
        return f"{100 * v['mean']:.1f} [{100 * v['ci95'][0]:.1f}, {100 * v['ci95'][1]:.1f}]"
    lines = [r'\begin{tabular}{lcccc}', r'\toprule',
             r'Method & Part acc. & Part WG & Answer agreement & Pron.$-$name \\', r'\midrule']
    for key, label in rows:
        r = d['results'][key]['parts_mean']
        pn = d['pronouns_minus_names'][key]['mean']['profession']
        lines.append(label + ' & ' + ' & '.join([cell(r['profession']), cell(r['worst_group']),
                                                 cell(r['balanced_source_agreement']), cell(pn)]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (TAB / 'naacl_use_intervals.tex').write_text('\n'.join(lines) + '\n')


def gemma_targets(path):
    d = json.loads(path.read_text())['runs']
    return d, [next(k for k in d if 'w65k' in k), next(k for k in d if 'l0low16k' in k)]


def gemma_entries(path):
    d, targets = gemma_targets(path)
    out = []
    for t, label in zip(targets, ['Gemma Scope 65k', r'Gemma Scope 16k, lowest $L_0$']):
        p, c = d[t]['point'], d[t]['contrasts']['part_sup_hungarian-complete_sup_hungarian']
        out.append(dict(label='SHIFT, Gemma-2-2B', targets=label, hungarian=f3(p['hungarian_decoder']['parts']),
                        natural=f3(p['native']['parts']), readout=f3(p['readout']['parts']),
                        complete=f3(p['complete_sup_hungarian']['parts']), part=f3(p['part_sup_hungarian']['parts']),
                        delta_part=delta(c['parts']), delta_complete=delta(c['complete'])))
    return out


def gemma_parts(path):
    d, targets = gemma_targets(path)
    rows = [('hungarian_decoder', 'Hungarian, decoder'), ('hungarian_activation', 'Hungarian, activation'),
            ('top2_decoder', 'Top-2 decoder'), ('native', 'Natural-text fit'), ('readout', r'Re-applied edit$^\dagger$'),
            ('matched_part_sup_hungarian', 'Matched gains, part sup.'), ('complete_sup_hungarian', 'Complete sup.'),
            ('part_sup_hungarian', 'Part sup.'), ('subset_sup_hungarian', 'Part and subset sup.')]
    lines = [r'\begin{tabular}{lcccccc}', r'\toprule',
             r' & \multicolumn{3}{c}{Gemma Scope 65k} & \multicolumn{3}{c}{Gemma Scope 16k, lowest $L_0$} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
             r'Method & Pronouns & Women & Agreement & Pronouns & Women & Agreement \\', r'\midrule']
    for key, label in rows:
        cells = []
        for t in targets:
            pp, ag = d[t]['per_part'][key], d[t]['agreement'][key]
            cells += [f3(pp['pronouns']), f3(pp['female_descriptions']), f"{100 * ag['parts']:.1f}"]
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (TAB / 'naacl_gemma_parts.tex').write_text('\n'.join(lines) + '\n')


def gemma_part_use(path):
    d = json.loads(path.read_text())['runs']
    targets = [next(k for k in d if 'w65k' in k), next(k for k in d if 'l0low16k' in k)]
    rows = [('source', 'Original explanation'), ('readout', r'Re-applied edit$^\dagger$'),
            ('hungarian_decoder', 'Hungarian, decoder'), ('native', 'Natural-text fit'),
            ('complete_sup_hungarian', 'Complete sup.'), ('part_sup_hungarian', 'Part sup.')]
    lines = [r'\begin{tabular}{lcccc}', r'\toprule', r' & \multicolumn{2}{c}{65k} & \multicolumn{2}{c}{16k, lowest $L_0$} \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}', r'Deleted part & Pron. & Women & Pron. & Women \\', r'\midrule']
    for key, label in rows:
        cells = []
        for t in targets:
            o = d[t][key]['original']
            cells += [f"{100 * o['pronouns']['worst_group']:.1f}", f"{100 * o['female_descriptions']['worst_group']:.1f}"]
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
        if key == 'source':
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    (TAB / 'naacl_gemma_part_use.tex').write_text('\n'.join(lines) + '\n')


def gemma_use(path):
    d, targets = gemma_targets(path)
    rows = [('clean', 'Unedited'), ('oracle', 'Trained on balanced data'), ('source', 'Original explanation'),
            ('readout', r'Re-applied edit$^\dagger$'), ('hungarian_decoder', 'Hungarian, decoder'),
            ('hungarian_activation', 'Hungarian, activation'), ('top2_decoder', 'Top-2 decoder'),
            ('native', 'Natural-text fit'), ('matched_part_sup_hungarian', 'Matched gains, part sup.'),
            ('complete_sup_hungarian', 'Complete sup.'), ('part_sup_hungarian', 'Part sup.'),
            ('subset_sup_hungarian', 'Part and subset sup.')]

    def pct(x):
        return f'{100 * x:.1f}'
    lines = [r'\begin{tabular}{lcccccccccc}', r'\toprule',
             r' & \multicolumn{5}{c}{Gemma Scope 65k} & \multicolumn{5}{c}{Gemma Scope 16k, lowest $L_0$} \\',
             r'\cmidrule(lr){2-6}\cmidrule(lr){7-11}',
             r' & \multicolumn{2}{c}{Original} & \multicolumn{3}{c}{Retrained} & \multicolumn{2}{c}{Original} & '
             r'\multicolumn{3}{c}{Retrained} \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}\cmidrule(lr){9-11}',
             r'Method & Prof. & Gender & Prof. & Gender & WG & Prof. & Gender & Prof. & Gender & WG \\', r'\midrule']
    for key, label in rows:
        cells = []
        for t in targets:
            u = d[t]['use'][key]
            o = u['original']
            r = u.get('retrained')
            cells += [pct(o['profession']), pct(o['gender'])]
            cells += [pct(r['profession']), pct(r['gender']), pct(r['worst_group'])] if r else ['--'] * 3
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
        if key in ('source', 'readout'):
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    (TAB / 'naacl_gemma_use.tex').write_text('\n'.join(lines) + '\n')

    # Main-text table: response fidelity and the SHIFT outcome after retraining, with the new dictionary alone.
    main = [r'\begin{tabular}{lcccccccccc}', r'\toprule',
            r' & \multicolumn{5}{c}{Gemma Scope 65k} & \multicolumn{5}{c}{Gemma Scope 16k, lowest $L_0$} \\',
            r'\cmidrule(lr){2-6}\cmidrule(lr){7-11}',
            r'Method & Parts & New & Prof. & Gender & WG & Parts & New & Prof. & Gender & WG \\', r'\midrule']
    for key, label in rows:
        cells = []
        for t in targets:
            p, u = d[t]['point'].get(key), d[t]['use'][key]
            acc = u.get('retrained', u['original'])
            cells += ([f3(p['parts']), f3(p['new'])] if p else ['--', '--'])
            cells += [pct(acc['profession']), pct(acc['gender']), pct(acc['worst_group'])]
        main.append(label + ' & ' + ' & '.join(cells) + r' \\')
        if key in ('source', 'readout'):
            main.append(r'\midrule')
    main += [r'\bottomrule', r'\end{tabular}']
    (TAB / 'naacl_gemma.tex').write_text('\n'.join(main) + '\n')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--gemma', type=Path)
    parser.add_argument('--gemma-use', action='store_true', help='also write the Gemma use tables; needs USE.json in every run')
    parser.add_argument('--gemma-part-use', type=Path)
    args = parser.parse_args()
    settings()
    use()
    member_information()
    single_part()
    full_results()
    shift_parts()
    dictionary_quality()
    use_intervals()
    entries = [shift_entry()] + supervision_entries()
    if args.gemma:
        entries += gemma_entries(args.gemma)
        gemma_parts(args.gemma)
    if args.gemma_use:
        gemma_use(args.gemma)
    if args.gemma_part_use:
        gemma_part_use(args.gemma_part_use)
    supervision(entries)
    print('tables written', len(entries))


if __name__ == '__main__':
    main()
