from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]
CIRC = ROOT / 'artifacts' / 'circuits'
TAB = ROOT / 'paper' / 'tables'
DESCRIPTION = 'LaTeX tables of the dictionary-change manuscript, written from CIRCUIT_SUMMARY.json and the oracle rows.'


def fmt(v, digits=2):
    s = f'{v:.{digits}f}'
    return s[1:] if s.startswith('0.') else s


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
                                 fmt(r['nrmse_whole_frozen']['median']), fmt(same['same_new']),
                                 fmt(same['keeps_base'])]) + '\\\\')
    head = ('Target & $n$ & Corr. & Site & Rand. & All & Later & Slices & Same amt. & Best & Frozen & New & Old\\\\')
    (TAB / 'ux_targets.tex').write_text('\\begin{tabular}{@{}lrrrrrrrrrrrr@{}}\n\\toprule\n' + head + '\n\\midrule\n' +
                                       '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', type=Path, default=Path('D:/CCAD_Storage/runs/circuits'))
    args = parser.parse_args()
    s = json.loads((CIRC / 'CIRCUIT_SUMMARY.json').read_text())
    graphs(args.runs)
    targets(s)


if __name__ == '__main__':
    main()
