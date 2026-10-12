import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DESCRIPTION = ('One summary of every circuit run used in the paper: medians and quartiles over supernodes or units '
               'with a source effect, answer agreement, the per-supernode residual pairs of the write-support test, '
               'the converse direction, CLT-to-CLT transfer, prompt generalization and dictionary similarity.')

ORACLE = {'gemma_plt_r1': 'CIRC_P1E_ORACLE_GEMMA_PLT_TO_PLTL0R1_v6',
          'gemma_plt_r2': 'CIRC_P1E_ORACLE_GEMMA_PLT_TO_PLTL0R2_v5',
          'gemma_plt_r3': 'CIRC_P1E_ORACLE_GEMMA_PLT_TO_PLTL0R3_v5',
          'gemma_plt_r4': 'CIRC_P1E_ORACLE_GEMMA_PLT_TO_PLTL0R4_v5',
          'gemma_clt426k': 'CIRC_P1E_ORACLE_GEMMA_PLT_TO_CLT426K_v9',
          'gemma_clt2500k': 'CIRC_P1E_ORACLE_GEMMA_PLT_TO_CLT2500K_v9',
          'llama_clt524k': 'CIRC_P1E_ORACLE_LLAMA_PLT_TO_CLT524K_v4'}
# Gemma-3-1B with Gemma Scope 2 dictionaries: units are the 20 most influential feature nodes of the PLT's own graphs.
ORACLE_G3 = {'g3_plt_big': 'CIRC_G3_ORACLE_PLT_TO_PLTBIG_v1', 'g3_clt262k': 'CIRC_G3_ORACLE_PLT_TO_CLT262K_v2'}
# Response-aware test: whole target features fitted to the complete MLP-output change at the edited positions.
RESPONSE = {'gemma_plt_r2': 'CIRC_RESP_GEMMA_PLT_TO_PLTL0R2_v1', 'gemma_clt426k': 'CIRC_RESP_GEMMA_PLT_TO_CLT426K_v1',
            'gemma_clt2500k': 'CIRC_RESP_GEMMA_PLT_TO_CLT2500K_v1', 'llama_clt524k': 'CIRC_RESP_LLAMA_PLT_TO_CLT524K_v1',
            'g3_clt262k': 'CIRC_RESP_G3_PLT_TO_CLT262K_v1', 'g3_plt_big': 'CIRC_RESP_G3_PLT_TO_PLTBIG_v1'}
# Coverage on circuits: fits that add the explanation's own uses on prompt variants, compared on the graphs with
# variants with the generic-text fits (FITS), matching and the best edit.
# Entries: covering run, generic-text run (directory and rows file), oracle run key and best-edit metric. The CLT rows
# use the expanded candidate pool (same-layer correlation plus earlier-read features by cosine) for both fits; the
# same-layer-only CLT runs are kept as their own entries.
COVER = {'gemma_plt_r2': ('CIRC_COV_GEMMA_PLT_TO_PLTL0R2_v1', 'CIRC_P1_GEMMA_PLT_TO_PLTL0R2_v2_best', 'gemma_plt_r2',
                          'nrmse_whole'),
         'gemma_clt426k_whole': ('CIRC_COVX_GEMMA_PLT_TO_CLT426K_covering_whole_v1',
                                 'CIRC_COVX_GEMMA_PLT_TO_CLT426K_generic_whole_v1', 'gemma_clt426k', 'nrmse_whole_best'),
         'gemma_clt426k_slices': ('CIRC_COVX_GEMMA_PLT_TO_CLT426K_covering_slices_v1',
                                  'CIRC_COVX_GEMMA_PLT_TO_CLT426K_generic_slices_v1', 'gemma_clt426k', 'nrmse_slice'),
         'samelayer_clt426k_whole': ('CIRC_COV_GEMMA_PLT_TO_CLT426K_v1', 'CIRC_P1_GEMMA_PLT_TO_CLT426K_v2_pooled',
                                     'gemma_clt426k', 'nrmse_whole_best'),
         'samelayer_clt426k_slices': ('CIRC_COV_GEMMA_PLT_TO_CLT426K_slices_v1', 'CIRC_P1_GEMMA_PLT_TO_CLT426K_v2_slices',
                                      'gemma_clt426k', 'nrmse_slice')}
FITS = {'gemma_plt_r2': 'CIRC_P1_GEMMA_PLT_TO_PLTL0R2_v2_best',
        'gemma_clt426k_whole': 'CIRC_P1_GEMMA_PLT_TO_CLT426K_v2_pooled',
        'gemma_clt426k_slices': 'CIRC_P1_GEMMA_PLT_TO_CLT426K_v2_slices'}
MATCH = {'gemma_clt426k': 'CIRC_P0_GEMMA_PLT_TO_CLT426K_v1'}
REVERSE = {'gemma': 'CIRC_P1R_REVERSE_GEMMA_CLT426K_TO_PLT_v2', 'llama': 'CIRC_P1R_REVERSE_LLAMA_CLT524K_TO_PLT_v1',
           'gemma3': 'CIRC_G3_REVERSE_CLT262K_TO_PLT_v1'}
PAIR = 'CIRC_P1C_CLT426K_TO_CLT2500K_v1'
GENERALIZE = 'CIRC_P2_GENERALIZE_GEMMA_PLT_v1'
GENERALIZE_CLT = 'CIRC_P2_GENERALIZE_GEMMA_CLT426K_v1'
MIN_ENERGY = 1e-3
MIN_DROP = .1


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def quart(v):
    v = np.asarray([x for x in v if x is not None], dtype=float)
    return dict(median=float(np.median(v)), q25=float(np.quantile(v, .25)), q75=float(np.quantile(v, .75)), n=len(v))


def answers(rows, methods):
    # Agreement of the top next token after deletion with the source deletion, over all rows and over the rows whose
    # source deletion changes the model's answer.
    # keeps_base is the share of those rows whose target deletion leaves the original answer in place.
    changed = [r for r in rows if r['top_source'] != r['top_base']]
    return dict(changed=len(changed),
                **{m: dict(agree=float(np.mean([r[f'top_{m}'] == r['top_source'] for r in rows])),
                           same_new=float(np.mean([r[f'top_{m}'] == r['top_source'] for r in changed])) if changed else None,
                           keeps_base=float(np.mean([r[f'top_{m}'] == r['top_base'] for r in changed])) if changed else None)
                   for m in methods})


def load_rows(path):
    return [r for r in json.loads(path.read_text())['rows'] if r.get('source_energy', 0) > MIN_ENERGY]


def boot_median(rows, key, draws=2000, seed=0):
    # 95% interval of the median over supernodes, resampling graphs with replacement.
    rng = np.random.default_rng(seed)
    by_graph = {}
    for r in rows:
        if r.get(key) is not None:
            by_graph.setdefault(r['graph'], []).append(r[key])
    graphs = list(by_graph)
    meds = [np.median(np.concatenate([by_graph[g] for g in rng.choice(graphs, len(graphs))])) for _ in range(draws)]
    return [float(np.quantile(meds, .025)), float(np.quantile(meds, .975))]


def oracle(path):
    rows = load_rows(path)
    keys = ['site_residual', 'site_residual_random', 'site_residual_whole', 'candidate_later_share', 'nrmse_slice',
            'nrmse_whole', 'nrmse_whole_frozen', 'nrmse_source_frozen', 'nrmse_whole_best', 'recon_own_layer_share',
            'recon_layer_lag']
    methods = [m for m in ['slice', 'whole', 'whole_frozen', 'source_frozen', 'whole_best']
               if rows[0].get(f'top_{m}') is not None]
    out = dict(run=path.parent.name, supernodes=len(rows),
               **{k: quart([r.get(k) for r in rows]) for k in keys if rows[0].get(k) is not None},
               answers=answers(rows, methods))
    out['ci'] = {k: boot_median(rows, k) for k in ['site_residual', 'site_residual_whole', 'nrmse_slice', 'nrmse_whole',
                                                    'nrmse_whole_best'] if rows[0].get(k) is not None}
    if rows[0].get('site_residual_whole') is not None:
        out['pairs'] = [[r['site_residual'], r['site_residual_whole'], r['candidate_later_share']] for r in rows
                        if r.get('site_residual_whole') is not None]
    return out


def response(path):
    rows = load_rows(path)
    keys = ['response_share', 'response_residual', 'response_residual_random', 'nrmse_response', 'nrmse_sanity']
    a = answers(rows, ['response', 'sanity'])
    return dict(response_run=path.parent.name, **{k: quart([r[k] for r in rows]) for k in keys},
                response_answers=a, response_ci=boot_median(rows, 'nrmse_response'))


def fit(path):
    data = json.loads(path.read_text())
    rows = load_rows(path)
    methods = [k[6:] for k in rows[0] if k.startswith('nrmse_')]
    return dict(run=path.parent.name, supernodes=len(rows), heldout_text=data['heldout_text'],
                **{m: quart([r[f'nrmse_{m}'] for r in rows]) for m in methods}, answers=answers(rows, methods))


def match(path):
    rows = load_rows(path)
    return dict(run=path.parent.name, supernodes=len(rows), one_to_one=quart([r['nrmse'] for r in rows]),
                match_corr=quart([r['mean_match_corr'] for r in rows]), answers=answers(rows, ['target']))


def reverse(path):
    rows = json.loads(path.read_text())['rows']
    own = [(res, n) for r in rows for k, res, n in r['residual_by_offset'] if k == 0]
    later = [(res, n) for r in rows for k, res, n in r['residual_by_offset'] if k > 0]
    # Residuals of the writes into the unit's own layer and into later layers, weighted by the norm of each write.
    wavg = lambda xs: float(sum(r * n for r, n in xs) / sum(n for _, n in xs))
    sq = lambda xs: sum(n ** 2 for _, n in xs)
    single = [r for r in rows if r['features'] == 1]
    reached = {(r['graph'], name) for r in rows for name, s in r.get('published_supernodes', {}).items() if s >= .1}
    effect = [r for r in rows if r['source_energy'] > MIN_ENERGY]
    # Runs without a registry of published graphs (Gemma-3) have no published share.
    has_registry = json.loads((path.parent / 'config.json').read_text()).get('registry', 'None') != 'None'
    shares = [r['published_share'] for r in single if has_registry and r.get('published_share') is not None]
    return dict(run=path.parent.name, units=len(rows), single_units=len(single), residual_own=wavg(own),
                residual_later=wavg(later), later_share_norm=float(sum(n for _, n in later) / sum(n for _, n in own + later)),
                later_share_sq=float(sq(later) / (sq(own) + sq(later))), residual_random=quart([r['site_residual_random'] for r in effect]),
                nrmse_layerwise=quart([r['nrmse_layerwise'] for r in effect]),
                published_share=quart(shares) if shares else None,
                supernodes_reached=len(reached) if has_registry else None, answers=answers(effect, ['layerwise']))


def pair(path):
    rows = load_rows(path)
    keys = ['best_corr', 'residual_whole', 'residual_whole_random', 'residual_slices', 'nrmse_whole', 'nrmse_slices']
    return dict(run=path.parent.name, units=len(rows), **{k: quart([r[k] for r in rows]) for k in keys},
                answers=answers(rows, ['whole', 'slices']))


def generalize(path):
    rows = [g for g in json.loads(path.read_text())['rows'] if g['drop_original'] >= MIN_DROP and g['variants']]
    ret = [float(np.median([v['drop'] / g['drop_original'] for v in g['variants']])) for g in rows]
    return dict(run=path.parent.name, supernodes=len(rows), graphs=len({g['graph'] for g in rows}),
                variants=sum(len(g['variants']) for g in rows), retention=quart(ret))


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', type=Path, default=Path('D:/CCAD_Storage/runs/circuits'))
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[2] / 'artifacts' / 'circuits' / 'CIRCUIT_SUMMARY.json')
    args = parser.parse_args()
    d = args.runs
    summary = dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION,
                   min_energy=MIN_ENERGY, min_drop=MIN_DROP,
                   oracle={k: oracle(d / v / 'ORACLE_ROWS.json') for k, v in ORACLE.items()},
                   oracle_g3={k: oracle(d / v / 'ORACLE_ROWS.json') for k, v in ORACLE_G3.items()},
                   fits={k: fit(d / v / 'P1_ROWS.json') for k, v in FITS.items()},
                   match={k: match(d / v / 'P0_ROWS.json') for k, v in MATCH.items()},
                   reverse={k: reverse(d / v / 'REVERSE_ROWS.json') for k, v in REVERSE.items()},
                   clt_pair=pair(d / PAIR / 'CLT_PAIR_ROWS.json'),
                   generalize=generalize(d / GENERALIZE / 'GENERALIZE_ROWS.json'),
                   generalize_clt=generalize(d / GENERALIZE_CLT / 'GENERALIZE_ROWS.json'),
                   similarity={m: json.loads((d / f'SIMILARITY_{m.upper()}_PLT.json').read_text())['targets']
                               for m in ['gemma', 'llama']})
    for k, v in RESPONSE.items():
        (summary['oracle'] if k in summary['oracle'] else summary['oracle_g3'])[k].update(
            response(d / v / 'RESPONSE_ROWS.json'))
    # Paired difference of the response-aware effect error between each Gemma-2 CLT and the same-family PLT, per
    # supernode, with a bootstrap over graphs for the median.
    by_key = lambda name: {(r['graph'], r['supernode']): r['nrmse_response'] for r in load_rows(d / RESPONSE[name] / 'RESPONSE_ROWS.json')}
    plt_resp = by_key('gemma_plt_r2')
    summary['response_vs_plt'] = {}
    for name in ['gemma_clt426k', 'gemma_clt2500k']:
        clt_resp = by_key(name)
        keys = sorted(set(plt_resp) & set(clt_resp))
        rows_diff = [dict(graph=k[0], diff=clt_resp[k] - plt_resp[k]) for k in keys]
        summary['response_vs_plt'][name] = dict(n=len(keys), median=float(np.median([r['diff'] for r in rows_diff])),
                                                ci=boot_median(rows_diff, 'diff'),
                                                clt_worse=float(np.mean([r['diff'] > 0 for r in rows_diff])))
    print('response vs plt', summary['response_vs_plt'])
    summary['coverage'] = {}
    for name, (run, generic_run, oracle_key, best_metric) in COVER.items():
        cov = {(r['graph'], r['supernode']): r for r in load_rows(d / run / 'P1_ROWS.json')}
        gen = {(r['graph'], r['supernode']): r for r in load_rows(d / generic_run / 'P1_ROWS.json')}
        orc = {(r['graph'], r['supernode']): r for r in load_rows(d / ORACLE[oracle_key] / 'ORACLE_ROWS.json')}
        out = {}
        for part, keep in [('covered', True), ('uncovered', False)]:
            keys = sorted(k for k, r in cov.items() if r['covered'] == keep and k in gen and k in orc)
            pick = lambda table, metric: [table[k][metric] for k in keys]
            diff = [dict(graph=k[0], diff=cov[k]['nrmse_fitted'] - gen[k]['nrmse_fitted']) for k in keys]
            changed = [k for k in keys if cov[k]['top_source'] != cov[k]['top_base']]
            out[part] = dict(supernodes=len(keys), graphs=len({k[0] for k in keys}),
                             covered_fit=quart(pick(cov, 'nrmse_fitted')), generic_fit=quart(pick(gen, 'nrmse_fitted')),
                             matched=quart(pick(cov, 'nrmse_one_to_one')), best=quart(pick(orc, best_metric)),
                             covered_minus_generic=dict(median=float(np.median([x['diff'] for x in diff])),
                                                        ci=boot_median(diff, 'diff')) if keys else None,
                             changed=len(changed),
                             same_new=dict(covered=float(np.mean([cov[k]['top_fitted'] == cov[k]['top_source'] for k in changed])),
                                           generic=float(np.mean([gen[k]['top_fitted'] == gen[k]['top_source'] for k in changed])),
                                           matched=float(np.mean([cov[k]['top_one_to_one'] == cov[k]['top_source'] for k in changed])))
                             if changed else None)
        summary['coverage'][name] = dict(run=run, generic_run=generic_run, **out)
        print('coverage', name, {p: {k: (round(v['median'], 3) if isinstance(v, dict) and 'median' in v else v)
                                     for k, v in out[p].items()} for p in out})
    # Whether the site residual, computed from the dictionaries on the prompt, predicts the effect error of the
    # reconstructed (slice) deletion, per supernode and target, pooled and within each target.
    pairs = {k: [(r['site_residual'], r['nrmse_slice']) for r in load_rows(d / v / 'ORACLE_ROWS.json')]
             for k, v in ORACLE.items()}
    pooled = [p for v in pairs.values() for p in v]
    summary['residual_predicts_effect'] = dict(
        pooled=spearman(*zip(*pooled)), pairs=len(pooled),
        within={k: spearman(*zip(*v)) for k, v in pairs.items()})
    args.out.write_text(json.dumps(summary, indent=1))
    print('residual predicts effect', summary['residual_predicts_effect'])
    for k, v in {**summary['oracle'], **summary['oracle_g3']}.items():
        print(k, v['supernodes'], {m: round(v[m]['median'], 3) for m in v if isinstance(v[m], dict) and 'median' in v[m]},
              {m: (round(a['same_new'], 2) if a['same_new'] is not None else None) for m, a in v['answers'].items() if isinstance(a, dict)})
    for group in ['fits', 'match', 'reverse']:
        for k, v in summary[group].items():
            print(group, k, {m: (round(x['median'], 3) if isinstance(x, dict) and 'median' in x else x) for m, x in v.items()
                             if m not in ['answers', 'run']})
    print('pair', {m: round(x['median'], 3) for m, x in summary['clt_pair'].items() if isinstance(x, dict) and 'median' in x})
    print('generalize', summary['generalize'])
    print('generalize clt', summary['generalize_clt'])


if __name__ == '__main__':
    main()
