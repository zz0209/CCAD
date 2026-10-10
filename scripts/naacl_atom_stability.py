from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from naacl_supervision_table import SETTINGS, load, nrmse, per_item

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts' / 'naacl_sprint_20261010'
BULK = Path('D:/CCAD_Storage/runs/naacl_sprint_20261010')
MIRROR = BULK / 'mayo_mirror'
DESCRIPTION = ('Atom stability of the explained features in every target dictionary, measured as the mean decoder cosine '
               'of their Hungarian matches, against the part error of one-to-one matching and of the learned many-to-many '
               'correspondence.')
METHODS = ['hungarian', 'readout', 'complete', 'part']


def json_load(path):
    return json.loads(Path(path).read_text())


def supervision_rows(sup, key, setting, cosines, targets):
    rows = []
    names = dict(hungarian='hungarian_decoder', readout='readout', complete='complete', part='part')
    for i, (target, cos) in enumerate(zip(targets, cosines)):
        errors = {m: sup[key]['point'][names[m]]['per_run'][i]['parts'] for m in METHODS}
        rows.append(dict(setting=setting, target=target, hungarian_cos=float(np.mean(cos)), errors=errors))
    return rows


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--output', type=Path, default=ART / 'e4_scale' / 'ATOM_STABILITY.json')
    args = parser.parse_args()
    rows = []
    # SHIFT in Pythia-70M: matching relations of the explained features at every site, errors from the E1 panel.
    e1 = json_load(ART / 'e1_matching' / 'E1_SHIFT_R59PANEL.json')['per_target']
    keys = dict(hungarian='hungarian_decoder', readout='raw_reconstruction', complete='whole', part='parts')
    for seed in (2, 3, 4, 5):
        rel = json_load(BULK / f'matching_relations_seed{seed}' / 'MATCHING_RELATIONS.json')['sites']
        cos = [c for site in rel.values() for c in site['decoder_cosine_of_match']]
        run = e1[f'NAACL_E1_shift_r59panel_seed{seed}_v2_20261010']['methods']
        # Calibrated one-to-one matching: the decoder matches with a natural-text gain per source feature.
        rows.append(dict(setting='SHIFT, Pythia-70M', target=f'seed {seed}', hungarian_cos=float(np.mean(cos)),
                         errors={m: run[keys[m]]['parts']['nrmse'] for m in METHODS},
                         calibrated=run['geometry_gain']['parts']['nrmse']))
    sup = json_load(ART / 'e2_supervision' / 'SUPERVISION.json')['settings']
    inf = json_load(BULK / 'infinitive_matching' / 'INFINITIVE_MATCHING.json')['targets']
    inf_rows = supervision_rows(sup, 'infinitive', 'Infinitive, Pythia-70M', [inf[str(t)]['decoder_cosine'] for t in (1, 3, 4, 5)],
                                [f'seed {t}' for t in (1, 3, 4, 5)])
    setting = next(s for s in SETTINGS if s['key'] == 'infinitive')
    for row, run in zip(inf_rows, setting['runs']):
        values, fam, groups = load(run, 'infinitive')
        err, eff = per_item(values, fam, 'geometry_gain')
        row['calibrated'] = float(nrmse(err['parts'], eff['parts'], groups, np.ones(len(groups))))
    rows += inf_rows
    public = json_load(ART / 'e4_scale' / 'E4A_RESULTS.json')['per_run']
    names = [f'NAACL_E4A_public_t{t}_v1_20261010' for t in range(101, 117)]
    rows += supervision_rows(sup, 'infinitive_public', 'Infinitive, SAEBench', [public[n]['target']['hungarian_decoder_cosine'] for n in names],
                             [f"{public[n]['target']['dict_size']} k{public[n]['target']['k']} {public[n]['target']['trainer']}" for n in names])
    gpt2 = json_load(BULK / 'grammar_matching_maps_s1' / 'MATCHING_MAPS.json')['targets']
    # Grammar settings use the fixed correspondence, which runs with the target SAE alone.
    rows += supervision_rows(sup, 'gpt2_fixed', 'Grammar, GPT-2 Medium', [[gpt2[str(t)]['hungarian_decoder']['mean_cosine']] for t in (2, 3, 4, 5)],
                             [f'seed {t}' for t in (2, 3, 4, 5)])
    qwen = [json_load(MIRROR / f'qwen17_matching_maps_l14_v2_s{s}' / 'MATCHING_MAPS.json')['targets'][str(t)]['hungarian_decoder']['mean_cosine']
            for s, t in ((1, 2), (2, 1))]
    rows += supervision_rows(sup, 'qwen_fixed', 'Grammar, Qwen3-1.7B', [[c] for c in qwen], ['L0 50 to 100', 'L0 100 to 50'])
    # Gemma Scope targets: cosine from the read-back run, errors from the final and grid analyses.
    readback = json_load(MIRROR / 'NAACL_E10_GEMMA2B_readback_v1_20261010' / 'READBACK.json')
    gemma = {}
    for name in ['E4C_FINAL.json', 'E10_GRID.json', 'E12_MATCHED.json']:
        path = ART / 'e4_scale' / name
        if path.exists():
            gemma.update(json_load(path)['runs'])
    run_of = dict(w65k_sfc='final_w65k', w16k_lowest='final_l0low16k')
    gkeys = dict(hungarian='hungarian_decoder', readout='readout', complete='complete_sup_hungarian', part='part_sup_hungarian')
    for target, rb in readback.items():
        match = run_of.get(target, f'grid_{target}')
        run = next((v for k, v in gemma.items() if match in k), None)
        if run is None:
            continue
        # Matched gains: the Hungarian matches with deletion fractions trained by part supervision, from the same run
        # for the final targets and from the E12 run for the grid targets.
        matched = run if 'matched_part_sup_hungarian' in run['point'] else \
            next((v for k, v in gemma.items() if f'matched_{target}' in k), None)
        rows.append(dict(setting='SHIFT, Gemma-2-2B', target=target, hungarian_cos=rb['hungarian_cos_mean'],
                         errors={m: run['point'][gkeys[m]]['parts'] for m in METHODS},
                         calibrated=None if matched is None else matched['point']['matched_part_sup_hungarian']['parts']))
    cos = np.array([r['hungarian_cos'] for r in rows])
    stats = {}
    for m in METHODS:
        err = np.array([r['errors'][m] for r in rows])
        stats[m] = dict(spearman_with_cos=float(spearmanr(cos, np.log(err)).statistic))
    ratio = np.array([r['errors']['hungarian'] / r['errors']['part'] for r in rows])
    setting = np.array([r['setting'] for r in rows])
    # Within-setting association: centre both quantities on their setting means before correlating.
    cc, lr = cos.copy(), np.log(ratio)
    for s in set(setting):
        cc[setting == s] -= cc[setting == s].mean()
        lr[setting == s] -= lr[setting == s].mean()
    stats['hungarian_over_part'] = dict(spearman_with_cos=float(spearmanr(cos, np.log(ratio)).statistic),
                                        within_setting_spearman=float(spearmanr(cc, lr).statistic),
                                        within_setting_pearson=float(np.corrcoef(cc, lr)[0, 1]),
                                        part_lower=int((ratio > 1).sum()), targets=len(rows),
                                        geometric_mean_ratio=float(np.exp(np.log(ratio).mean())))
    stats['per_setting'] = {s: dict(targets=int((setting == s).sum()), cos_mean=float(cos[setting == s].mean()),
                                    geometric_mean_ratio=float(np.exp(np.log(ratio[setting == s]).mean())))
                            for s in sorted(set(setting))}
    public_rows = [r for r in rows if r['setting'] == 'Infinitive, SAEBench']
    stats['saebench_spearman'] = float(spearmanr([r['hungarian_cos'] for r in public_rows],
                                                 [np.log(r['errors']['hungarian'] / r['errors']['part']) for r in public_rows]).statistic)
    # Re-applying the source edit on SAEBench, against the retention of the source features' activation.
    retained = np.array([np.mean(public[n]['target']['source_activation_retained']) for n in names])
    widths = np.array([public[n]['target']['dict_size'] for n in names])
    readout_ratio = np.array([r['errors']['readout'] / r['errors']['part'] for r in rows if r['setting'] == 'Infinitive, SAEBench'])
    readout = dict(retained=retained.tolist(), widths=widths.tolist(), ratio=readout_ratio.tolist(),
                   spearman_within_width={int(w): float(spearmanr(retained[widths == w], np.log(readout_ratio[widths == w])).statistic)
                                          for w in (4096, 16384)},
                   part_lower=int((readout_ratio > 1).sum()))
    lp = np.log([r['errors']['part'] for r in rows])
    for s in set(setting):
        lp[setting == s] -= lp[setting == s].mean()
    stats['part_within_setting_spearman'] = float(spearmanr(cc, lp).statistic)
    calibrated = {}
    for s in sorted(set(setting)):
        values = [r['calibrated'] / r['errors']['part'] for r in rows if r['setting'] == s and r.get('calibrated') is not None]
        if values:
            calibrated[s] = dict(targets=len(values), min_ratio=float(min(values)), max_ratio=float(max(values)))
    stats['calibrated_over_part'] = calibrated
    for r in rows:
        print(f"{r['setting']:<24} {r['target']:<22} cos {r['hungarian_cos']:.3f}  " +
              '  '.join(f"{m} {r['errors'][m]:.3f}" for m in METHODS) +
              ('' if r.get('calibrated') is None else f"  calibrated {r['calibrated']:.3f}"))
    print(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in readout.items() if k in ('spearman_within_width', 'part_lower')}))
    args.output.write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION,
                                           rows=rows, stats=stats, readout=readout), indent=1))


if __name__ == '__main__':
    main()
