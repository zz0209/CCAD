import argparse
import json
from pathlib import Path

DESCRIPTION = ('Print the Markdown tables of the P2 step-2 README from a STEP2_DECISION_phase<N>.json written by '
               'step2_analysis.py: swap success per checkpoint and set, J-lens contrasts with the H1 rule, H2, H5, the '
               'OLMo-1B trajectory and the tuned-lens convergence record. Numbers are read, never recomputed.')
ORDER = ['olmo1b_step10000', 'olmo1b_step20000', 'olmo1b_step50000', 'olmo1b_step100000', 'olmo1b_step200000',
         'olmo1b_step430000', 'olmo1b_step900000', 'olmo1b_step1907359', 'olmo1b_main', 'pythia410m', 'pythia14b',
         'pythia28b']
METHODS = ['jlens', 'logit', 'logit_nm', 'tmatch', 'tmatch_nm', 'tuned', 'tuned_nm']
X = ['logit', 'logit_nm', 'tmatch', 'tmatch_nm', 'tuned', 'tuned_nm']


def pct(x):
    return f'{100 * x:.1f}'


def rate(r):
    return f"{r['k']} = {pct(r['rate'])} [{pct(r['ci95'][0])}, {pct(r['ci95'][1])}]" if r['n'] else 'n = 0'


def con(c):
    return f"{100 * c['delta']:+.1f} [{100 * c['ci95'][0]:+.1f}, {100 * c['ci95'][1]:+.1f}]"


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--decision', type=Path, required=True)
    args = parser.parse_args()
    d = json.loads(args.decision.read_text())
    lens = d['lens']
    labels = [lb for lb in ORDER if lb in lens]
    for s in ('v1', 'v2'):
        print(f'\n### Swap success, {s} (count = % [95% interval])\n')
        print('| Checkpoint | n | ' + ' | '.join(METHODS) + ' |')
        print('|---' * (len(METHODS) + 2) + '|')
        for lb in labels:
            b = lens[lb][s]
            print(f"| {lb} | {b['n']} | " + ' | '.join(rate(b['rates'][m]) for m in METHODS) + ' |')
        print(f'\n### J-lens minus each lens, {s} (points [95% interval]); H1 rule\n')
        print('| Checkpoint | ' + ' | '.join(f'J − {x}' for x in X) + ' | H1 rule |')
        print('|---' * (len(X) + 2) + '|')
        for lb in labels:
            b = lens[lb][s]
            print(f'| {lb} | ' + ' | '.join(con(b['contrasts'][f'jlens-{x}']) for x in X) +
                  f" | {'met' if b['h1_rule'] else 'not met'} |")
        print(f'\n### McNemar p (J-only/other-only), {s}\n')
        print('| Checkpoint | ' + ' | '.join(f'J − {x}' for x in X) + ' |')
        print('|---' * (len(X) + 1) + '|')
        for lb in labels:
            b = lens[lb][s]
            print(f'| {lb} | ' + ' | '.join(
                f"{b['contrasts'][f'jlens-{x}']['mcnemar_p']:.2g} ({b['contrasts'][f'jlens-{x}']['a_only']}/"
                f"{b['contrasts'][f'jlens-{x}']['b_only']})" for x in X) + ' |')
        print(f'\n### Median perturbation norm and bridge readout (correct prompts), {s}\n')
        print('| Checkpoint | ' + ' | '.join(METHODS) + ' | bridge top 5: J / logit / tuned |')
        print('|---' * (len(METHODS) + 2) + '|')
        for lb in labels:
            b = lens[lb][s]
            ro = b['bridge_readout_correct_prompts']
            print(f'| {lb} | ' + ' | '.join(f"{b['median_pert_norm'][m]:.3g}" for m in METHODS) +
                  ' | ' + ' / '.join(pct(ro[x]['top5_frac']) for x in ('jlens', 'logit', 'tuned')) + ' |')
    print('\n### Tuned lenses (TL-full, TL-match)\n')
    print('| Checkpoint | TL-full steps | met 1% rule | conv. KL first → last | held-out KL tuned / logit (band mean) '
          '| TL-match conv. KL first → last | TL-match held-out KL |')
    print('|---' * 7 + '|')
    for lb in labels:
        f, m = lens[lb]['v1']['lenses']['tl_full'], lens[lb]['v1']['lenses']['tl_match']
        fk = ', '.join(f'{a:.3f} → {b:.3f}' for a, b in [(sum(x[0]) / len(x[0]), sum(x[1]) / len(x[1]))
                                                         for x in f['conv_kl_first_last']])
        mk = ', '.join(f'{a:.3f} → {b:.3f}' for a, b in [(sum(x[0]) / len(x[0]), sum(x[1]) / len(x[1]))
                                                         for x in m['conv_kl_first_last']]) if m else ''
        mh = f"{m['heldout_kl_mean']['tuned']:.3f}" if m else ''
        print(f"| {lb} | {f['steps']} | {'yes' if f['all_converged'] else 'no'} | {fk} | "
              f"{f['heldout_kl_mean']['tuned']:.3f} / {f['heldout_kl_mean']['logit']:.3f} | {mk} | {mh} |")
    print('\n### H2\n')
    print(json.dumps(d['H2'], indent=1))
    print('\n### H5\n')
    for lb, x in d['H5'].items():
        for s, v in x.items():
            print(f"{lb} {s}: n {v['n']}, S_J original {rate(v['rates']['orig'])}, refit {rate(v['rates']['refit'])}, "
                  f"Δ {con(v['contrast'])}, McNemar p {v['contrast']['mcnemar_p']:.2g}, stable {v['stable']}")
    print('\n### Trajectory\n')
    for s, t in d['olmo_trajectory'].items():
        print(f"{s}: fixed panel n {t['fixed_panel_n']}; spearman own {t['spearman_own']}; fixed {t['spearman_fixed']}")
        for lb in t['own_prompts']:
            print(f"  {lb}: own {rate(t['own_prompts'][lb])}; fixed panel {rate(t['fixed_panel'][lb])}")
    print('\n### Holm\n')
    print(json.dumps(d['holm'], indent=1))
    print('\n### Verdicts and stop rules\n')
    print(json.dumps({k: d[k]['verdict'] for k in ('H1', 'H2', 'H3', 'H4')}), json.dumps(d['stop_rules']),
          json.dumps(d['stop_rules_training']), json.dumps(d['H1']['rule_by_checkpoint']),
          json.dumps(d['eligibility_counts']))


if __name__ == '__main__':
    main()
