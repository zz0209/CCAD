import argparse
import json
from pathlib import Path

import numpy as np

from summarize_depth_referee import later

DESCRIPTION = ('Counterfactual-token removal by kind of CLT unit: always-active early features (listed), early stand-ins '
               '(removing the PLT units of the prompt takes out more of the later writes than random PLT features, by '
               'over .3), other early units, and units read after layer 1. For each kind, the median fraction of the '
               'claimed later writes removed by replacing the token at the unit\'s position (u_tok) and by removing '
               'the unit\'s input (u), with quartiles, and the share of units that the replacement switches off.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', required=True, help='name=DEPTH_REFEREE_ROWS.json')
    parser.add_argument('--always', nargs='*', default=[], help='name=FEATURE,FEATURE (e.g. gemma=L0F7081,L1F3148)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    always = {k: set(v.split(',')) for k, v in (s.split('=') for s in args.always)}
    out = {}
    for spec in args.runs:
        name, path = spec.split('=')
        rows = [r for r in json.loads(Path(path).read_text())['rows']
                if len(r['claim_energy']) > 1 and sum(r['claim_energy'][1:]) > 0 and 'removed_u_tok' in r]

        def kind(r):
            if r['layer'] >= 2:
                return 'later'
            if r['unit'].split('P')[0] in always.get(name, set()):
                return 'always_active'
            if 'removed_a' in r and later(r, 'removed_a') - later(r, 'removed_ra') > .3:
                return 'stand_in'
            return 'other_early'

        d = {}
        groups = {k: [k] for k in ('always_active', 'stand_in', 'other_early', 'later')}
        groups['early_not_always'] = ['stand_in', 'other_early']
        for k, kinds in groups.items():
            sel = [r for r in rows if kind(r) in kinds]
            if not sel:
                continue
            tok = np.array([later(r, 'removed_u_tok') for r in sel])
            u = np.array([later(r, 'removed_u') for r in sel])
            d[k] = dict(units=len(sel), tok=[float(np.median(tok)), float(np.quantile(tok, .25)), float(np.quantile(tok, .75))],
                        u=float(np.median(u)), tok_off=float(np.mean([r['tok_off'] for r in sel])))
        out[name] = d
        print(name, json.dumps(d))
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
