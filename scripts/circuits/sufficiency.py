import argparse
import json
from pathlib import Path

import numpy as np

from summarize_depth_referee import later

DESCRIPTION = ('Sufficiency and joint-subspace removal of the unit test, by kind of CLT unit (always-active early '
               'features, other early units, units read after layer 1). For units that the token replacement switches '
               'off and whose claimed later writes it removes (at least .05), the sufficiency of the read direction is '
               'the share of those writes that restoring the unit\'s read projection on the replaced token brings back, '
               '1 - R_suff / R_tok, and likewise for the span of all active features\' read directions; the joint '
               'removal is the share of the later writes removed when every CLT feature active at the site is '
               'switched off on the clean prompt. Medians with quartiles and counts.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', required=True, help='name=DEPTH_REFEREE_ROWS.json')
    parser.add_argument('--always', nargs='*', default=[], help='name=FEATURE,FEATURE')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    always = {k: set(v.split(',')) for k, v in (s.split('=') for s in args.always)}
    q = lambda v: [float(np.median(v)), float(np.quantile(v, .25)), float(np.quantile(v, .75)), len(v)] if len(v) else None
    out = {}
    for spec in args.runs:
        name, path = spec.split('=')
        rows = [r for r in json.loads(Path(path).read_text())['rows']
                if len(r['claim_energy']) > 1 and sum(r['claim_energy'][1:]) > 0 and 'removed_u_suff' in r]
        kinds = {'always_active': lambda r: r['layer'] <= 1 and r['unit'].split('P')[0] in always.get(name, set()),
                 'early_other': lambda r: r['layer'] <= 1 and r['unit'].split('P')[0] not in always.get(name, set()),
                 'later': lambda r: r['layer'] >= 2}
        d = {}
        for k, keep in kinds.items():
            sel = [r for r in rows if keep(r)]
            tok = np.array([later(r, 'removed_u_tok') for r in sel])
            ok = [r for r, t in zip(sel, tok) if r['tok_off'] and t >= .05]
            suff = [1 - later(r, 'removed_u_suff') / later(r, 'removed_u_tok') for r in ok]
            suff_joint = [1 - later(r, 'removed_u_suff_joint') / later(r, 'removed_u_tok') for r in ok]
            d[k] = dict(units=len(sel), tested=len(ok), removed_u=q([later(r, 'removed_u') for r in sel]),
                        removed_tok=q(tok), removed_joint=q([later(r, 'removed_u_joint') for r in sel]),
                        sufficiency=q(suff), sufficiency_joint=q(suff_joint),
                        active_at_site=float(np.median([r['active_at_site'] for r in sel])) if sel else None)
        out[name] = d
        print(name, json.dumps({k: {m: (v[m][:2] if isinstance(v[m], list) else v[m]) for m in v} for k, v in d.items()}))
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
