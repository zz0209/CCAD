from pathlib import Path
import argparse
import json

import numpy as np

DESCRIPTION = ('List, for the Gemma transfer, the source features whose learned membership spreads over several target '
               'features, with decoder cosine, membership weight and the pronoun share of each target feature.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--map', default='part_sup_hungarian_s0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    req = json.loads((args.run / 'requests.json').read_text())
    maps = json.loads((args.run / 'maps.json').read_text())[args.map]
    z = np.load(args.run / 'natural_statistics.npz')
    rows = []
    for site, entry in maps.items():
        idx, w = np.array(entry['idx']), np.array(entry['weights'])
        cand = z[site + '__candidates']
        cos, share = z[site + '__cos'], z[site + '__pron_share_target']
        src_share = z[site + '__pron_share_source']
        feats = [f for f in req['features'] if f['site'] == site]
        for r, f in enumerate(feats):
            active = np.flatnonzero(w[r] > .05)
            members = []
            for j in active[np.argsort(-w[r][active])]:
                k = int(np.flatnonzero(cand == idx[j])[0])
                members.append(dict(target=int(idx[j]), weight=float(w[r][j]), cosine=float(cos[r, k]),
                                    pronoun_share=float(share[k])))
            rows.append(dict(site=site, source=f['index'], part=f['part'], comment=f['comment'],
                             source_pronoun_share=float(src_share[r]), members=members,
                             best_cosine=float(cos[r].max())))
    rows.sort(key=lambda x: (-len(x['members']), x['best_cosine']))
    args.output.write_text(json.dumps(rows, indent=1))
    for x in rows[:8]:
        print(x['site'], x['source'], x['part'], x['comment'], 'src pron', round(x['source_pronoun_share'], 2),
              'best cos', round(x['best_cosine'], 2))
        for m in x['members'][:6]:
            print('   ', m['target'], 'w', round(m['weight'], 2), 'cos', round(m['cosine'], 2), 'pron', round(m['pronoun_share'], 2))


if __name__ == '__main__':
    main()
