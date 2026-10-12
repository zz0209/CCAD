import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

import toy_referee as T

DESCRIPTION = ('Why restoring a read projection does not bring back the two-step circuit of the toy model. For each seed '
               'the CLT is retrained exactly as in toy_referee.py, and for every CLT feature labeled with the two-step '
               'circuit (A) the share of its squared encoder norm that lies in the input dimensions 0 to 3 is reported, '
               'with, on each row that enters the toy summary, the change of the restored input in dimensions 0 to 3 as '
               'a share of the counterfactual change there, and whether the conjunction comes back.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--features', type=int, default=16)
    parser.add_argument('--lam', type=float, default=.05)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    x = T.inputs()
    label = 'A two-step'
    others = [d for d in range(T.D_MODEL) if d not in T.INPUTS[label]]
    features, rows = [], []
    for seed in args.seeds:
        clt, _ = T.train(True, args.features, args.lam, args.steps, seed, x)
        with torch.no_grad():
            ins, outs = T.forward(x)
            z = clt.encode(0, ins[0])
            for f in range(args.features):
                later = torch.stack([clt.dec(0, k)[f] for k in range(1, T.L)])
                if later.norm() < 1e-3 or not (z[:, f] > 0).any():
                    continue
                share = {n: float((later[:, d] ** 2).sum()) for n, d in T.CIRCUITS.items()}
                if max(share, key=share.get) != label or share[label] / float((later ** 2).sum()) <= .6:
                    continue
                w = clt.W_enc[0, f]
                features.append(dict(seed=seed, feature=f, read_in_share=float((w[:4] ** 2).sum() / (w ** 2).sum())))
                for i in torch.nonzero(z[:, f] > 0).flatten().tolist():
                    if float((outs[1][i, 8:9] ** 2).sum() + (outs[2][i, 8:9] ** 2).sum()) <= 1e-6:
                        continue
                    same = (x[:, others] == x[i, others]).all(1) & (z[:, f] <= 0)
                    j = torch.nonzero(same).flatten().tolist()[0]
                    diff = ins[0][i] - ins[0][j]
                    d_s = w * (w @ diff) / (w @ w)
                    _, o_s = T.forward(x[j:j + 1], {k: (lambda r, d=d_s: r + d) for k in range(T.L)})
                    rows.append(dict(seed=seed, feature=f, input=i,
                                     restored_in_share=float(d_s[:4] @ diff[:4] / (diff[:4] @ diff[:4])),
                                     conjunction_clean=float(outs[1][i, 8]), conjunction_restored=float(o_s[1][0, 8])))
        print(json.dumps(dict(seed=seed, features=len(features), rows=len(rows))), flush=True)
    med = lambda v: float(torch.tensor(v).median()) if v else None
    summary = dict(features=len(features), rows=len(rows),
                   read_in_share_median=med([f['read_in_share'] for f in features]),
                   features_read_in_share_below_half=sum(f['read_in_share'] < .5 for f in features),
                   restored_in_share_median=med([r['restored_in_share'] for r in rows]),
                   rows_conjunction_restored=sum(r['conjunction_restored'] > .5 * r['conjunction_clean'] for r in rows))
    print(json.dumps(summary, indent=1))
    (args.out / 'TWOSTEP_READS.json').write_text(json.dumps(dict(
        written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, config=vars(args) | dict(out=str(args.out)),
        summary=summary, features=features, rows=rows), indent=1))


if __name__ == '__main__':
    main()
