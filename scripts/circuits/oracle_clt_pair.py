import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from circuit_tracer import ReplacementModel
from common import Dictionary, add_hooks, load_target
from oracle_sites import box_least_squares

DESCRIPTION = ('CIRCUITS_ACROSS_DICTIONARIES P1c: transfer between two cross-layer transcoders of the same model '
               '(same architecture, separately trained). Units are the single CLT-A features of a P1r run (the most '
               'influential nodes of the CLT-A attribution graph on each published prompt). Each unit deletion writes '
               'into its layer and every later one at its position. In CLT-B it is reconstructed (box-constrained '
               'least squares, each feature deleted by at most its activation) either as whole CLT-B features, all of '
               'whose writes are matched at once across layers, or slice by slice, one layer at a time from CLT-B '
               'features of that or earlier layers. Effects on the final-position next-token distribution are '
               'compared with the CLT-A deletion; random directions of equal norm are the capacity control; the best '
               'activation correlation of each unit feature with a CLT-B feature of its layer on generic text '
               'measures how alike the two dictionaries are.')
UNIT = re.compile(r'^L(\d+)F(\d+)P(\d+)$')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--units-from', type=Path, required=True, help='REVERSE_ROWS.json of a P1r run on CLT-A')
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--text', type=Path, required=True)
    parser.add_argument('--docs', type=int, default=128)
    parser.add_argument('--context', type=int, default=64)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'config.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    t0 = time.time()

    def log(**kw):
        row = dict(elapsed=round(time.time() - t0, 1), **kw)
        print(json.dumps(row), flush=True)
        with (args.out / 'progress.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
    dev = torch.device('cuda')
    model = ReplacementModel.from_pretrained('google/gemma-2-2b', args.source, dtype=torch.bfloat16, backend='transformerlens')
    model.requires_grad_(False)
    A = Dictionary(model.transcoders)
    B = Dictionary(load_target(args.target, dev, torch.bfloat16))
    A.t.requires_grad_(False)
    B.t.requires_grad_(False)
    L, d = model.cfg.n_layers, model.cfg.d_model
    units = {}
    for r in json.loads(args.units_from.read_text())['rows']:
        m = UNIT.match(r['unit'])
        if m:
            units.setdefault((r['graph'], r['prompt']), []).append(tuple(int(v) for v in m.groups()))
    log(stage='loaded', prompts=len(units), units=sum(len(v) for v in units.values()))
    names = [f'blocks.{l}.{A.input_hook}' for l in range(L)] + [f'blocks.{l}.{B.input_hook}' for l in range(L)]
    names = list(dict.fromkeys(names))
    gen = torch.Generator().manual_seed(0)

    # Best activation correlation of each unit feature with a CLT-B feature of the same layer, on generic text.
    texts = json.loads(args.text.read_text())['train'][:args.docs * 2]
    seqs = [model.tokenizer(t, return_tensors='pt').input_ids[0, :args.context] for t in texts]
    seqs = torch.stack([s for s in seqs if len(s) == args.context][:args.docs])
    need = {}
    for us in units.values():
        for l, f, _ in us:
            need.setdefault(l, set()).add(f)
    need = {l: sorted(fs) for l, fs in need.items()}
    st = {l: dict(n=0., sa=0., sb=0., saa=0., sbb=0., sab=0.) for l in need}
    with torch.no_grad():
        for k in range(0, len(seqs), 8):
            _, cache = model.run_with_cache(seqs[k:k + 8].to(dev), names_filter=lambda n: n in names)
            for l, fs in need.items():
                za = A.encode(cache[f'blocks.{l}.{A.input_hook}'][:, 1:], l, fs).float().flatten(0, 1)
                zb = B.encode(cache[f'blocks.{l}.{B.input_hook}'][:, 1:], l).float().flatten(0, 1)
                s = st[l]
                s['n'] += za.shape[0]
                for key, v in [('sa', za.sum(0)), ('sb', zb.sum(0)), ('saa', (za ** 2).sum(0)), ('sbb', (zb ** 2).sum(0)),
                               ('sab', za.T @ zb)]:
                    s[key] = s[key] + v
    best = {}
    for l, fs in need.items():
        s = st[l]
        ma, mb = s['sa'] / s['n'], s['sb'] / s['n']
        corr = (s['sab'] / s['n'] - ma[:, None] * mb[None]) / ((s['saa'] / s['n'] - ma ** 2)[:, None].clamp_min(1e-12) *
                                                              (s['sbb'] / s['n'] - mb ** 2)[None].clamp_min(1e-12)).sqrt()
        for i, f in enumerate(fs):
            best[(l, f)] = float(corr[i].max())
    log(stage='similarity', median=float(np.median(list(best.values()))))

    def run(ids, deltas):
        with torch.no_grad():
            logits = model.run_with_hooks(ids, fwd_hooks=add_hooks(deltas))
        return torch.log_softmax(logits[0, -1].float(), -1)

    rows = []
    for (key, prompt), us in units.items():
        ids = model.tokenizer(prompt, return_tensors='pt').input_ids.to(dev)
        n = ids.shape[1]
        with torch.no_grad():
            _, cache = model.run_with_cache(ids, names_filter=lambda n_: n_ in names)
            zb = {l: B.encode(cache[f'blocks.{l}.{B.input_hook}'][0], l).float() for l in range(L)}
        base = run(ids, {})
        blocks_at = {}
        for l, f, p in us:
            if p not in blocks_at:
                # Active CLT-B features at position p with all their writes, stacked over layers 0..L-1.
                cols, zz, decs = [], [], []
                for lb in range(L):
                    act = (zb[lb][p] > 0).nonzero().flatten()
                    if len(act):
                        dec = B.decoders(lb, act.tolist())
                        full = torch.zeros(len(act), L, d, device=dev)
                        full[:, lb:] = dec
                        cols.append(full)
                        zz.append(zb[lb][p, act])
                        decs.append((lb, dec))
                blocks_at[p] = (torch.cat(cols), torch.cat(zz), decs)
            Dfull, z, decs = blocks_at[p]
            za = A.encode(cache[f'blocks.{l}.{A.input_hook}'][0, p], l, [f]).float()[0]
            dec_a = A.decoders(l, [f])[0]
            target = torch.zeros(L, d, device=dev)
            target[l:] = za * dec_a
            src = {l + k: torch.zeros(n, d, device=dev) for k in range(dec_a.shape[0])}
            for k in range(dec_a.shape[0]):
                src[l + k][p] -= target[l + k]
            # Whole CLT-B features: every write matched at once.
            c = box_least_squares(Dfull.flatten(1), target.flatten(), z)
            recon = (c[:, None, None] * Dfull).sum(0)
            res_whole = float((target - recon).norm() / target.norm())
            rnd = torch.randn(target.shape, generator=gen).to(dev)
            rnd = rnd * target.norm() / rnd.norm()
            res_whole_rnd = float((rnd.flatten() - box_least_squares(Dfull.flatten(1), rnd.flatten(), z) @ Dfull.flatten(1)).norm() / rnd.norm())
            whole = {k: torch.zeros(n, d, device=dev) for k in range(L)}
            for k in range(L):
                whole[k][p] -= recon[k]
            # Slices: each written layer reconstructed from CLT-B writes into that layer only.
            sl, res_sl, norms = {}, [], []
            for k in range(l, L):
                Dk = Dfull[:, k]
                keep = Dk.norm(dim=1) > 0
                ck = box_least_squares(Dk[keep], target[k], z[keep])
                rk = ck @ Dk[keep]
                res_sl.append(float((target[k] - rk).norm() / target[k].norm().clamp_min(1e-9)))
                norms.append(float(target[k].norm()))
                sl.setdefault(k, torch.zeros(n, d, device=dev))[p] -= rk
            ea = run(ids, src) - base
            ew, es = run(ids, whole) - base, run(ids, sl) - base
            w = base.exp()
            energy = float((w * ea ** 2).sum())
            nr = lambda e: float((w * (e - ea) ** 2).sum() / max(energy, 1e-12)) ** .5
            top = lambda e: model.tokenizer.decode(int((base + e).argmax()))
            rows.append(dict(graph=key, unit=f'L{l}F{f}P{p}', prompt=prompt, best_corr=best[(l, f)], source_energy=energy,
                             residual_whole=res_whole, residual_whole_random=res_whole_rnd,
                             residual_slices=float(np.average(res_sl, weights=norms)), nrmse_whole=nr(ew),
                             nrmse_slices=nr(es), top_base=top(torch.zeros_like(base)), top_source=top(ea),
                             top_whole=top(ew), top_slices=top(es)))
        log(stage='graph', graph=key, units=len(us))
    (args.out / 'CLT_PAIR_ROWS.json').write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(),
                                                                description=DESCRIPTION, rows=rows), indent=1))
    log(stage='done', rows=len(rows))


if __name__ == '__main__':
    main()
