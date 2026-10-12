import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from circuit_tracer import ReplacementModel
from common import Dictionary, add_hooks, choose_prompt, filter_items, gemma_items, load_target

DESCRIPTION = ('CIRCUITS_ACROSS_DICTIONARIES P1e: an upper bound on expressing each published supernode deletion in '
               'another dictionary of the same model, on the published prompt itself. At every edit site (layer and '
               'position) the source write of the supernode features is reconstructed from the target features active '
               'at that position, each deleted by at most its activation, through the decoder slices that write into '
               'that layer (target features of that layer or, for a cross-layer transcoder, of any earlier layer); '
               'box-constrained least squares by accelerated projected gradient. The reconstructed deletion is applied to the model either through those '
               'slices only or as whole target features (every slice, as circuit-tracer intervenes), and its effect on '
               'the final-position next-token distribution is compared with the source deletion.')


def box_least_squares(D, t, upper, iters=2000):
    # min_c ||c D - t||^2 subject to 0 <= c <= upper, by accelerated projected gradient (FISTA) with step 1/L,
    # L the largest eigenvalue of D D^T.
    D, t, upper = D.double(), t.double(), upper.double()
    G = D @ D.T
    b = D @ t
    step = 1. / torch.linalg.eigvalsh(G)[-1].clamp_min(1e-12)
    c = torch.zeros_like(upper)
    y, s = c.clone(), 1.
    for _ in range(iters):
        c_next = (y - step * (G @ y - b)).clamp(min=0.).minimum(upper)
        s_next = (1. + (1. + 4. * s * s) ** .5) / 2.
        y = c_next + ((s - 1.) / s_next) * (c_next - c)
        c, s = c_next, s_next
    return c.float()


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--model', default='google/gemma-2-2b')
    parser.add_argument('--registry-model', default='gemma-2-2b')
    parser.add_argument('--source', default='gemma')
    parser.add_argument('--target', required=True)
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
    model = ReplacementModel.from_pretrained(args.model, args.source, dtype=torch.bfloat16, backend='transformerlens')
    model.requires_grad_(False)
    A = Dictionary(model.transcoders)
    B = Dictionary(load_target(args.target, dev, torch.bfloat16))
    A.t.requires_grad_(False)
    B.t.requires_grad_(False)
    L = model.cfg.n_layers
    cross = B.decoders(0, [0]).shape[1] > 1
    items, dropped = filter_items(gemma_items(args.registry, args.registry_model), L, A.t.d_transcoder)
    log(stage='loaded', target=args.target, cross_layer=cross, graphs=len(items), dropped=dropped)
    names = [f'blocks.{l}.{A.input_hook}' for l in range(L)] + [f'blocks.{l}.{B.input_hook}' for l in range(L)]

    def run(ids, deltas, frozen=None):
        # With frozen (clean MLP outputs per layer), every MLP output is held at its clean value plus the edit, so
        # later MLPs do not respond to the edit.
        if frozen is None:
            hooks = add_hooks(deltas)
        else:
            hooks = [(f'blocks.{l}.hook_mlp_out',
                      (lambda act, hook, l=l: (frozen[l] + deltas[l].to(act.dtype)[None]) if l in deltas else frozen[l]))
                     for l in range(L)]
        with torch.no_grad():
            logits = model.run_with_hooks(ids, fwd_hooks=hooks)
        return torch.log_softmax(logits[0, -1].float(), -1)

    rows = []
    gen = torch.Generator().manual_seed(0)
    for key, prompt, sn in items:
        def active_fraction(text, sn=sn):
            ids = model.tokenizer(text, return_tensors='pt').input_ids.to(dev)
            with torch.no_grad():
                _, cache = model.run_with_cache(ids, names_filter=lambda n: n in names)
            flags = [p < ids.shape[1] and float(A.encode(cache[f'blocks.{l}.{A.input_hook}'][0, p], l, [f])) > 0
                     for fs in sn.values() for l, p, f in fs]
            return sum(flags) / len(flags)
        prompt, activity = choose_prompt(prompt, active_fraction)
        ids = model.tokenizer(prompt, return_tensors='pt').input_ids.to(dev)
        n = ids.shape[1]
        mlp_names = [f'blocks.{l}.hook_mlp_out' for l in range(L)]
        with torch.no_grad():
            _, cache = model.run_with_cache(ids, names_filter=lambda n_: n_ in names or n_ in mlp_names)
            zb = {l: B.encode(cache[f'blocks.{l}.{B.input_hook}'][0], l).float() for l in range(L)}
        clean_mlp = {l: cache[f'blocks.{l}.hook_mlp_out'] for l in range(L)}
        base = run(ids, {})
        for sname, fs in sn.items():
            src, sl, wh = {}, {}, {}
            resid, resid_random = [], []
            sites = sorted({(l, p) for l, p, _ in fs if p < n})
            for l, p in sites:
                feats = [f for ll, pp, f in fs if ll == l and pp == p]
                za = A.encode(cache[f'blocks.{l}.{A.input_hook}'][0, p], l, feats).float()
                target = (za[:, None] * A.decoders(l, feats)[:, 0]).sum(0)
                src.setdefault(l, torch.zeros(n, target.shape[0], device=dev))[p] -= target
                # Active target features whose decoders write into layer l at position p.
                blocks = []
                for lb in (range(l + 1) if cross else [l]):
                    act = (zb[lb][p] > 0).nonzero().flatten()
                    if len(act):
                        blocks.append((lb, B.decoders(lb, act.tolist()), zb[lb][p, act]))
                if not blocks:
                    resid.append(1.)
                    continue
                D = torch.cat([dec[:, l - lb] for lb, dec, _ in blocks])
                z = torch.cat([zz for _, _, zz in blocks])
                if float(target.norm()) == 0.:
                    # No source feature of this site is active on the prompt; nothing to reconstruct.
                    continue
                c = box_least_squares(D, target, z)
                recon = c @ D
                resid.append(float((target - recon).norm() / target.norm().clamp_min(1e-9)))
                # Capacity control: a random direction of the same norm, reconstructed from the same active features.
                rnd = torch.randn(target.shape, generator=gen, device='cpu').to(dev)
                rnd = rnd * target.norm() / rnd.norm()
                resid_random.append(float((rnd - box_least_squares(D, rnd, z) @ D).norm() / rnd.norm()))
                sl.setdefault(l, torch.zeros(n, target.shape[0], device=dev))[p] -= recon
                # Whole-feature deletion of the same amounts: every slice of each chosen target feature.
                start = 0
                for lb, dec, zz in blocks:
                    cb = c[start:start + len(zz)]
                    start += len(zz)
                    for k in range(dec.shape[1]):
                        wh.setdefault(lb + k, torch.zeros(n, dec.shape[2], device=dev))[p] -= cb @ dec[:, k]
            if not src:
                continue
            ea, es, ew = run(ids, src) - base, run(ids, sl) - base, run(ids, wh) - base
            # The same edits with every MLP output frozen at its clean value plus the edit.
            ewf, eaf = run(ids, wh, clean_mlp) - base, run(ids, src, clean_mlp) - base
            w = base.exp()
            energy = float((w * ea ** 2).sum())
            nr = lambda e: float((w * (e - ea) ** 2).sum() / max(energy, 1e-12)) ** .5
            top = lambda e: model.tokenizer.decode(int((base + e).argmax()))
            rows.append(dict(graph=key, supernode=sname, prompt=prompt, prompt_activity=activity, sites=len(sites),
                             source_energy=energy, site_residual=float(np.mean(resid)) if resid else None,
                             site_residual_random=float(np.mean(resid_random)) if resid_random else None, nrmse_slice=nr(es),
                             nrmse_whole=nr(ew), nrmse_whole_frozen=nr(ewf), nrmse_source_frozen=nr(eaf),
                             top_base=top(torch.zeros_like(base)), top_source=top(ea), top_slice=top(es),
                             top_whole=top(ew), top_whole_frozen=top(ewf), top_source_frozen=top(eaf)))
        log(stage='graph', graph=key)
    (args.out / 'ORACLE_ROWS.json').write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(),
                                                              description=DESCRIPTION, target=args.target, rows=rows), indent=1))
    log(stage='done', rows=len(rows))


if __name__ == '__main__':
    main()
