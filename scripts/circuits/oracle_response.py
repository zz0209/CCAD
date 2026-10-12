import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from circuit_tracer import ReplacementModel
from common import Dictionary, add_hooks, choose_prompt, filter_items, gemma_items, load_target
from oracle_sites import box_least_squares

DESCRIPTION = ('CIRCUITS_ACROSS_DICTIONARIES response-aware test: whole target features are fitted to the complete change '
               'that a source supernode deletion causes in every MLP output at each edited position (the deleted write '
               'plus the responses of later MLPs), and applied with the MLP outputs at the edited positions held at '
               'their clean values minus the fitted writes, other positions free. A cross-layer transcoder thus gets '
               'to stand its later-layer writes in for the downstream responses, as its replacement semantics '
               'intend. The true change applied in the same way reproduces the source deletion (sanity check).')


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
    d = model.cfg.d_model
    cross = B.decoders(0, [0]).shape[1] > 1
    items, dropped = filter_items(gemma_items(args.registry, args.registry_model), L, A.t.d_transcoder)
    log(stage='loaded', target=args.target, cross_layer=cross, graphs=len(items), dropped=dropped)
    names = [f'blocks.{l}.{A.input_hook}' for l in range(L)] + [f'blocks.{l}.{B.input_hook}' for l in range(L)]
    mlp_names = [f'blocks.{l}.hook_mlp_out' for l in range(L)]

    def run(ids, deltas, store=None):
        # Adds deltas to the MLP outputs; with store, also records every MLP output after the edit.
        hooks = add_hooks(deltas)
        if store is not None:
            hooks = hooks + [(nm, (lambda act, hook: store.__setitem__(hook.name, act[0].float().clone()))) for nm in mlp_names]
        with torch.no_grad():
            logits = model.run_with_hooks(ids, fwd_hooks=hooks)
        return torch.log_softmax(logits[0, -1].float(), -1)

    def run_fixed(ids, fixed):
        # fixed: {layer: {position: value}}; the MLP output at those positions is replaced, other positions run free.
        def make(k):
            def hook(act, hook):
                for p, v in fixed.get(k, {}).items():
                    act[0, p] = v.to(act.dtype)
                return act
            return hook
        hooks = [(f'blocks.{k}.hook_mlp_out', make(k)) for k in range(L)]
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
        with torch.no_grad():
            _, cache = model.run_with_cache(ids, names_filter=lambda n_: n_ in names or n_ in mlp_names)
            zb = {l: B.encode(cache[f'blocks.{l}.{B.input_hook}'][0], l).float() for l in range(L)}
        clean = torch.stack([cache[nm][0].float() for nm in mlp_names])  # (L, n, d)
        base = run(ids, {})
        for sname, fs in sn.items():
            src = {}
            for l, p, f in fs:
                if p >= n:
                    continue
                za = A.encode(cache[f'blocks.{l}.{A.input_hook}'][0, p], l, [f]).float()
                src.setdefault(l, torch.zeros(n, d, device=dev))[p] -= za[0] * A.decoders(l, [f])[0, 0]
            if not src:
                continue
            store = {}
            ea = run(ids, src, store) - base
            after = torch.stack([store[nm] for nm in mlp_names])  # (L, n, d)
            positions = sorted({p for _, p, _ in fs if p < n})
            fixed_true, fixed_fit = {}, {}
            resid, resid_random, resp_share = [], [], []
            for p in positions:
                T = after[:, p] - clean[:, p]  # (L, d): complete change of every MLP output at p
                for k in range(L):
                    fixed_true.setdefault(k, {})[p] = after[k, p]
                edited = sorted({l for l, pp, _ in fs if pp == p})
                total = float((T ** 2).sum())
                resp_share.append(float((T[[k for k in range(L) if k not in edited]] ** 2).sum()) / max(total, 1e-12))
                # Every active target feature at p, with its whole write over all layers (zero before its own layer).
                blocks = []
                for lb in range(L):
                    act = (zb[lb][p] > 0).nonzero().flatten()
                    if len(act):
                        blocks.append((lb, B.decoders(lb, act.tolist()), zb[lb][p, act]))
                full = torch.zeros(sum(len(zz) for _, _, zz in blocks), L, d, device=dev)
                start = 0
                for lb, dec, zz in blocks:
                    full[start:start + len(zz), lb:lb + dec.shape[1]] = dec
                    start += len(zz)
                z = torch.cat([zz for _, _, zz in blocks])
                flat = full.flatten(1)
                # A deletion subtracts c_j W_j, so the fitted writes match -T.
                c = box_least_squares(flat, -T.flatten(), z)
                recon = (c @ flat).view(L, d)
                resid.append(float((recon + T).norm() / T.norm().clamp_min(1e-9)))
                rnd = torch.randn(T.shape, generator=gen, device='cpu').to(dev)
                rnd = rnd * T.norm() / rnd.norm()
                resid_random.append(float((rnd.flatten() - box_least_squares(flat, rnd.flatten(), z) @ flat).norm()
                                          / rnd.norm()))
                for k in range(L):
                    fixed_fit.setdefault(k, {})[p] = clean[k, p] - recon[k]
            et = run_fixed(ids, fixed_true) - base
            ef = run_fixed(ids, fixed_fit) - base
            w = base.exp()
            energy = float((w * ea ** 2).sum())
            nr = lambda e: float((w * (e - ea) ** 2).sum() / max(energy, 1e-12)) ** .5
            top = lambda e: model.tokenizer.decode(int((base + e).argmax()))
            rows.append(dict(graph=key, supernode=sname, prompt=prompt, prompt_activity=activity, positions=len(positions),
                             source_energy=energy, response_share=float(np.mean(resp_share)),
                             response_residual=float(np.mean(resid)), response_residual_random=float(np.mean(resid_random)),
                             nrmse_response=nr(ef), nrmse_sanity=nr(et), top_base=top(torch.zeros_like(base)),
                             top_source=top(ea), top_response=top(ef), top_sanity=top(et)))
        log(stage='graph', graph=key)
    (args.out / 'RESPONSE_ROWS.json').write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(),
                                                                 description=DESCRIPTION, target=args.target, rows=rows), indent=1))
    log(stage='done', rows=len(rows))


if __name__ == '__main__':
    main()
