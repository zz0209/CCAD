import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from circuit_tracer import ReplacementModel
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
from common import filter_items, gemma_items

DESCRIPTION = ('CIRCUITS_ACROSS_DICTIONARIES pilot P0: every supernode of the published Gemma-2-2B explanations is '
               'deleted at its published positions through the dictionary that defined it (source A) and through '
               'feature-by-feature matches in another dictionary of the same model (target B), matched by activation '
               'correlation on generic text and the published prompts. The readout is the change of the next-token '
               'log-probabilities at the final position of the prompt, with the full model running after the edit.')


class Dictionary:
    def __init__(self, transcoders, input_hook):
        self.t, self.input_hook = transcoders, input_hook
        self.dtype = next(transcoders.parameters()).dtype

    def encode(self, x, layer, feats=None):
        z = self.t.encode_layer(x.to(self.dtype), layer)
        return z if feats is None else z[..., feats]

    def edits(self, layer, feats, coeffs, n_layers):
        # Per-layer write of feature contributions: coeffs (..., n) times decoders, spread over output layers.
        dec = self.t._get_decoder_vectors(layer, torch.as_tensor(feats, device=coeffs.device)).float()
        if dec.ndim == 2:
            out = {layer: coeffs @ dec}
        else:
            out = {layer + k: coeffs @ dec[:, k] for k in range(dec.shape[1])}
        return out


def run(model, tokens, deltas):
    # deltas: {layer: {pos: vector}} added to blocks.layer.hook_mlp_out; returns final-position log-probabilities.
    hooks = []
    for layer, by_pos in deltas.items():
        def hook(act, hook, by_pos=by_pos):
            act = act.clone()
            for pos, v in by_pos.items():
                act[0, pos] += v.to(act.dtype)
            return act
        hooks.append((f'blocks.{layer}.hook_mlp_out', hook))
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    return torch.log_softmax(logits[0, -1].float(), -1)


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--target', required=True, help='Hugging Face reference of dictionary B')
    parser.add_argument('--text', type=Path, required=True, help='generic text slice (json with a train list)')
    parser.add_argument('--docs', type=int, default=256)
    parser.add_argument('--context', type=int, default=128)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    log = lambda **kw: print(json.dumps(dict(elapsed=round(time.time() - t0, 1), **kw)), flush=True)
    dev = torch.device('cuda')
    model = ReplacementModel.from_pretrained('google/gemma-2-2b', 'gemma', dtype=torch.bfloat16, backend='transformerlens')
    A = Dictionary(model.transcoders, model.transcoders.feature_input_hook)
    clt, _ = load_transcoder_from_hub(args.target, device=dev, dtype=torch.bfloat16, lazy_decoder=True)
    B = Dictionary(clt, clt.feature_input_hook)
    n_layers = model.cfg.n_layers
    log(stage='loaded', a_hook=A.input_hook, b_hook=B.input_hook, b_width=clt.d_transcoder)

    items, dropped = filter_items(gemma_items(args.registry), n_layers, model.transcoders.d_transcoder)
    log(stage='graphs', kept=len(items), dropped=dropped)

    # Source features per layer, for the correlation statistics.
    need = {}
    for _, _, sn in items:
        for feats in sn.values():
            for layer, pos, f in feats:
                need.setdefault(layer, set()).add(f)
    need = {l: sorted(fs) for l, fs in need.items()}

    # Activation correlation between each needed A feature and every B feature of the same layer, over generic text
    # and the published prompts.
    texts = json.loads(args.text.read_text())['train'][:args.docs]
    seqs = [model.tokenizer(t, return_tensors='pt').input_ids[0, :args.context] for t in texts]
    seqs = [s for s in seqs if len(s) == args.context]
    seqs += [model.tokenizer(p, return_tensors='pt').input_ids[0] for _, p, _ in items]
    stats = {l: dict(n=0., sa=0., sb=0., saa=0., sbb=0., sab=0.) for l in need}
    for k, ids in enumerate(seqs):
        names = [f'blocks.{l}.{A.input_hook}' for l in need] + [f'blocks.{l}.{B.input_hook}' for l in need]
        with torch.no_grad():
            _, cache = model.run_with_cache(ids[None].to(dev), names_filter=lambda n: n in names)
            for l, fs in need.items():
                za = A.encode(cache[f'blocks.{l}.{A.input_hook}'][0, 1:], l, fs).float()
                zb = B.encode(cache[f'blocks.{l}.{B.input_hook}'][0, 1:], l).float()
                s = stats[l]
                s['n'] += za.shape[0]
                s['sa'] = s['sa'] + za.sum(0)
                s['sb'] = s['sb'] + zb.sum(0)
                s['saa'] = s['saa'] + (za ** 2).sum(0)
                s['sbb'] = s['sbb'] + (zb ** 2).sum(0)
                s['sab'] = s['sab'] + za.T @ zb
        if (k + 1) % 32 == 0:
            log(stage='correlation', items=k + 1, total=len(seqs))
    match = {}
    for l, fs in need.items():
        s = stats[l]
        cov = s['sab'] / s['n'] - (s['sa'] / s['n'])[:, None] * (s['sb'] / s['n'])[None]
        va = s['saa'] / s['n'] - (s['sa'] / s['n']) ** 2
        vb = s['sbb'] / s['n'] - (s['sb'] / s['n']) ** 2
        corr = cov / (va[:, None].clamp_min(1e-12) * vb[None].clamp_min(1e-12)).sqrt()
        best = corr.max(1)
        for i, f in enumerate(fs):
            match[(l, f)] = (int(best.indices[i]), float(best.values[i]))
    log(stage='matched', features=len(match))

    rows = []
    for key, prompt, sn in items:
        ids = model.tokenizer(prompt, return_tensors='pt').input_ids.to(dev)
        names = [f'blocks.{l}.{A.input_hook}' for l in range(n_layers)] + [f'blocks.{l}.{B.input_hook}' for l in range(n_layers)]
        with torch.no_grad():
            _, cache = model.run_with_cache(ids, names_filter=lambda n: n in names)
        base = run(model, ids, {})
        for name, feats in sn.items():
            da, db, active, matched_active, corrs = {}, {}, 0, 0, []
            for layer, pos, f in feats:
                if pos >= ids.shape[1]:
                    continue
                za = A.encode(cache[f'blocks.{layer}.{A.input_hook}'][0, pos], layer, [f]).float()
                active += int(za.item() > 0)
                for out_l, v in A.edits(layer, [f], -za, n_layers).items():
                    da.setdefault(out_l, {}).setdefault(pos, 0.)
                    da[out_l][pos] = da[out_l][pos] + v
                j, c = match[(layer, f)]
                corrs.append(c)
                zb = B.encode(cache[f'blocks.{layer}.{B.input_hook}'][0, pos], layer, [j]).float()
                matched_active += int(zb.item() > 0)
                for out_l, v in B.edits(layer, [j], -zb, n_layers).items():
                    db.setdefault(out_l, {}).setdefault(pos, 0.)
                    db[out_l][pos] = db[out_l][pos] + v
            ea = run(model, ids, da) - base
            eb = run(model, ids, db) - base
            w = base.exp()
            # Probability-weighted effect energies and the error of the target edit relative to the source effect.
            energy = float((w * ea ** 2).sum())
            err = float((w * (eb - ea) ** 2).sum())
            top = lambda e: model.tokenizer.decode(int((base + e).argmax()))
            rows.append(dict(graph=key, supernode=name, features=len(feats), source_active=active,
                             matched_active=matched_active, mean_match_corr=float(np.mean(corrs)) if corrs else None,
                             source_energy=energy, nrmse=(err / energy) ** .5 if energy > 0 else None,
                             answer_drop_source=float(-ea[int(base.argmax())]), answer_drop_target=float(-eb[int(base.argmax())]),
                             top_base=top(torch.zeros_like(base)), top_source=top(ea), top_target=top(eb)))
            log(stage='supernode', **{k: rows[-1][k] for k in ['graph', 'supernode', 'source_active', 'matched_active', 'nrmse']})
    (args.out / 'P0_ROWS.json').write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(),
                                                          description=DESCRIPTION, target=args.target, rows=rows), indent=1))
    log(stage='done', rows=len(rows))


if __name__ == '__main__':
    main()
