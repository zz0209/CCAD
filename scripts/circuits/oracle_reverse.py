import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from circuit_tracer import attribute
from circuit_tracer.graph import prune_graph
from common import Dictionary, add_hooks, load_target, replacement_model
from oracle_sites import box_least_squares

DESCRIPTION = ('CIRCUITS_ACROSS_DICTIONARIES P1r: the reverse of P1e. Explanation units come from the attribution '
               'graph that the cross-layer transcoder (source A) draws on each published prompt: the most influential '
               'feature nodes, deleted alone and in groups at one position. A CLT unit writes into its own layer and '
               'every later one; each of those writes is reconstructed from the per-layer transcoder features (target '
               'B) active at that layer and position, deleting each by at most its activation (box-constrained least '
               'squares), and the reconstructed multi-layer deletion is compared with the CLT deletion on the final-'
               'position next-token distribution, with a random direction of equal norm as capacity control.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--prompts-from', type=Path, required=True, help='ORACLE_ROWS.json of a P1e run')
    parser.add_argument('--model', default='google/gemma-2-2b')
    parser.add_argument('--source', required=True, help='cross-layer transcoder reference')
    parser.add_argument('--target', default='gemma')
    parser.add_argument('--units', type=int, default=20)
    parser.add_argument('--registry', type=Path, default=None,
                        help='published explanations; adds the share of each reconstruction carried by published '
                             'supernode features of the same graph, layer and position')
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
    model = replacement_model(args.model, args.source, dev, torch.bfloat16)
    model.requires_grad_(False)
    A = Dictionary(model.transcoders)
    B = Dictionary(load_target(args.target, dev, torch.bfloat16))
    A.t.requires_grad_(False)
    B.t.requires_grad_(False)
    L = model.cfg.n_layers
    prompts = {}
    for r in json.loads(args.prompts_from.read_text())['rows']:
        prompts.setdefault(r['graph'], r['prompt'])
    published = {}
    if args.registry is not None:
        for key, g in json.loads(args.registry.read_text())['graphs'].items():
            published[key] = {(l, p, f): name for name, fs in g['variants'][0]['supernodes'].items() for l, p, f in fs}
    log(stage='loaded', source=args.source, target=args.target, prompts=len(prompts))
    names = [f'blocks.{l}.{A.input_hook}' for l in range(L)] + [f'blocks.{l}.{B.input_hook}' for l in range(L)]
    gen = torch.Generator().manual_seed(0)

    def run(ids, deltas):
        with torch.no_grad():
            logits = model.run_with_hooks(ids, fwd_hooks=add_hooks(deltas))
        return torch.log_softmax(logits[0, -1].float(), -1)

    rows = []
    for key, prompt in prompts.items():
        graph = attribute(prompt, model, max_n_logits=10, desired_logit_prob=0.95, batch_size=128)
        scores = prune_graph(graph).cumulative_scores
        n_feat = len(graph.selected_features)
        order = torch.argsort(scores[:n_feat])[:args.units]
        nodes = [tuple(int(v) for v in graph.active_features[graph.selected_features[i]]) for i in order.tolist()]
        ids = model.tokenizer(prompt, return_tensors='pt').input_ids.to(dev)
        n = ids.shape[1]
        with torch.no_grad():
            _, cache = model.run_with_cache(ids, names_filter=lambda n_: n_ in names)
            zb = {l: B.encode(cache[f'blocks.{l}.{B.input_hook}'][0], l).float() for l in range(L)}
        base = run(ids, {})
        units = [(f'L{l}F{f}P{p}', [(l, p, f)]) for l, p, f in nodes]
        for p in sorted({p for _, p, _ in nodes}):
            group = [(l, pp, f) for l, pp, f in nodes if pp == p]
            if len(group) > 1:
                units.append((f'pos{p}x{len(group)}', group))
        for name, feats in units:
            src, rec = {}, {}
            resid, resid_random, spans = [], [], []
            mass_total, mass_pub, pub_names = 0., 0., {}
            by_offset = []
            for l, p, f in feats:
                za = A.encode(cache[f'blocks.{l}.{A.input_hook}'][0, p], l, [f]).float()
                dec = A.decoders(l, [f])[0]
                spans.append(dec.shape[0])
                for k in range(dec.shape[0]):
                    target = za[0] * dec[k]
                    src.setdefault(l + k, torch.zeros(n, target.shape[0], device=dev))[p] -= target
                    act = (zb[l + k][p] > 0).nonzero().flatten()
                    if float(target.norm()) == 0. or len(act) == 0:
                        continue
                    D = B.decoders(l + k, act.tolist())[:, 0]
                    z = zb[l + k][p, act]
                    c = box_least_squares(D, target, z)
                    recon = c @ D
                    # Share of the reconstruction (coefficient times decoder norm) on published supernode features.
                    weight = (c * D.norm(dim=1)).tolist()
                    for j, wj in zip(act.tolist(), weight):
                        mass_total += wj
                        hit = published.get(key, {}).get((l + k, p, j))
                        if hit is not None:
                            mass_pub += wj
                            pub_names[hit] = pub_names.get(hit, 0.) + wj
                    resid.append(float((target - recon).norm() / target.norm()))
                    by_offset.append((k, float((target - recon).norm() / target.norm()), float(target.norm())))
                    rnd = torch.randn(target.shape, generator=gen).to(dev)
                    rnd = rnd * target.norm() / rnd.norm()
                    resid_random.append(float((rnd - box_least_squares(D, rnd, z) @ D).norm() / rnd.norm()))
                    rec.setdefault(l + k, torch.zeros(n, target.shape[0], device=dev))[p] -= recon
            ea, eb = run(ids, src) - base, run(ids, rec) - base
            w = base.exp()
            energy = float((w * ea ** 2).sum())
            top = lambda e: model.tokenizer.decode(int((base + e).argmax()))
            rows.append(dict(graph=key, unit=name, features=len(feats), mean_span=float(np.mean(spans)), prompt=prompt,
                             source_energy=energy, site_residual=float(np.mean(resid)) if resid else None,
                             site_residual_random=float(np.mean(resid_random)) if resid_random else None,
                             nrmse_layerwise=float((w * (eb - ea) ** 2).sum() / max(energy, 1e-12)) ** .5,
                             residual_by_offset=by_offset,
                             published_share=mass_pub / mass_total if mass_total > 0 else None,
                             published_supernodes={k_: v / mass_total for k_, v in pub_names.items()} if mass_total > 0 else {},
                             top_base=top(torch.zeros_like(base)), top_source=top(ea), top_layerwise=top(eb)))
        log(stage='graph', graph=key, units=len(units))
    (args.out / 'REVERSE_ROWS.json').write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(),
                                                               description=DESCRIPTION, rows=rows), indent=1))
    log(stage='done', rows=len(rows))


if __name__ == '__main__':
    main()
