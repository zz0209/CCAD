import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from circuit_tracer import ReplacementModel
from common import Dictionary, filter_items, gemma_items, load_target

DESCRIPTION = ('How alike two dictionaries are on the features of published explanations: for every supernode '
               'feature of the source dictionary, the highest activation correlation with any target feature of the '
               'same layer on generic text; median and quartiles per target.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--model', default='google/gemma-2-2b')
    parser.add_argument('--registry-model', default='gemma-2-2b')
    parser.add_argument('--source', default='gemma')
    parser.add_argument('--targets', nargs='+', required=True)
    parser.add_argument('--text', type=Path, required=True)
    parser.add_argument('--docs', type=int, default=256)
    parser.add_argument('--context', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dev = torch.device('cuda')
    model = ReplacementModel.from_pretrained(args.model, args.source, dtype=torch.bfloat16, backend='transformerlens')
    model.requires_grad_(False)
    A = Dictionary(model.transcoders)
    A.t.requires_grad_(False)
    L = model.cfg.n_layers
    items, _ = filter_items(gemma_items(args.registry, args.registry_model), L, A.t.d_transcoder)
    need = {}
    for _, _, sn in items:
        for fs in sn.values():
            for l, _, f in fs:
                need.setdefault(l, set()).add(f)
    need = {l: sorted(fs) for l, fs in need.items()}
    texts = json.loads(args.text.read_text())['train'][:args.docs * 2]
    seqs = [model.tokenizer(t, return_tensors='pt').input_ids[0, :args.context] for t in texts]
    seqs = torch.stack([s for s in seqs if len(s) == args.context][:args.docs])
    out = dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, model=args.model,
               source=args.source, features=sum(len(v) for v in need.values()), targets={})
    for ref in args.targets:
        B = Dictionary(load_target(ref, dev, torch.bfloat16))
        B.t.requires_grad_(False)
        names = {f'blocks.{l}.{A.input_hook}' for l in need} | {f'blocks.{l}.{B.input_hook}' for l in need}
        st = {l: dict(n=0., sa=0., sb=0., saa=0., sbb=0., sab=0.) for l in need}
        with torch.no_grad():
            for k in range(0, len(seqs), 8):
                _, cache = model.run_with_cache(seqs[k:k + 8].to(dev), names_filter=lambda n: n in names)
                for l, fs in need.items():
                    za = A.encode(cache[f'blocks.{l}.{A.input_hook}'][:, 1:], l, fs).float().flatten(0, 1)
                    zb = B.encode(cache[f'blocks.{l}.{B.input_hook}'][:, 1:], l).float().flatten(0, 1)
                    s = st[l]
                    s['n'] += za.shape[0]
                    for key, v in [('sa', za.sum(0)), ('sb', zb.sum(0)), ('saa', (za ** 2).sum(0)),
                                   ('sbb', (zb ** 2).sum(0)), ('sab', za.T @ zb)]:
                        s[key] = s[key] + v
        best = []
        for l, fs in need.items():
            s = st[l]
            ma, mb = s['sa'] / s['n'], s['sb'] / s['n']
            corr = (s['sab'] / s['n'] - ma[:, None] * mb[None]) / ((s['saa'] / s['n'] - ma ** 2)[:, None].clamp_min(1e-12) *
                                                                  (s['sbb'] / s['n'] - mb ** 2)[None].clamp_min(1e-12)).sqrt()
            silent = (s['saa'] / s['n'] - ma ** 2) <= 1e-12
            best += corr.max(1).values[~silent].tolist()
        best = np.array(best)
        out['targets'][ref] = dict(median=float(np.median(best)), q25=float(np.quantile(best, .25)),
                                   q75=float(np.quantile(best, .75)), above_09=float(np.mean(best > .9)), n=len(best))
        print(ref, out['targets'][ref], flush=True)
        del B
        torch.cuda.empty_cache()
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
