import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from common import Dictionary, filter_items, gemma_items, load_target, replacement_model

DESCRIPTION = ('Features of a dictionary that are active at almost every token: over every position (after the first) '
               'of the graph prompts and their concept-preserving variants, the features per layer active at a share '
               'of at least --threshold of positions, with their mean activation and, for every layer they write into, '
               'the mean cosine of their claimed write with the MLP output there. With --units, the share of a '
               'graph\'s top units that are such features.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--model', default='google/gemma-2-2b')
    parser.add_argument('--dictionary', required=True)
    parser.add_argument('--as-replacement', action='store_true', help='load the dictionary inside the replacement model')
    parser.add_argument('--prompts-from', type=Path, required=True)
    parser.add_argument('--families', type=Path, default=None)
    parser.add_argument('--units', type=Path, default=None, help='registry of the dictionary\'s own graph units')
    parser.add_argument('--registry-model', default='gemma-2-2b')
    parser.add_argument('--threshold', type=float, default=.95)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'config.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    t0 = time.time()
    dev = torch.device('cuda')
    model = replacement_model(args.model, args.dictionary if args.as_replacement else 'gemma', dev, torch.bfloat16)
    model.requires_grad_(False)
    D = Dictionary(model.transcoders if args.as_replacement else load_target(args.dictionary, dev, torch.bfloat16))
    L = model.cfg.n_layers
    texts = sorted({r['prompt'] for r in json.loads(args.prompts_from.read_text())['rows']})
    if args.families is not None:
        for g in json.loads(args.families.read_text())['families'].values():
            texts += [g['template'].format(x) for x in g['fill']]
    names = [f'blocks.{l}.{D.input_hook}' for l in range(L)] + [f'blocks.{k}.hook_mlp_out' for k in range(L)]

    def caches():
        for text in texts:
            ids = model.tokenizer(text, return_tensors='pt').input_ids.to(dev)
            with torch.no_grad():
                _, cache = model.run_with_cache(ids, names_filter=lambda n_: n_ in names)
            yield cache

    # First pass: activation share of every feature; second pass: cosines of the dense features' writes.
    count, act_sum, total = torch.zeros(L, D.t.d_transcoder, device=dev), torch.zeros(L, D.t.d_transcoder, device=dev), 0
    for cache in caches():
        for l in range(L):
            z = D.encode(cache[f'blocks.{l}.{D.input_hook}'][0, 1:], l).float()
            count[l] += (z > 0).float().sum(0)
            act_sum[l] += z.sum(0)
        total += z.shape[0]
    share = count / total
    picked = {l: torch.nonzero(share[l] >= args.threshold).flatten().tolist() for l in range(L)}
    decs = {(l, f): D.decoders(l, [f])[0] for l in range(L) for f in picked[l]}
    cos = {key: [0.] * dec.shape[0] for key, dec in decs.items()}
    n_on = {key: 0 for key in decs}
    for cache in caches():
        for l in range(L):
            if not picked[l]:
                continue
            z = D.encode(cache[f'blocks.{l}.{D.input_hook}'][0, 1:], l, picked[l]).float().reshape(-1, len(picked[l]))
            for j, f in enumerate(picked[l]):
                for p in torch.nonzero(z[:, j] > 0).flatten().tolist():
                    n_on[(l, f)] += 1
                    for k in range(decs[(l, f)].shape[0]):
                        m = cache[f'blocks.{l + k}.hook_mlp_out'][0, p + 1].float()
                        cos[(l, f)][k] += float(torch.nn.functional.cosine_similarity(decs[(l, f)][k], m, dim=0))
    dense = {f'L{l}F{f}': dict(layer=l, share=float(share[l, f]), mean_activation=float(act_sum[l, f] / count[l, f]),
                               cosine_by_written_layer=[c / max(n_on[(l, f)], 1) for c in cos[(l, f)]])
             for (l, f) in decs}
    out = dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, dictionary=args.dictionary,
               texts=len(texts), positions=total, threshold=args.threshold,
               dense_per_layer=[sum(1 for v in dense.values() if v['layer'] == l) for l in range(L)], dense=dense)
    if args.units is not None:
        items, _ = filter_items(gemma_items(args.units, args.registry_model), L, D.t.d_transcoder)
        units = [x for _, _, sn in items for fs in sn.values() for x in fs]
        hits = [x for x in units if f'L{x[0]}F{x[2]}' in dense]
        out['top_units'] = len(units)
        out['top_units_dense'] = len(hits)
        out['top_units_dense_features'] = sorted({f'L{x[0]}F{x[2]}' for x in hits})
    out['elapsed'] = time.time() - t0
    (args.out / 'DENSE_SCAN.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'dense'}), flush=True)
    for name, v in sorted(dense.items(), key=lambda kv: kv[1]['layer']):
        c = v['cosine_by_written_layer']
        print(name, round(v['share'], 3), round(v['mean_activation'], 2), 'cos last', [round(x, 2) for x in c[-4:]], flush=True)


if __name__ == '__main__':
    main()
