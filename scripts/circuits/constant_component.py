import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from common import Dictionary, replacement_model

DESCRIPTION = ('Is the part of the late MLP outputs that a CLT feature claims the same at every token? For each listed '
               'feature and each later layer k, over every position (after the first) of the graph prompts and their '
               'variants: the projection of the MLP output of layer k on the unit direction of the feature\'s decoder '
               'for k, its mean and standard deviation across positions and their ratio, and the same for the '
               'feature\'s own claimed write a d_k; for comparison, the projection on random unit directions.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--model', default='google/gemma-2-2b')
    parser.add_argument('--clt', required=True)
    parser.add_argument('--features', nargs='+', required=True)
    parser.add_argument('--prompts-from', type=Path, required=True)
    parser.add_argument('--families', type=Path, default=None)
    parser.add_argument('--last', type=int, default=6, help='number of last layers reported')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'config.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    dev = torch.device('cuda')
    torch.manual_seed(0)
    model = replacement_model(args.model, args.clt, dev, torch.bfloat16)
    model.requires_grad_(False)
    C = Dictionary(model.transcoders)
    L = model.cfg.n_layers
    texts = sorted({r['prompt'] for r in json.loads(args.prompts_from.read_text())['rows']})
    if args.families is not None:
        for g in json.loads(args.families.read_text())['families'].values():
            texts += [g['template'].format(x) for x in g['fill']]
    feats = [(int(s[1:].split('F')[0]), int(s.split('F')[1])) for s in args.features]
    layers = list(range(L - args.last, L))
    rand = torch.nn.functional.normalize(torch.randn(len(layers), 8, model.cfg.d_model, device=dev), dim=-1)
    proj = {s: {k: [] for k in layers} for s in args.features}
    claim = {s: {k: [] for k in layers} for s in args.features}
    rproj = {k: [] for k in layers}
    names = [f'blocks.{l}.{C.input_hook}' for l, _ in feats] + [f'blocks.{k}.hook_mlp_out' for k in layers]
    for text in texts:
        ids = model.tokenizer(text, return_tensors='pt').input_ids.to(dev)
        with torch.no_grad():
            _, cache = model.run_with_cache(ids, names_filter=lambda n_: n_ in names)
        for j, k in enumerate(layers):
            m = cache[f'blocks.{k}.hook_mlp_out'][0, 1:].float()
            rproj[k] += (m @ rand[j].T).flatten().tolist()
        for s, (l, f) in zip(args.features, feats):
            z = C.encode(cache[f'blocks.{l}.{C.input_hook}'][0, 1:], l, [f]).float().reshape(-1)
            dec = C.decoders(l, [f])[0]
            for k in layers:
                d = dec[k - l]
                m = cache[f'blocks.{k}.hook_mlp_out'][0, 1:].float()
                proj[s][k] += (m @ (d / d.norm())).tolist()
                claim[s][k] += (z * d.norm()).tolist()
    stats = lambda v: dict(mean=float(torch.tensor(v).mean()), std=float(torch.tensor(v).std()),
                           ratio=float(torch.tensor(v).std() / torch.tensor(v).mean().abs()))
    out = dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, texts=len(texts),
               features={s: {str(k): dict(mlp_projection=stats(proj[s][k]), claimed=stats(claim[s][k])) for k in layers}
                         for s in args.features},
               random_directions={str(k): dict(mean_abs=float(torch.tensor(rproj[k]).abs().mean()),
                                                std=float(torch.tensor(rproj[k]).std())) for k in layers})
    (args.out / 'CONSTANT_COMPONENT.json').write_text(json.dumps(out, indent=1))
    for s in args.features:
        for k in layers:
            p, c = out['features'][s][str(k)]['mlp_projection'], out['features'][s][str(k)]['claimed']
            print(s, k, 'mlp proj mean %.2f std %.2f' % (p['mean'], p['std']), 'claimed mean %.2f std %.2f' % (c['mean'], c['std']),
                  'random std %.2f' % out['random_directions'][str(k)]['std'], flush=True)


if __name__ == '__main__':
    main()
