import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from common import Dictionary, replacement_model

DESCRIPTION = ('How often listed CLT features are active, over every position of the graph prompts and of their '
               'concept-preserving variants, and how much of the later MLP outputs they claim: for each feature the '
               'share of positions where it is active, its mean activation there, and the mean cosine between its '
               'claimed write into each later layer and the MLP output of that layer at active positions.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--model', default='google/gemma-2-2b')
    parser.add_argument('--clt', required=True)
    parser.add_argument('--features', nargs='+', required=True, help='LxFy')
    parser.add_argument('--prompts-from', type=Path, required=True)
    parser.add_argument('--families', type=Path, default=None, help='PROMPT_FAMILIES.json with variants')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'config.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    t0 = time.time()
    dev = torch.device('cuda')
    model = replacement_model(args.model, args.clt, dev, torch.bfloat16)
    model.requires_grad_(False)
    C = Dictionary(model.transcoders)
    L = model.cfg.n_layers
    feats = [(int(s[1:].split('F')[0]), int(s.split('F')[1])) for s in args.features]
    texts = sorted({r['prompt'] for r in json.loads(args.prompts_from.read_text())['rows']})
    if args.families is not None:
        fam = json.loads(args.families.read_text())
        for g in fam['families'].values():
            texts += [g['template'].format(x) for x in g['fill']]
    stats = {f'L{l}F{f}': dict(positions=0, active=0, act_sum=0., cos_sum=[0.] * L, cos_n=0) for l, f in feats}
    names = [f'blocks.{l}.{C.input_hook}' for l in range(L)] + [f'blocks.{k}.hook_mlp_out' for k in range(L)]
    for text in texts:
        ids = model.tokenizer(text, return_tensors='pt').input_ids.to(dev)
        with torch.no_grad():
            _, cache = model.run_with_cache(ids, names_filter=lambda n_: n_ in names)
        n = ids.shape[1]
        for l, f in feats:
            s = stats[f'L{l}F{f}']
            z = C.encode(cache[f'blocks.{l}.{C.input_hook}'][0, 1:], l, [f]).float().reshape(-1)
            s['positions'] += n - 1
            s['active'] += int((z > 0).sum())
            s['act_sum'] += float(z[z > 0].sum())
            dec = C.decoders(l, [f])[0]
            for p in torch.nonzero(z > 0).flatten().tolist():
                s['cos_n'] += 1
                for k in range(l, L):
                    m = cache[f'blocks.{k}.hook_mlp_out'][0, p + 1].float()
                    s['cos_sum'][k] += float(torch.nn.functional.cosine_similarity(dec[k - l], m, dim=0))
    out = {}
    for name, s in stats.items():
        out[name] = dict(positions=s['positions'], active_share=s['active'] / max(s['positions'], 1),
                         mean_activation=s['act_sum'] / max(s['active'], 1),
                         cosine_with_mlp_output=[c / max(s['cos_n'], 1) for c in s['cos_sum']])
        print(name, json.dumps({k: v for k, v in out[name].items() if k != 'cosine_with_mlp_output'}),
              'cos late', [round(c, 2) for c in out[name]['cosine_with_mlp_output'][-6:]], flush=True)
    (args.out / 'FEATURE_FREQUENCY.json').write_text(json.dumps(dict(
        written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, texts=len(texts),
        elapsed=time.time() - t0, features=out), indent=1))


if __name__ == '__main__':
    main()
