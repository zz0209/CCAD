import argparse
import json

import torch

from common import replacement_model

DESCRIPTION = ('How does the MLP input that the Gemma Scope 2 dictionaries read (mlp.hook_in) relate to the residual '
               'stream before the MLP (hook_resid_mid)? Prints, for a few layers, the relative difference between '
               'mlp.hook_in and ln2(resid_mid) = resid_mid / ln2.hook_scale * ln2.w, so that an edit of the MLP input '
               'can be carried to the residual stream.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--model', default='google/gemma-3-1b-pt')
    parser.add_argument('--clt', default='mwhanna/gemma-scope-2-1b-pt/clt/width_262k_l0_medium_affine')
    parser.add_argument('--prompt', default='Fact: the capital of the state containing Dallas is')
    args = parser.parse_args()
    dev = torch.device('cuda')
    model = replacement_model(args.model, args.clt, dev, torch.bfloat16)
    ids = model.tokenizer(args.prompt, return_tensors='pt').input_ids.to(dev)
    L = model.cfg.n_layers
    names = [f'blocks.{l}.{h}' for l in range(L) for h in ('hook_resid_mid', 'ln2.hook_scale', 'ln2.hook_normalized', 'mlp.hook_in')]
    names = [n for n in names if n in model.hook_dict]
    with torch.no_grad():
        _, cache = model.run_with_cache(ids, names_filter=lambda n: n in names)
    out = {}
    for l in [0, 1, 5, 12, L - 1]:
        x = cache[f'blocks.{l}.hook_resid_mid'][0].float()
        hin = cache[f'blocks.{l}.mlp.hook_in'][0].float()
        w = model.blocks[l].ln2.w.float()
        scale = cache[f'blocks.{l}.ln2.hook_scale'][0].float()
        pred = x / scale * w
        out[l] = dict(rel_diff=float((hin - pred).norm() / hin.norm()),
                      normalized_vs_hook_in=float((cache[f'blocks.{l}.ln2.hook_normalized'][0].float() * w - hin).norm() / hin.norm()),
                      scale_shape=list(scale.shape), w_mean=float(w.mean()), w_min_abs=float(w.abs().min()),
                      w_small=int((w.abs() < 1e-2).sum()), scale_at_positions=scale.flatten().tolist())
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
