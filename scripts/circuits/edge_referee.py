import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from common import Dictionary, filter_items, gemma_items, replacement_model

DESCRIPTION = ('Edges of a dictionary\'s own attribution graph checked against the model. For every ordered pair of '
               'its top units, u read at layer l_u and position p_u and v read later at a position not before p_u, the '
               'graph claims that u changes the pre-activation of v through u\'s writes. The claimed change deletes '
               'u\'s writes with every MLP output of the model held at its clean value plus the deletion, so that only '
               'the direct paths through the residual stream and attention remain, as in the graph\'s edges. The '
               'model\'s change removes the input that u reads (the smallest change of the dictionary\'s input at '
               '(l_u, p_u) that switches u off) and lets the model run. Both changes of v\'s pre-activation are stored.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--model', default='google/gemma-2-2b')
    parser.add_argument('--dictionary', required=True)
    parser.add_argument('--units-from', type=Path, default=None, help='REVERSE_ROWS.json of a CLT graph')
    parser.add_argument('--registry', type=Path, default=None, help='registry of a PLT graph\'s own units')
    parser.add_argument('--registry-model', default='gemma-2-2b')
    parser.add_argument('--prompts-from', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--edit-residual', action='store_true',
                        help='carry the removal of u from its MLP-input hook to hook_resid_mid, as d_resid = d_in * '
                             'ln2.hook_scale / ln2.w, so that it reaches later layers through the residual stream')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'config.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    t0 = time.time()
    dev = torch.device('cuda')
    model = replacement_model(args.model, args.dictionary, dev, torch.bfloat16)
    model.requires_grad_(False)
    D = Dictionary(model.transcoders)
    L = model.cfg.n_layers
    prompts = {}
    for r in json.loads(args.prompts_from.read_text())['rows']:
        prompts.setdefault(r['graph'], r['prompt'])
    units = {}
    if args.units_from is not None:
        for r in json.loads(args.units_from.read_text())['rows']:
            if r['features'] == 1:
                l, rest = r['unit'][1:].split('F')
                f, p = rest.split('P')
                units.setdefault(r['graph'], []).append((r['unit'], int(l), int(p), int(f)))
    else:
        items, _ = filter_items(gemma_items(args.registry, args.registry_model), L, D.t.d_transcoder)
        for key, _, sn in items:
            units[key] = [(name, l, p, f) for name, fs in sn.items() for l, p, f in fs[:1]]

    def encoder(l, feats):
        if hasattr(D.t, 'transcoders'):
            tc = D.t.transcoders[l]
            return tc.W_enc[feats].float(), tc.b_enc[feats].float()
        return D.t.W_enc[l][feats].float(), D.t.b_enc[l][feats].float()

    names_in = [f'blocks.{l}.{D.input_hook}' for l in range(L)]
    names_out = [f'blocks.{k}.hook_mlp_out' for k in range(L)]
    names_scale = [f'blocks.{k}.ln2.hook_scale' for k in range(L)] if args.edit_residual else []

    def pre_acts(cache, us):
        # Pre-activations of the listed units from a cache of the dictionary's inputs.
        out = []
        for _, l, p, f in us:
            W, b = encoder(l, [f])
            out.append(float((W @ cache[f'blocks.{l}.{D.input_hook}'][0, p].float() + b)[0]))
        return torch.tensor(out)

    rows = []
    for g, us in units.items():
        if g not in prompts:
            continue
        ids = model.tokenizer(prompts[g], return_tensors='pt').input_ids.to(dev)
        n = ids.shape[1]
        us = [u for u in us if u[2] < n]
        with torch.no_grad():
            _, base = model.run_with_cache(ids, names_filter=lambda x: x in names_in + names_out + names_scale)
        pre0 = pre_acts(base, us)
        for i, (uname, l, p, f) in enumerate(us):
            targets = [j for j, (_, lv, pv, _) in enumerate(us) if lv > l and pv >= p]
            if not targets:
                continue
            x = base[f'blocks.{l}.{D.input_hook}'][0, p].float()
            a = D.encode(base[f'blocks.{l}.{D.input_hook}'][0, p], l, [f]).float().reshape(1)
            dec = D.decoders(l, [f])[0]
            # Claimed: delete u's writes with every MLP output held at its clean value plus the deletion.
            def clamp(act, hook, l=l, p=p, a=a, dec=dec):
                k = int(hook.name.split('.')[1])
                new = base[hook.name].clone()
                if l <= k < l + dec.shape[0]:
                    new[0, p] = new[0, p] - (a * dec[k - l]).to(new.dtype)
                return new
            # Model: remove the input that u reads at (l, p) and let the model run.
            W, b = encoder(l, [f])
            delta = -W.T @ torch.linalg.solve(W @ W.T + 1e-6 * torch.eye(1, device=dev), W @ x + b)
            remove_at = f'blocks.{l}.{D.input_hook}'
            if args.edit_residual:
                # Pseudo-inverse of the norm weight, whose zero dimensions no residual change can reach.
                w_ln = model.blocks[l].ln2.w.float()
                w_inv = torch.where(w_ln != 0, 1 / w_ln, torch.zeros_like(w_ln))
                delta = delta * base[f'blocks.{l}.ln2.hook_scale'][0, p].float() * w_inv
                remove_at = f'blocks.{l}.hook_resid_mid'
            def remove(act, hook, p=p, delta=delta):
                act[0, p] = act[0, p] + delta.to(act.dtype)
                return act
            res = {}
            for name, hooks in [('claimed', [(nm, clamp) for nm in names_out]),
                                ('model', [(remove_at, remove)])]:
                store = {}
                def grab(act, hook):
                    store[hook.name] = act.detach().clone()
                    return act
                with torch.no_grad():
                    model.run_with_hooks(ids, fwd_hooks=hooks + [(nm, grab) for nm in names_in])
                res[name] = pre_acts(store, us) - pre0
            for j in targets:
                vname, lv, pv, fv = us[j]
                rows.append(dict(graph=g, u=uname, v=vname, u_layer=l, v_layer=lv, u_pos=p, v_pos=pv,
                                 u_activation=float(a), v_pre=float(pre0[j]), claimed=float(res['claimed'][j]),
                                 model=float(res['model'][j])))
        print(json.dumps(dict(elapsed=round(time.time() - t0, 1), graph=g, rows=len(rows))), flush=True)
    (args.out / 'EDGE_REFEREE_ROWS.json').write_text(json.dumps(dict(
        written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, dictionary=args.dictionary,
        rows=rows), indent=1))
    print(json.dumps(dict(stage='done', rows=len(rows))), flush=True)


if __name__ == '__main__':
    main()
