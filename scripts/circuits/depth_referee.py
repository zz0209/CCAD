import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from common import Dictionary, filter_items, gemma_items, load_target, replacement_model

DESCRIPTION = ('Where in depth does the model produce what a CLT unit claims to write? For each single-feature unit '
               'of a CLT graph (feature f read at layer l and position p), the CLT claims a write c_k = a d_k into '
               'the MLP output of every layer k >= l at p. Three removals of inputs are applied to the underlying '
               'model and the change of every MLP output at p is projected on c_k: (u) the smallest change of the '
               'residual stream at layer l and p that switches the unit off; (s) the same change, at their own MLP '
               'inputs, for the per-layer transcoder (PLT) units of the same prompt that sit at p after layer l; '
               '(r) the same for randomly drawn active PLT features matched to (s) in layer and number; (a) and (ra) '
               'repeat (s) and (r) with the PLT units at p and at every earlier position, which adds what attention '
               'brings to p. The removed '
               'fraction of c_k says whether the model computes the claimed write from that input. The unit\'s own '
               'layer calibrates redundancy of its input, since (u) is the same edit for every k. Logit effects of '
               'the three removals and of deleting the claimed writes with the MLP outputs at p clamped (no '
               'response of later MLPs at p, as in constrained patching) or free are stored as well. For every PLT '
               'unit of the same prompts, the fraction of its own write that its read removal takes out of the MLP '
               'output of its layer gives the same redundancy calibration on the PLT side.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--model', default='google/gemma-2-2b')
    parser.add_argument('--clt', required=True)
    parser.add_argument('--plt', required=True)
    parser.add_argument('--clt-units-from', type=Path, required=True, help='REVERSE_ROWS.json of the CLT graphs')
    parser.add_argument('--plt-units', type=Path, required=True, help='registry of the PLT graph units')
    parser.add_argument('--registry-model', default='gemma-2-2b')
    parser.add_argument('--prompts-from', type=Path, required=True, help='rows with the chosen prompt per graph')
    parser.add_argument('--max-graphs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'config.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    t0 = time.time()
    rng = random.Random(args.seed)
    dev = torch.device('cuda')
    model = replacement_model(args.model, args.clt, dev, torch.bfloat16)
    model.requires_grad_(False)
    C = Dictionary(model.transcoders)
    C.t.requires_grad_(False)
    P = Dictionary(load_target(args.plt, dev, torch.bfloat16))
    P.t.requires_grad_(False)
    L = model.cfg.n_layers
    for d in (C, P):
        assert f'blocks.0.{d.input_hook}' in model.hook_dict, d.input_hook

    prompts = {}
    for r in json.loads(args.prompts_from.read_text())['rows']:
        prompts.setdefault(r['graph'], r['prompt'])
    clt_units = {}
    for r in json.loads(args.clt_units_from.read_text())['rows']:
        if r['features'] == 1:
            l, rest = r['unit'][1:].split('F')
            f, p = rest.split('P')
            clt_units.setdefault(r['graph'], []).append((r['unit'], int(l), int(p), int(f)))
    items, _ = filter_items(gemma_items(args.plt_units, args.registry_model), L, P.t.d_transcoder)
    plt_units = {k: [x for fs in sn.values() for x in fs] for k, _, sn in items}
    graphs = [g for g in clt_units if g in prompts and g in plt_units]
    if args.max_graphs is not None:
        graphs = graphs[:args.max_graphs]

    def encoder(D, l, feats):
        if hasattr(D.t, 'transcoders'):
            tc = D.t.transcoders[l]
            return tc.W_enc[feats].float(), tc.b_enc[feats].float()
        return D.t.W_enc[l][feats].float(), D.t.b_enc[l][feats].float()

    def removal(D, l, x, feats):
        # Smallest change of x that sets every listed pre-activation to zero.
        W, b = encoder(D, l, feats)
        pre = W @ x + b
        return -W.T @ torch.linalg.solve(W @ W.T + 1e-6 * torch.eye(len(feats), device=dev), pre)

    def run(ids, hooks, p):
        # Final-position log-probabilities and the MLP outputs at position p of every layer.
        mlp = {}
        def grab(act, hook):
            mlp[int(hook.name.split('.')[1])] = act[0, p].float().clone()
            return act
        caps = [(f'blocks.{k}.hook_mlp_out', grab) for k in range(L)]
        with torch.no_grad():
            logits = model.run_with_hooks(ids, fwd_hooks=hooks + caps)
        return torch.log_softmax(logits[0, -1].float(), -1), torch.stack([mlp[k] for k in range(L)])

    def edit_hooks(edits):
        # edits: {(hook name, position): delta}
        by_name = {}
        for (name, p), delta in edits.items():
            by_name.setdefault(name, {})[p] = by_name.get(name, {}).get(p, 0) + delta
        hooks = []
        for name, by_pos in by_name.items():
            def hook(act, hook, by_pos=by_pos):
                for p, delta in by_pos.items():
                    act[0, p] = act[0, p] + delta.to(act.dtype)
                return act
            hooks.append((name, hook))
        return hooks

    rows, plt_rows = [], []
    for g in graphs:
        ids = model.tokenizer(prompts[g], return_tensors='pt').input_ids.to(dev)
        n = ids.shape[1]
        names = sorted({f'blocks.{k}.{d.input_hook}' for k in range(L) for d in (C, P)})
        with torch.no_grad():
            _, cache = model.run_with_cache(ids, names_filter=lambda n_: n_ in names)
        for unit, l, p, f in clt_units[g]:
            if p >= n:
                continue
            base, mlp0 = run(ids, [], p)
            w0 = base.exp()
            answer = int(base.argmax())
            x = cache[f'blocks.{l}.{C.input_hook}'][0, p].float()
            a = C.encode(cache[f'blocks.{l}.{C.input_hook}'][0, p], l, [f]).float().reshape(1)
            dec = C.decoders(l, [f])[0]  # (n_out, d_model) for layers l, l+1, ...
            claim = torch.zeros(L, dec.shape[-1], device=dev)
            claim[l:l + dec.shape[0]] = a * dec
            delta_u = removal(C, l, x, [f])
            check_u = float(C.encode((x + delta_u).to(C.dtype), l, [f]).float().abs().max())
            # PLT units of the same prompt at p after layer l, and random active PLT features matched in layer.
            # (s) at position p only; (a) at p and every earlier position, which adds what attention brings to p.
            mid, wide = {}, {}
            for ll, pp, ff in plt_units[g]:
                if l < ll < L and pp <= p:
                    wide.setdefault((ll, pp), set()).add(ff)
                    if pp == p:
                        mid.setdefault((ll, pp), set()).add(ff)
            def matched(sel):
                rand = {}
                for (ll, pp), fs in sel.items():
                    xa = cache[f'blocks.{ll}.{P.input_hook}'][0, pp]
                    active = torch.nonzero(P.encode(xa, ll).float().reshape(-1) > 0).flatten().tolist()
                    pool = [ff for ff in active if ff not in fs]
                    if len(pool) >= len(fs):
                        rand[(ll, pp)] = rng.sample(pool, len(fs))
                return rand if len(rand) == len(sel) else None
            def plt_edits(sel):
                return {(f'blocks.{ll}.{P.input_hook}', pp):
                        removal(P, ll, cache[f'blocks.{ll}.{P.input_hook}'][0, pp].float(), sorted(fs))
                        for (ll, pp), fs in sel.items()}
            conds = {'u': edit_hooks({(f'blocks.{l}.{C.input_hook}', p): delta_u})}
            for name, sel in [('s', mid), ('a', wide)]:
                rand = matched(sel) if sel else None
                if sel and rand is not None:
                    conds[name] = edit_hooks(plt_edits(sel))
                    conds['r' if name == 's' else 'ra'] = edit_hooks(plt_edits(rand))
            # Deleting the claimed writes: free (later MLPs respond) and clamped at p.
            free = {k: torch.zeros(n, dec.shape[-1], device=dev) for k in range(l, L)}
            for k in range(l, L):
                free[k][p] = -claim[k]
            conds['w_free'] = [(f'blocks.{k}.hook_mlp_out', (lambda act, hook, d=free[k]: act + d.to(act.dtype)[None]))
                               for k in range(l, L)]
            def clamp(act, hook):
                k = int(hook.name.split('.')[1])
                act[0, p] = (mlp0[k] - claim[k]).to(act.dtype)
                return act
            conds['w_clamped'] = [(f'blocks.{k}.hook_mlp_out', clamp) for k in range(l, L)]
            energy = (claim ** 2).sum(-1)
            row = dict(graph=g, unit=unit, layer=l, position=p, final=p == n - 1, activation=float(a),
                       claim_energy=energy[l:].tolist(), check_off=check_u,
                       plt_mid=sorted((ll, len(fs)) for (ll, _), fs in mid.items()),
                       plt_mid_units=sum(len(fs) for fs in mid.values()), plt_wide_units=sum(len(fs) for fs in wide.values()))
            for name, hooks in conds.items():
                lp, mlp = run(ids, hooks, p)
                e = lp - base
                row[f'energy_{name}'] = float((w0 * e ** 2).sum())
                row[f'drop_{name}'] = float(-e[answer])
                if not name.startswith('w_'):
                    removed = -((mlp - mlp0) * claim).sum(-1)
                    row[f'removed_{name}'] = removed[l:].tolist()
            rows.append(row)
        # Calibration on the PLT side: the fraction of each PLT unit's own write that its read removal takes out of
        # the MLP output of its layer, with the same edit as condition (s).
        for l, p, f in plt_units[g]:
            if p >= n:
                continue
            base, mlp0 = run(ids, [], p)
            xin = cache[f'blocks.{l}.{P.input_hook}'][0, p]
            a = P.encode(xin, l, [f]).float().reshape(1)
            claim = a * P.decoders(l, [f])[0, 0]
            delta = removal(P, l, xin.float(), [f])
            lp, mlp = run(ids, edit_hooks({(f'blocks.{l}.{P.input_hook}', p): delta}), p)
            plt_rows.append(dict(graph=g, unit=f'L{l}F{f}P{p}', layer=l, position=p, activation=float(a),
                                 claim_energy=float((claim ** 2).sum()),
                                 removed=float(-((mlp[l] - mlp0[l]) * claim).sum()),
                                 energy_read=float((base.exp() * (lp - base) ** 2).sum())))
        print(json.dumps(dict(elapsed=round(time.time() - t0, 1), graph=g, units=len(clt_units[g]), rows=len(rows))),
              flush=True)
    (args.out / 'DEPTH_REFEREE_ROWS.json').write_text(json.dumps(dict(
        written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, clt=args.clt, plt=args.plt,
        n_layers=L, rows=rows, plt_rows=plt_rows), indent=1))
    print(json.dumps(dict(stage='done', rows=len(rows))), flush=True)


if __name__ == '__main__':
    main()
