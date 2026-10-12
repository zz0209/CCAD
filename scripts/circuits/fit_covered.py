import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from circuit_tracer import ReplacementModel
from common import Dictionary, add_hooks, choose_prompt, filter_items, gemma_items, load_target

DESCRIPTION = ('CIRCUITS_ACROSS_DICTIONARIES coverage test on circuits: the generic-text correspondence of P1 '
               '(fit_transfer.py) plus the explanation\'s own uses. Each step adds, to the generic-text error, the error '
               'of the final-position next-token distribution on concept-preserving variants of the published prompts '
               '(PROMPT_FAMILIES.json; the published prompt itself is never fitted), with requests that delete one '
               'supernode or a random subset of the graph\'s supernode features at their published positions. '
               'Evaluation deletes each supernode on the published prompts, as in P1, separately for graphs with and '
               'without prompt variants.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--families', type=Path, required=True)
    parser.add_argument('--source', default='gemma')
    parser.add_argument('--target', required=True)
    parser.add_argument('--text', type=Path, required=True)
    parser.add_argument('--docs', type=int, default=1024)
    parser.add_argument('--context', type=int, default=64)
    parser.add_argument('--candidates', type=int, default=8)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--task-batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=.02)
    parser.add_argument('--init', type=float, default=.05)
    parser.add_argument('--init-mode', choices=['uniform', 'best'], default='uniform')
    parser.add_argument('--slices', action='store_true', help='one weight per output layer of each candidate')
    parser.add_argument('--no-text', action='store_true', help='fit on the prompt variants only')
    parser.add_argument('--seed', type=int, default=0)
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
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device('cuda')
    model = ReplacementModel.from_pretrained('google/gemma-2-2b', args.source, dtype=torch.bfloat16, backend='transformerlens')
    model.requires_grad_(False)
    A = Dictionary(model.transcoders)
    B = Dictionary(load_target(args.target, dev, torch.bfloat16))
    model.transcoders.requires_grad_(False)
    B.t.requires_grad_(False)
    L = model.cfg.n_layers
    log(stage='loaded', source=args.source, target=args.target)

    items, dropped = filter_items(gemma_items(args.registry), L, model.transcoders.d_transcoder)
    chosen = []
    for key, prompt, sn in items:
        def active_fraction(text, sn=sn):
            ids = model.tokenizer(text, return_tensors='pt').input_ids.to(dev)
            hooks = {f'blocks.{l}.{A.input_hook}' for fs in sn.values() for l, _, _ in fs}
            with torch.no_grad():
                _, cache = model.run_with_cache(ids, names_filter=lambda n: n in hooks)
            flags = [p < ids.shape[1] and float(A.encode(cache[f'blocks.{l}.{A.input_hook}'][0, p], l, [f])) > 0
                     for fs in sn.values() for l, p, f in fs]
            return sum(flags) / len(flags)
        text, frac = choose_prompt(prompt, active_fraction)
        chosen.append((key, text, sn, frac))
    items = [(k, t, sn) for k, t, sn, _ in chosen]
    prompt_activity = {k: fr for k, _, _, fr in chosen}
    feats = sorted({(l, f) for _, _, sn in items for fs in sn.values() for l, _, f in fs})
    layers = sorted({l for l, _ in feats})
    by_layer = {l: [f for ll, f in feats if ll == l] for l in layers}
    index = {lf: i for i, lf in enumerate(feats)}

    # Prompt variants with the token length of the chosen published prompt, the published prompt excluded.
    families = json.loads(args.families.read_text())['families']
    variants = {}
    for key, prompt, sn in items:
        if key not in families:
            continue
        n0 = len(model.tokenizer(prompt).input_ids)
        fam = families[key]
        for fill in fam['fill']:
            text = fam['template'].format(fill)
            if text.strip() != prompt.strip() and len(model.tokenizer(text).input_ids) == n0:
                variants.setdefault(key, []).append(text)
    task = [(key, v) for key in variants for v in variants[key]]
    log(stage='graphs', kept=len(items), dropped=dropped, with_variants=len(variants), variants=len(task),
        source_features=len(feats))

    texts = json.loads(args.text.read_text())['train']
    seqs = [model.tokenizer(t, return_tensors='pt').input_ids[0, :args.context] for t in texts[:args.docs * 2]]
    seqs = torch.stack([s for s in seqs if len(s) == args.context][:args.docs])
    names = [f'blocks.{l}.{A.input_hook}' for l in layers] + [f'blocks.{l}.{B.input_hook}' for l in layers]

    def inputs(ids):
        with torch.no_grad():
            _, cache = model.run_with_cache(ids, names_filter=lambda n: n in names)
        return cache

    # Activation correlation of every source feature with every target feature of its layer, on generic text.
    st = {l: dict(n=0., sa=0., sb=0., saa=0., sbb=0., sab=0.) for l in layers}
    for k in range(0, len(seqs), 16):
        cache = inputs(seqs[k:k + 16].to(dev))
        with torch.no_grad():
            for l in layers:
                za = A.encode(cache[f'blocks.{l}.{A.input_hook}'][:, 1:], l, by_layer[l]).float().flatten(0, 1)
                zb = B.encode(cache[f'blocks.{l}.{B.input_hook}'][:, 1:], l).float().flatten(0, 1)
                s = st[l]
                s['n'] += za.shape[0]
                for key, v in [('sa', za.sum(0)), ('sb', zb.sum(0)), ('saa', (za ** 2).sum(0)),
                               ('sbb', (zb ** 2).sum(0)), ('sab', za.T @ zb)]:
                    s[key] = s[key] + v
    cand = {}
    for l in layers:
        s = st[l]
        mean_a, mean_b = s['sa'] / s['n'], s['sb'] / s['n']
        cov = s['sab'] / s['n'] - mean_a[:, None] * mean_b[None]
        corr = cov / ((s['saa'] / s['n'] - mean_a ** 2)[:, None].clamp_min(1e-12) *
                      (s['sbb'] / s['n'] - mean_b ** 2)[None].clamp_min(1e-12)).sqrt()
        top = corr.topk(args.candidates, dim=1)
        for i, f in enumerate(by_layer[l]):
            cand[(l, f)] = top.indices[i].tolist()
    log(stage='candidates')

    K = args.candidates
    S = L if args.slices else 1
    init = torch.full((len(feats), K, S), args.init, device=dev)
    if args.init_mode == 'best':
        init[:, 0] = 1.
    param = torch.nn.Parameter(init)
    opt = torch.optim.Adam([param], lr=args.lr)
    dec_a = {l: A.decoders(l, by_layer[l]) for l in layers}
    cand_flat = {l: [j for f in by_layer[l] for j in cand[(l, f)]] for l in layers}
    dec_b = {l: B.decoders(l, cand_flat[l]) for l in layers}

    def deltas(cache, q, weights, which, positions=None):
        # q: (batch, n_feats) deletion strengths; positions (batch, seq, n_feats) restricts the deletion to the
        # positions where the mask is one. Returns {output layer: (batch, seq, d_model)}.
        out = {}
        for l in layers:
            idx = [index[(l, f)] for f in by_layer[l]]
            ql = q[:, idx]
            mask = positions[:, :, idx] if positions is not None else None
            if which == 'A':
                z = A.encode(cache[f'blocks.{l}.{A.input_hook}'], l, by_layer[l]).float()
                c = -z * ql[:, None, :]
                if mask is not None:
                    c = c * mask
                for k in range(dec_a[l].shape[1]):
                    out[l + k] = out.get(l + k, 0.) + c @ dec_a[l][:, k]
                continue
            z = B.encode(cache[f'blocks.{l}.{B.input_hook}'], l, cand_flat[l]).float()
            mask = mask.repeat_interleave(K, -1) if mask is not None else None
            dec = dec_b[l]
            for k in range(dec.shape[1]):
                w = weights[idx][:, :, k if weights.shape[2] > 1 else 0]
                c = -z * (ql[:, :, None] * w[None]).flatten(1).clamp(max=1.)[:, None, :]
                if mask is not None:
                    c = c * mask
                out[l + k] = out.get(l + k, 0.) + c @ dec[:, k]
        return out

    with torch.no_grad():
        _, cache = model.run_with_cache(seqs[:64].to(dev), names_filter=lambda n: n == f'blocks.{L - 1}.hook_resid_post')
    lw = LedoitWolf().fit(cache[f'blocks.{L - 1}.hook_resid_post'][:, 1:].float().flatten(0, 1).cpu().numpy())
    evals, evecs = np.linalg.eigh(lw.covariance_)
    white = torch.tensor(evecs @ np.diag(evals ** -.5) @ evecs.T, device=dev, dtype=torch.float32)

    def readout(ids, hooks):
        store = {}

        def keep(act, hook):
            store['h'] = act
            return act
        logits = model.run_with_hooks(ids, fwd_hooks=hooks + [(f'blocks.{L - 1}.hook_resid_post', keep)])
        return torch.log_softmax(logits.float(), -1), store['h'].float()

    sn_of = {key: sn for key, _, sn in items}

    def request(key, n):
        # One supernode of the graph, or a random subset of its supernode features, at their published positions.
        sn = sn_of[key]
        q = torch.zeros(1, len(feats), device=dev)
        pos = torch.zeros(1, n, len(feats), device=dev)
        if rng.random() < .5:
            names_ = list(sn)
            members = sn[names_[rng.integers(len(names_))]]
        else:
            allf = [x for fs in sn.values() for x in fs]
            p = rng.random()
            members = [x for x in allf if rng.random() < p] or [allf[rng.integers(len(allf))]]
        for l, p_, f in members:
            if p_ < n:
                q[0, index[(l, f)]] = 1.
                pos[0, p_, index[(l, f)]] = 1.
        return q, pos

    task_cache = {}
    for key, text in task:
        ids = model.tokenizer(text, return_tensors='pt').input_ids.to(dev)
        task_cache[text] = ids
    n_train = int(len(seqs) * .8)
    for step in range(args.steps):
        weights = param.clamp(0., 1.)
        loss = 0.
        if not args.no_text:
            ix = rng.choice(n_train, args.batch, replace=False)
            ids = seqs[ix].to(dev)
            cache = inputs(ids)
            p = torch.rand(args.batch, 1, device=dev)
            q = (torch.rand(args.batch, len(feats), device=dev) < p).float()
            with torch.no_grad():
                lp0, h0 = readout(ids, [])
                lps, hs = readout(ids, add_hooks(deltas(cache, q, None, 'A')))
                w0 = lp0.exp()
                ds, dh = lps - lp0, (hs - h0) @ white
                eff = (w0 * ds ** 2).sum((1, 2)) + (dh ** 2).sum((1, 2))
            lpt, ht = readout(ids, add_hooks(deltas(cache, q, weights, 'B')))
            err = (w0 * (lpt - lp0 - ds) ** 2).sum((1, 2)) + (((ht - h0) @ white - dh) ** 2).sum((1, 2))
            loss = loss + err.sum() / eff.sum().clamp_min(1e-9)
        errs, effs = 0., 0.
        for j in rng.choice(len(task), args.task_batch, replace=False):
            key, text = task[j]
            ids = task_cache[text]
            cache = inputs(ids)
            q, pos = request(key, ids.shape[1])
            with torch.no_grad():
                lp0, _ = readout(ids, [])
                lps, _ = readout(ids, add_hooks(deltas(cache, q, None, 'A', positions=pos)))
                w0 = lp0[0, -1].exp()
                ds = lps[0, -1] - lp0[0, -1]
            lpt, _ = readout(ids, add_hooks(deltas(cache, q, weights, 'B', positions=pos)))
            errs = errs + (w0 * (lpt[0, -1] - lp0[0, -1] - ds) ** 2).sum()
            effs = effs + (w0 * ds ** 2).sum()
        loss = loss + errs / max(float(effs), 1e-9)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            param.clamp_(0., 1.)
        if (step + 1) % 25 == 0:
            log(stage='fit', step=step + 1, loss=float(loss), task_error=float(errs) / max(float(effs), 1e-9))
    weights = param.detach().clamp(0., 1.)
    one = torch.zeros_like(weights)
    one[:, 0] = 1.
    torch.save(dict(feats=feats, cand={str(k): v for k, v in cand.items()}, weights=weights.cpu()), args.out / 'maps.pt')

    rows = []
    for key, prompt, sn in items:
        ids = model.tokenizer(prompt, return_tensors='pt').input_ids.to(dev)
        cache = inputs(ids)
        with torch.no_grad():
            lp0, _ = readout(ids, [])
        base = lp0[0, -1]
        for sname, fs in sn.items():
            q = torch.zeros(1, len(feats), device=dev)
            pos = torch.zeros(1, ids.shape[1], len(feats), device=dev)
            for l, p_, f in fs:
                if p_ < ids.shape[1]:
                    q[0, index[(l, f)]] = 1.
                    pos[0, p_, index[(l, f)]] = 1.
            if q.sum() == 0:
                continue
            effects = {}
            with torch.no_grad():
                for name, which, w in [('source', 'A', None), ('fitted', 'B', weights), ('one_to_one', 'B', one)]:
                    lp, _ = readout(ids, add_hooks(deltas(cache, q, w, which, positions=pos)))
                    effects[name] = lp[0, -1] - base
            wb = base.exp()
            energy = float((wb * effects['source'] ** 2).sum())
            row = dict(graph=key, supernode=sname, features=len(fs), source_energy=energy, prompt=prompt,
                       covered=key in variants, prompt_activity=prompt_activity[key],
                       top_base=model.tokenizer.decode(int(base.argmax())))
            for name in ['fitted', 'one_to_one']:
                row[f'nrmse_{name}'] = float((wb * (effects[name] - effects['source']) ** 2).sum() / max(energy, 1e-12)) ** .5
                row[f'top_{name}'] = model.tokenizer.decode(int((base + effects[name]).argmax()))
            row['top_source'] = model.tokenizer.decode(int((base + effects['source']).argmax()))
            rows.append(row)
        log(stage='graph', graph=key)
    (args.out / 'P1_ROWS.json').write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(),
                                                          description=DESCRIPTION, heldout_text={}, rows=rows), indent=1))
    log(stage='done', rows=len(rows))


if __name__ == '__main__':
    main()
