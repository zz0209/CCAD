import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch

from circuit_tracer import attribute
from circuit_tracer.graph import prune_graph

from common import Dictionary, replacement_model

DESCRIPTION = ('Do graphs filtered by the unit test predict held-out interventions better? On each held-out prompt the '
               'dictionary draws its circuit-tracer graph. The total effect of each feature node on the answer\'s '
               'demeaned logit, per unit of activation, is the sum over all paths of the graph\'s direct effects. For a '
               'CLT graph the --pool most influential units are tested (input removal, bounded share of all claimed '
               'writes S_all) and four node lists of --top units are formed: the most influential units (all), those '
               'that pass S_all >= --tau (test), the pool with as many randomly drawn units dropped as the test drops '
               '(random), and the units read at layer 2 or later (layer); a PLT graph gives its most influential units '
               '(plt). For each list, --subsets node subsets (the first 1, 2, 3, 5, 10 and 20 units and random subsets of '
               '1 to 10 units) are intervened on in the model: (rs) the projection of the dictionary input on each '
               'node\'s read direction is replaced by its value at a random position of another held-out prompt, and '
               '(rm) the input is removed so that each pre-activation is zero, jointly per site. The change is shown '
               'only to the MLPs that the dictionary claims the feature writes into (its layer and every later layer '
               'for a CLT, its own layer for a PLT), so that attention and the unembedding do not read it directly. The '
               'graph predicts the change of the answer logit as the sum over the subset of the change of each node\'s '
               'activation times its total effect; the model\'s change is measured with the changed MLP outputs '
               'propagating freely.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--model', required=True)
    parser.add_argument('--dictionary', required=True)
    parser.add_argument('--role', choices=['clt', 'plt'], required=True)
    parser.add_argument('--prompts', type=Path, required=True, help='rows of graph and prompt')
    parser.add_argument('--edit-residual', action='store_true',
                        help='carry edits of a dictionary that reads the MLP input to the residual stream before the MLP')
    parser.add_argument('--pool', type=int, default=60)
    parser.add_argument('--top', type=int, default=20)
    parser.add_argument('--subsets', type=int, default=40)
    parser.add_argument('--tau', type=float, default=.25)
    parser.add_argument('--max-prompts', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--attr-batch', type=int, default=128,
                        help='batch size of circuit-tracer attribution (memory only; the graph is the same)')
    parser.add_argument('--resume-from', type=Path, default=None,
                        help='FILTER_PREDICTION_ROWS.partial.json of an interrupted run with the same settings; its prompts '
                             'are kept and skipped (every prompt is computed independently, with its own random seed)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'config.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    # State for the local progress page: status.json (RUNNING, then PASS) and progress.json (current prompt).
    status = dict(status='RUNNING', started_at_utc=datetime.now(timezone.utc).isoformat())
    (args.out / 'status.json').write_text(json.dumps(status, indent=1))
    rows = json.loads(args.resume_from.read_text())['rows'] if args.resume_from else []
    done = {r['graph'] for r in rows}
    t0 = time.time()
    proc = psutil.Process()

    def memory():
        # Host memory of this process (private commit and resident set) and GPU memory held by torch, in GiB.
        m = proc.memory_info()
        return dict(private_gib=round(m.private / 2**30, 2), resident_gib=round(m.rss / 2**30, 2),
                    gpu_gib=round(torch.cuda.memory_reserved() / 2**30, 2))

    dev = torch.device('cuda')
    print(json.dumps(dict(stage='start', **memory())), flush=True)
    model = replacement_model(args.model, args.dictionary, dev, torch.bfloat16)
    model.requires_grad_(False)
    print(json.dumps(dict(stage='model', elapsed=round(time.time() - t0, 1), **memory())), flush=True)
    D = Dictionary(model.transcoders)
    D.t.requires_grad_(False)
    L = model.cfg.n_layers
    rows_in = json.loads(args.prompts.read_text())['rows']
    prompts = list(dict.fromkeys((r['graph'], r['prompt']) for r in rows_in))[:args.max_prompts]

    def encoder(l, feats):
        if hasattr(D.t, 'transcoders'):
            tc = D.t.transcoders[l]
            return tc.W_enc[feats].float(), tc.b_enc[feats].float()
        return D.t.W_enc[l][feats].float(), D.t.b_enc[l][feats].float()

    names = {f'blocks.{k}.{D.input_hook}' for k in range(L)} | {f'blocks.{k}.ln2.hook_scale' for k in range(L)}

    def cache_of(ids):
        with torch.no_grad():
            _, cache = model.run_with_cache(ids, names_filter=lambda n_: n_ in names)
        return cache

    # Donor bank: dictionary inputs at every position after the first of every held-out prompt, per layer.
    bank = {k: [] for k in range(L)}
    for g, prompt in prompts:
        ids = model.tokenizer(prompt, return_tensors='pt').input_ids.to(dev)
        cache = cache_of(ids)
        for k in range(L):
            for q in range(1, ids.shape[1]):
                bank[k].append((g, cache[f'blocks.{k}.{D.input_hook}'][0, q].float().cpu()))
    print(json.dumps(dict(stage='bank', elapsed=round(time.time() - t0, 1), vectors=len(bank[0]), **memory())), flush=True)

    for gi, (g, prompt) in enumerate(prompts):
        if g in done:
            continue
        rng = random.Random(f'{args.seed}-{g}')
        ids = model.tokenizer(prompt, return_tensors='pt').input_ids.to(dev)
        cache = cache_of(ids)
        graph = attribute(prompt, model, max_n_logits=10, desired_logit_prob=0.95, batch_size=args.attr_batch)
        A = graph.adjacency_matrix.float().to(dev)
        n_f = len(graph.selected_features)
        feats = graph.active_features[graph.selected_features].tolist()
        act = graph.activation_values[graph.selected_features].float().to(dev)
        target = int(graph.logit_probabilities.argmax())
        answer = int(graph.logit_token_ids[target])
        row_t = A.shape[0] - len(graph.logit_targets) + target
        # Total effect per unit activation: tau = w (I - W)^-1 over feature nodes, as a power series (the graph is
        # acyclic, so it ends after at most n_layers steps).
        W = A[:n_f, :n_f] / act[None, :]
        cur = A[row_t, :n_f] / act
        tau = cur.clone()
        for _ in range(4 * L):
            cur = cur @ W
            tau += cur
            if float(cur.abs().max()) < 1e-9:
                break
        order = torch.argsort(prune_graph(graph).cumulative_scores[:n_f])[:args.pool].tolist()
        pool = [dict(node=i, layer=feats[i][0], pos=feats[i][1], feature=feats[i][2], tau=float(tau[i])) for i in order]
        del A, W
        for u in pool:
            l, p, f = u['layer'], u['pos'], u['feature']
            u['act'] = float(D.encode(cache[f'blocks.{l}.{D.input_hook}'][0, p], l, [f]).float().reshape(-1)[0])

        def edit_hooks(deltas):
            # deltas: {(layer, pos): change of the dictionary input}, applied at the input hook or carried to the
            # residual stream before MLP layer.
            by_name = {}
            for (l, p), d in deltas.items():
                if args.edit_residual:
                    w_ln = model.blocks[l].ln2.w.float()
                    w_inv = torch.where(w_ln != 0, 1 / w_ln, torch.zeros_like(w_ln))
                    d = d * cache[f'blocks.{l}.ln2.hook_scale'][0, p].float() * w_inv
                    name = f'blocks.{l}.hook_resid_mid'
                else:
                    name = f'blocks.{l}.{D.input_hook}'
                by_name.setdefault(name, {})[p] = d
            hooks = []
            for name, by_pos in by_name.items():
                def hook(x, hook, by_pos=by_pos):
                    for p, d in by_pos.items():
                        x[0, p] = x[0, p] + d.to(x.dtype)
                    return x
                hooks.append((name, hook))
            return hooks

        def mlp_hooks(deltas):
            # deltas: {(layer, pos): change of the dictionary input}. The change is mapped to the residual stream and
            # shown only to the MLPs that the dictionary claims the feature writes into (layer and every later layer
            # for a CLT, its own layer for a PLT), at their normalized inputs; attention and the unembedding do not
            # see it, and the changed MLP outputs propagate freely.
            by_layer = {}
            for (l, p), d in deltas.items():
                scale = cache[f'blocks.{l}.ln2.hook_scale'][0, p].float()
                if D.input_hook == 'hook_resid_mid':
                    dx = d
                elif D.input_hook == 'ln2.hook_normalized':
                    dx = d * scale
                else:
                    w_ln = model.blocks[l].ln2.w.float()
                    dx = d * scale * torch.where(w_ln != 0, 1 / w_ln, torch.zeros_like(w_ln))
                for k in (range(l, L) if args.role == 'clt' else [l]):
                    s_k = cache[f'blocks.{k}.ln2.hook_scale'][0, p].float()
                    by_layer.setdefault(k, {})
                    by_layer[k][p] = by_layer[k].get(p, 0) + dx / s_k
            hooks = []
            for k, by_pos in by_layer.items():
                def hook(x, hook, by_pos=by_pos):
                    for p, d in by_pos.items():
                        x[0, p] = x[0, p] + d.to(x.dtype)
                    return x
                hooks.append((f'blocks.{k}.ln2.hook_normalized', hook))
            return hooks

        def run(hooks, mlp_pos=None):
            mlp = {}
            def grab(x, hook):
                mlp[int(hook.name.split('.')[1])] = x[0, mlp_pos].float().clone()
                return x
            caps = [(f'blocks.{k}.hook_mlp_out', grab) for k in range(L)] if mlp_pos is not None else []
            with torch.no_grad():
                logits = model.run_with_hooks(ids, fwd_hooks=hooks + caps)[0, -1].float()
            return float(logits[answer] - logits.mean()), mlp

        base, _ = run([])
        if args.role == 'clt':
            # Unit test of every pool unit: bounded share of all its claimed writes that removing its input takes out.
            clean = {}
            for u in pool:
                l, p, f = u['layer'], u['pos'], u['feature']
                x = cache[f'blocks.{l}.{D.input_hook}'][0, p].float()
                Wf, bf = encoder(l, [f])
                d = -Wf.T @ torch.linalg.solve(Wf @ Wf.T + 1e-6 * torch.eye(1, device=dev), Wf @ x + bf)
                if p not in clean:
                    clean[p] = run([], p)[1]
                mlp0 = clean[p]
                _, mlp1 = run(edit_hooks({(l, p): d}), p)
                dec = D.decoders(l, [f])[0]
                num, den = 0., 0.
                for k in range(l, l + dec.shape[0]):
                    c = u['act'] * dec[k - l]
                    cn = float(c.norm())
                    pi = float(-((mlp1[k] - mlp0[k]) @ c)) / max(cn, 1e-12)
                    num += (cn - pi) ** 2
                    den += cn ** 2
                u['s_all'] = max(0., 1 - num / den) if den > 0 else 0.
                u['passes'] = u['s_all'] >= args.tau
            failing = [k for k, u in enumerate(pool) if not u['passes']]
            dropped = set(rng.sample(range(len(pool)), len(failing)))
            lists = {'all': pool[:args.top],
                     'test': [u for u in pool if u['passes']][:args.top],
                     'random': [u for k, u in enumerate(pool) if k not in dropped][:args.top],
                     'layer': [u for u in pool if u['layer'] >= 2][:args.top]}
        else:
            lists = {'plt': pool[:args.top]}

        results = {}
        for name, units in lists.items():
            m = len(units)
            subsets = [list(range(k)) for k in (1, 2, 3, 5, 10, 20) if k <= m]
            while len(subsets) < args.subsets and m > 0:
                subsets.append(sorted(rng.sample(range(m), rng.randint(1, min(10, m)))))
            out = []
            for s in subsets:
                sites = {}
                for k in s:
                    sites.setdefault((units[k]['layer'], units[k]['pos']), []).append(units[k])
                rec = dict(units=[units[k]['node'] for k in s])
                for kind in ('rs', 'rm'):
                    deltas, pred = {}, 0.
                    for (l, p), us in sites.items():
                        x = cache[f'blocks.{l}.{D.input_hook}'][0, p].float()
                        Ws, bs = encoder(l, [u['feature'] for u in us])
                        if kind == 'rs':
                            donors = [v for gg, v in bank[l] if gg != g]
                            tgt = torch.stack([Ws[j] @ donors[rng.randrange(len(donors))].to(dev) for j in range(len(us))])
                            d = torch.linalg.pinv(Ws) @ (tgt - Ws @ x)
                        else:
                            d = -torch.linalg.pinv(Ws) @ (Ws @ x + bs)
                        a1 = D.encode((x + d)[None], l, [u['feature'] for u in us]).float().reshape(-1)
                        pred += sum((float(a1[j]) - u['act']) * u['tau'] for j, u in enumerate(us))
                        deltas[(l, p)] = d
                    val, _ = run(mlp_hooks(deltas))
                    rec[f'actual_{kind}'] = val - base
                    rec[f'pred_{kind}'] = pred
                out.append(rec)
            results[name] = dict(units=[{k: v for k, v in u.items()} for u in units], subsets=out)
        rows.append(dict(graph=g, prompt=prompt, answer=model.tokenizer.decode(answer), base=base, n_features=n_f,
                         pool=pool, lists=results))
        print(json.dumps(dict(elapsed=round(time.time() - t0, 1), graph=g, done=gi + 1, of=len(prompts),
                              sizes={k: len(v) for k, v in lists.items()}, n_features=n_f, **memory())), flush=True)
        (args.out / 'FILTER_PREDICTION_ROWS.partial.json').write_text(json.dumps(dict(rows=rows)))
        (args.out / 'progress.json').write_text(json.dumps(dict(stage=f'prompt {gi + 1}/{len(prompts)}')))
    (args.out / 'FILTER_PREDICTION_ROWS.json').write_text(json.dumps(dict(
        written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, model=args.model,
        dictionary=args.dictionary, role=args.role, rows=rows), indent=1))
    status.update(status='PASS', ended_at_utc=datetime.now(timezone.utc).isoformat())
    (args.out / 'status.json').write_text(json.dumps(status, indent=1))
    print(json.dumps(dict(stage='done', rows=len(rows))), flush=True)


if __name__ == '__main__':
    main()
