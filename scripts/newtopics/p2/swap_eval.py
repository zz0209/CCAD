import argparse
import json
import time
from pathlib import Path

import jlens
import torch
import torch.nn.functional as F
from jlens.hooks import ActivationRecorder

from common import band_layers, cap_gpu_memory, first_token, load_model, readout_vectors, write_json

DESCRIPTION = ('Two-hop bridge-entity swap for one checkpoint. On every two-hop prompt the checkpoint answers '
               'correctly, the lens coordinates of the bridge country e and the swap target e2 are exchanged at the '
               'output of every band layer and every position after BOS (h <- h + V(swap(c) - c), c = pinv(V) h, '
               'alpha 1), in J-lens, logit-lens and tuned-lens coordinates. The logit_nm, tuned_nm and tmatch_nm '
               'variants keep their own swap direction and take, at each layer and position, the norm of the J-swap '
               'perturbation on the current activation. Concept and random controls add, at each layer and position, '
               'a perturbation whose norm equals the J-swap perturbation on the clean activation, along the '
               'diff-in-means direction m(e2) - m(e), along the part of that direction outside the J-space '
               '(non-negative gradient pursuit, k=16, over all J-lens vectors), or along a random direction. Every '
               'intervention records its mean perturbation norm. tmatch uses a second tuned lens (--tuned_match). '
               'Concept vectors come from the separate concept-fitting prompt set (--concept_fit): last-position '
               'activations centred by the mean of their relation pair, averaged per country; they are scored on '
               'held-out relation pairs only. dim_outside_j_clamp also holds the J-lens coordinates of e and e2 at '
               'their clean-pass values at every band layer and position. '
               'Success: an answer token of e2 enters the top 5 and the original answer token leaves the top 1.')
METHODS = ['jlens', 'logit', 'tuned', 'tmatch', 'dim', 'dim_outside_j', 'dim_outside_j_clamp', 'random']
# Lens swaps rescaled at each band layer and position to the norm of the J-swap perturbation on the current activation.
LENS_NM = ['logit_nm', 'tuned_nm', 'tmatch_nm']
CONCEPT = {'dim', 'dim_outside_j', 'dim_outside_j_clamp'}
K_PURSUIT = 16
CHUNK = 8192


def gradient_pursuit(D, atom, y, k):
    # Non-negative gradient pursuit (Blumensath & Davies 2008): greedy atom selection by positive correlation with
    # the residual, then one gradient step on the support coefficients, projected to x >= 0.
    support, x, r = [], torch.zeros(0, device=y.device), y.clone()
    for _ in range(k):
        corr = (D @ r.to(D.dtype)).float()
        if support:
            corr[torch.tensor(support, device=corr.device)] = -float('inf')
        i = int(corr.argmax())
        if corr[i] <= 0:
            break
        support.append(i)
        A = atom(support)
        x = torch.cat([x, torch.zeros(1, device=y.device)])
        g = A @ r
        Ag = g @ A
        x = (x + (g @ g) / (Ag @ Ag) * g).clamp(min=0)
        r = y - x @ A
    return y - r, r, support


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--jlens', required=True)
    parser.add_argument('--tuned', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=None, help='smoke test: first N items of each relation pair')
    parser.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS + LENS_NM)
    parser.add_argument('--tuned_match', default=None, help='second tuned lens for the tmatch methods')
    parser.add_argument('--concept_fit', default=None, help='concept-fitting prompt set (build_task.py --concept_fit)')
    parser.add_argument('--pairs', nargs='+', default=None, help='evaluate only these relation pairs')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu_cap_gib', type=float, default=18.0)
    args = parser.parse_args()
    if args.device == 'cuda':
        cap_gpu_memory(args.gpu_cap_gib)
    dev = args.device
    concept_methods = CONCEPT & set(args.methods)
    if concept_methods and args.concept_fit is None:
        raise ValueError('concept-vector methods need --concept_fit')
    if {'tmatch', 'tmatch_nm'} & set(args.methods) and args.tuned_match is None:
        raise ValueError('tmatch methods need --tuned_match')
    t0 = time.time()
    row, hf, tok, lm = load_model(args.manifest, args.label, dev)
    task = json.load(open(args.task))
    items = task['items']
    if args.pairs:
        items = [it for it in items if it['pair'] in args.pairs]
    if args.limit:
        items = [it for p in task['pairs'] for it in [i for i in items if i['pair'] == p][:args.limit]]
    band = band_layers(lm.n_layers)
    lens = jlens.JacobianLens.load(args.jlens)
    if not set(band) <= set(lens.source_layers):
        raise ValueError(f'J-lens layers {lens.source_layers} miss band {band}')
    tuned = torch.load(args.tuned, map_location='cpu', weights_only=True)
    if tuned['band'] != band:
        raise ValueError('tuned-lens band mismatch')
    tmatch = torch.load(args.tuned_match, map_location='cpu', weights_only=True) if args.tuned_match else None
    if tmatch is not None and tmatch['band'] != band:
        raise ValueError('matched tuned-lens band mismatch')
    final = lm.n_layers - 1
    n_vocab = min(len(tok), lm._lm_head.weight.shape[0])

    def single(name):
        ids = tok.encode(' ' + name, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f'{name} is not a single token')
        return ids[0]

    def run(prompt, fns=None, record=None, full=False):
        # Records band activations at the last position, or with full=True at every position after BOS.
        ids = lm.encode(prompt, max_length=512)
        if ids[0, 0].item() != tok.bos_token_id:
            raise ValueError('missing BOS')
        handles = []
        for l, fn in (fns or {}).items():
            def hook(mod, inp, out, fn=fn):
                h = out if torch.is_tensor(out) else out[0]
                h2 = h.clone()
                h2[:, 1:] = fn(h[:, 1:].float()).to(h.dtype)
                return h2 if torch.is_tensor(out) else (h2,) + tuple(out[1:])
            handles.append(lm.layers[l].register_forward_hook(hook))
        try:
            with torch.no_grad(), ActivationRecorder(lm.layers, at=(record or []) + [final]) as rec:
                lm.forward(ids)
                acts = {l: rec.activations[l][0, -1].float() for l in (record or []) + [final]}
                if full:
                    acts['full'] = {l: rec.activations[l][0, 1:].float() for l in record}
        finally:
            for hd in handles:
                hd.remove()
        return lm.unembed(acts[final]).float(), acts

    # Baseline: correctness of two-hop prompts and one-hop controls; last-position band activations.
    base = []
    for it in items:
        A = {first_token(tok, a) for a in it['answers']}
        A2 = {first_token(tok, a) for a in it['target_answers']}
        logits, acts = run(it['two_hop'], record=band)
        one, _ = run(it['one_hop'])
        tone, _ = run(it['target_one_hop'])
        top5 = logits.topk(5).indices.tolist()
        base.append(dict(id=it['id'], pair=it['pair'], held_out=it['held_out'], bridge=it['bridge'],
                         target=it['target'], correct=top5[0] in A, top5=top5,
                         one_hop_correct=one.argmax().item() in A, target_one_hop_correct=tone.argmax().item() in A2,
                         acts=torch.stack([acts[l] for l in band])))
    print(f'baseline done {time.time() - t0:.0f}s: {sum(r["correct"] for r in base)}/{len(base)} correct', flush=True)

    # Concept sources: last-position band activations of the concept-fitting prompts, centred by the mean of their
    # relation pair, so that a country's mean does not depend on which pairs it appears in.
    src = []
    if concept_methods:
        fit = json.load(open(args.concept_fit))
        raw = []
        for f in fit['items']:
            _, a = run(f['prompt'], record=band)
            raw.append((f['country'], f['pair'], torch.stack([a[l] for l in band])))
        pair_mean = {p: torch.stack([a for _, q, a in raw if q == p]).mean(0) for p in {q for _, q, _ in raw}}
        src = [(c, a - pair_mean[p]) for c, p, a in raw]
        print(f'concept-fitting prompts done {time.time() - t0:.0f}s: {len(src)}', flush=True)
    concept_countries = sorted({c for c, _ in src})

    # Per band layer: lens readout rank of the latent bridge at the last position, lens vectors of every country
    # token, and the J-space decomposition of the diff-in-means concept vectors.
    countries = sorted({it['bridge'] for it in items} | {it['target'] for it in items})
    ctok = torch.tensor([single(c) for c in countries], device=dev)
    u_c = readout_vectors(lm, ctok)
    acts_all = torch.stack([r['acts'] for r in base]).to(dev)
    vec = {m: {} for m in ('jlens', 'tuned', 'tmatch')}
    conc, outside, jfrac, top_atoms = ({e: {} for e in concept_countries} for _ in range(4))
    for li, l in enumerate(band):
        Jl = lens.jacobians[l].to(dev)
        W = tuned['translators'][l]['weight'].float().to(dev)
        b = tuned['translators'][l]['bias'].float().to(dev)
        H = acts_all[:, li]
        b_ids = torch.tensor([single(r['bridge']) for r in base], device=dev)
        for name, X in (('jlens', H @ Jl.T), ('logit', H), ('tuned', H + H @ W.T + b)):
            lg = lm.unembed(X).float()[:, :n_vocab]
            ranks = (lg > lg.gather(1, b_ids[:, None])).sum(1).tolist()
            for r, k in zip(base, ranks):
                r.setdefault('bridge_rank_by_layer', {}).setdefault(name, []).append(int(k))
        vec['jlens'][l] = dict(zip(countries, u_c @ Jl))
        vec['tuned'][l] = dict(zip(countries, u_c + u_c @ W))
        if tmatch is not None:
            Wm = tmatch['translators'][l]['weight'].float().to(dev)
            vec['tmatch'][l] = dict(zip(countries, u_c + u_c @ Wm))
            del Wm
        if concept_methods:
            D = torch.empty(n_vocab, lm.d_model, dtype=torch.bfloat16, device=dev)
            for s in range(0, n_vocab, CHUNK):
                sl = torch.arange(s, min(s + CHUNK, n_vocab), device=dev)
                D[s:s + CHUNK] = F.normalize(readout_vectors(lm, sl) @ Jl, dim=-1).to(torch.bfloat16)

            def atom(sup, Jl=Jl):
                return F.normalize(readout_vectors(lm, torch.tensor(sup, device=dev)) @ Jl, dim=-1)
            for e in concept_countries:
                ce = torch.stack([a[li] for c, a in src if c == e]).mean(0).to(dev)
                jp, rem, sup = gradient_pursuit(D, atom, ce, K_PURSUIT)
                conc[e][l], outside[e][l] = ce, rem
                jfrac[e][str(l)] = float(jp.norm() ** 2 / ce.norm() ** 2)
                top_atoms[e][str(l)] = [tok.decode([s]) for s in sup[:5]]
            del D
        del Jl, W, b
        if dev == 'cuda':
            torch.cuda.empty_cache()
    for r in base:
        r['bridge_rank_min'] = {k: min(v) for k, v in r['bridge_rank_by_layer'].items()}
    print(f'lens readouts and concept vectors done {time.time() - t0:.0f}s', flush=True)

    # Swaps on correctly answered prompts.
    rows = []
    for idx, (it, br) in enumerate(zip(items, base)):
        r = {k: v for k, v in br.items() if k != 'acts'}
        rows.append(r)
        if not br['correct']:
            continue
        A = {first_token(tok, a) for a in it['answers']}
        A2 = {first_token(tok, a) for a in it['target_answers']}
        u = readout_vectors(lm, torch.tensor([single(it['bridge']), single(it['target'])], device=dev))
        V = {'jlens': {l: torch.stack([vec['jlens'][l][it['bridge']], vec['jlens'][l][it['target']]], 1)
                       for l in band},
             'logit': {l: u.T for l in band},
             'tuned': {l: torch.stack([vec['tuned'][l][it['bridge']], vec['tuned'][l][it['target']]], 1)
                       for l in band}}
        if tmatch is not None:
            V['tmatch'] = {l: torch.stack([vec['tmatch'][l][it['bridge']], vec['tmatch'][l][it['target']]], 1)
                           for l in band}
        P = {m: {l: torch.linalg.pinv(V[m][l]) for l in band} for m in V}

        def delta(m, l, h):
            c = h @ P[m][l].T
            return (c.flip(-1) - c) @ V[m][l].T

        # Each intervention adds a perturbation and records its mean norm over band layers and patched positions.
        def swap(m, l, norms):
            def fn(h):
                d = delta(m, l, h)
                norms.append(d.norm(dim=-1).mean().item())
                return h + d
            return fn

        def swap_nm(m, l, norms):
            def fn(h):
                d = delta(m, l, h)
                n_j = delta('jlens', l, h).norm(dim=-1, keepdim=True)
                n_m = d.norm(dim=-1, keepdim=True)
                d = torch.where(n_m > 0, d * n_j / n_m.clamp_min(torch.finfo(d.dtype).tiny), torch.zeros_like(d))
                norms.append(d.norm(dim=-1).mean().item())
                return h + d
            return fn

        # Concept and random controls take, at each band layer and position, the norm of the J-swap perturbation on
        # the clean (unpatched) activation, so a direction that moves the e/e2 coordinates cannot enlarge its own norm
        # at later layers. The clamp resets the J-lens coordinates of e and e2 to their clean-pass values.
        if CONCEPT & set(args.methods) or 'random' in args.methods:
            _, clean = run(it['two_hop'], record=band, full=True)
            clean_n = {l: delta('jlens', l, clean['full'][l]).norm(dim=-1, keepdim=True) for l in band}
            clean_c = {l: clean['full'][l] @ P['jlens'][l].T for l in band}

        def matched(direction, l, norms, clamp=False):
            dvec = F.normalize(direction, dim=0)

            def fn(h):
                norms.append(clean_n[l].mean().item())
                h = h + clean_n[l] * dvec
                if clamp:
                    h = h + (clean_c[l] - h @ P['jlens'][l].T) @ V['jlens'][l].T
                return h
            return fn

        # Concept directions are scored on held-out relation pairs only.
        dirs = {}
        if it['held_out'] and it['bridge'] in conc and it['target'] in conc:
            dirs['dim'] = {l: conc[it['target']][l] - conc[it['bridge']][l] for l in band}
            dirs['dim_outside_j'] = {l: outside[it['target']][l] - outside[it['bridge']][l] for l in band}
            dirs['dim_outside_j_clamp'] = dirs['dim_outside_j']
        gen = torch.Generator().manual_seed(idx)
        dirs['random'] = {l: torch.randn(lm.d_model, generator=gen).to(dev) for l in band}
        res = {}
        for m in args.methods:
            norms = []
            if m in V:
                fns = {l: swap(m, l, norms) for l in band}
            elif m in LENS_NM:
                fns = {l: swap_nm(m[:-3], l, norms) for l in band}
            elif m in dirs:
                fns = {l: matched(dirs[m][l], l, norms, clamp=m == 'dim_outside_j_clamp') for l in band}
            else:
                res[m] = None
                continue
            logits, _ = run(it['two_hop'], fns=fns)
            top5 = logits.topk(5).indices.tolist()
            res[m] = dict(success=bool(set(top5) & A2) and top5[0] not in A, top5=top5,
                          new_rank=min(int((logits > logits[a]).sum()) for a in A2),
                          pert_norm=sum(norms) / len(norms))
        r['swap'] = res
    out = dict(label=args.label, revision=row['revision'], commit=row['commit'], band=band, n_items=len(items),
               k_pursuit=K_PURSUIT, held_out=task['held_out'], methods=args.methods, jlens=str(args.jlens),
               tuned=str(args.tuned), tuned_match=args.tuned_match, task=str(args.task), device=dev, rows=rows,
               concept=dict(fit_set=args.concept_fit, n_fit_prompts=len(src), countries=concept_countries,
                            jspace_energy_frac=jfrac, top_atoms=top_atoms),
               seconds=time.time() - t0)
    write_json(args.out, out)
    ok = [r for r in rows if r['correct']]
    print('SWAP_DONE', args.label, f'{len(ok)} correct;',
          {m: sum(bool(r['swap'][m] and r['swap'][m]['success']) for r in ok) for m in args.methods},
          f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
