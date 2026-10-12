import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import torch

from circuit_tracer import attribute
from circuit_tracer.graph import prune_graph
from common import Dictionary, add_hooks, gemma_items, filter_items, load_target, replacement_model
from oracle_sites import box_least_squares

DESCRIPTION = ('Material for reading published Gemma-2-2B explanations side by side with the CLT\'s own explanation of '
               'the same prompt: each published supernode and each of the CLT graph\'s most influential feature nodes '
               'with its effect on the model\'s answer, the features of the other dictionary that best reconstruct its '
               'write at each layer, and Neuronpedia explanations for every feature involved (gemmascope-transcoder-16k '
               'for the PLT, clt-hp for the 426k CLT, whose correspondence is checked on Neuronpedia\'s top activations).')


def neuronpedia(source_layer, index, cache):
    key = f'{source_layer}/{index}'
    if key not in cache:
        r = requests.get(f'https://www.neuronpedia.org/api/feature/gemma-2-2b/{source_layer}/{index}', timeout=30)
        r.raise_for_status()
        body = r.json()
        acts = body.get('activations') or []
        top = acts[0] if acts else None
        cache[key] = dict(explanations=[e.get('description') for e in (body.get('explanations') or [])][:3],
                          top_tokens=top.get('tokens') if top else None, top_values=top.get('values') if top else None,
                          max_value=top.get('maxValue') if top else None)
        time.sleep(.2)
    return cache[key]


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--prompts-from', type=Path, required=True, help='ORACLE_ROWS.json with the chosen prompts')
    parser.add_argument('--graphs', nargs='+', required=True)
    parser.add_argument('--clt', default='mntss/clt-gemma-2-2b-426k')
    parser.add_argument('--units', type=int, default=20)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'config.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    t0 = time.time()

    def log(**kw):
        print(json.dumps(dict(elapsed=round(time.time() - t0, 1), **kw)), flush=True)
    dev = torch.device('cuda')
    model = replacement_model('google/gemma-2-2b', args.clt, dev, torch.bfloat16)
    model.requires_grad_(False)
    C = Dictionary(model.transcoders)
    P = Dictionary(load_target('gemma', dev, torch.bfloat16))
    C.t.requires_grad_(False)
    P.t.requires_grad_(False)
    L = model.cfg.n_layers
    items, _ = filter_items(gemma_items(args.registry), L, P.t.d_transcoder)
    published = {k: sn for k, _, sn in items}
    prompts = {}
    for r in json.loads(args.prompts_from.read_text())['rows']:
        prompts.setdefault(r['graph'], r['prompt'])
    names = [f'blocks.{l}.{C.input_hook}' for l in range(L)] + [f'blocks.{l}.{P.input_hook}' for l in range(L)]
    labels = {}
    tok = model.tokenizer

    def run(ids, deltas):
        with torch.no_grad():
            logits = model.run_with_hooks(ids, fwd_hooks=add_hooks(deltas))
        return torch.log_softmax(logits[0, -1].float(), -1)

    def effect(ids, base, deltas, answer):
        lp = run(ids, deltas)
        d = (lp - base)
        up = d.topk(3).indices.tolist()
        return dict(answer_drop=float(base[answer] - lp[answer]), new_top=tok.decode(int(lp.argmax())),
                    raised=[tok.decode(int(i)) for i in up])

    def write_of(D, l, p, feats, cache, n):
        # The deletion writes of features (read layer l, position p) of dictionary D, as {output layer: (n, d)}.
        out = {}
        z = D.encode(cache[f'blocks.{l}.{D.input_hook}'][0, p], l, feats).float()
        dec = D.decoders(l, feats)
        for k in range(dec.shape[1]):
            out.setdefault(l + k, torch.zeros(n, dec.shape[2], device=dev))[p] -= (z[:, None] * dec[:, k]).sum(0)
        return out

    def best_reconstruction(D, cross, z_by_layer, target, l, p, top=4):
        # Features of D active at p whose writes into layer l best reconstruct the target write (box constrained).
        blocks = []
        for lb in (range(l + 1) if cross else [l]):
            act = (z_by_layer[lb][p] > 0).nonzero().flatten()
            if len(act):
                blocks.append((lb, act, D.decoders(lb, act.tolist())[:, l - lb], z_by_layer[lb][p, act]))
        Dm = torch.cat([b[2] for b in blocks])
        z = torch.cat([b[3] for b in blocks])
        ids = [(lb, int(a)) for lb, act, _, _ in blocks for a in act]
        c = box_least_squares(Dm, target, z)
        weight = c * Dm.norm(dim=1)
        resid = float((target - c @ Dm).norm() / target.norm().clamp_min(1e-9))
        order = weight.argsort(descending=True)[:top].tolist()
        share = float(weight[order].sum() / weight.sum().clamp_min(1e-12))
        return resid, [dict(layer=ids[i][0], index=ids[i][1], weight_share=float(weight[i] / weight.sum().clamp_min(1e-12)))
                       for i in order], share

    readings = []
    for key in args.graphs:
        prompt = prompts[key]
        ids = tok(prompt, return_tensors='pt').input_ids.to(dev)
        n = ids.shape[1]
        toks = [tok.decode(int(t)) for t in ids[0]]
        with torch.no_grad():
            _, cache = model.run_with_cache(ids, names_filter=lambda n_: n_ in names)
            zc = {l: C.encode(cache[f'blocks.{l}.{C.input_hook}'][0], l).float() for l in range(L)}
            zp = {l: P.encode(cache[f'blocks.{l}.{P.input_hook}'][0], l).float() for l in range(L)}
        base = run(ids, {})
        answer = int(base.argmax())
        entry = dict(graph=key, prompt=prompt, tokens=toks, answer=tok.decode(answer), supernodes=[], clt_units=[])
        # Published supernodes in the PLT and the CLT features that carry their write at each site.
        for sname, fs in published[key].items():
            fs = [x for x in fs if x[1] < n]
            deltas = {}
            sites = []
            for l, p in sorted({(l, p) for l, p, _ in fs}):
                feats = [f for ll, pp, f in fs if ll == l and pp == p]
                w = write_of(P, l, p, feats, cache, n)
                for k, v in w.items():
                    deltas[k] = deltas.get(k, 0.) + v
                target = -w[l][p]
                if float(target.norm()) == 0:
                    continue
                resid, top, share = best_reconstruction(C, True, zc, target, l, p)
                for t in top:
                    t['label'] = neuronpedia(f"{t['layer']}-clt-hp", t['index'], labels)['explanations']
                sites.append(dict(layer=l, position=p, token=toks[p],
                                  features=[dict(index=f, label=neuronpedia(f'{l}-gemmascope-transcoder-16k', f, labels)['explanations'])
                                            for f in feats],
                                  clt_residual=resid, clt_top=top, clt_top_share=share))
            entry['supernodes'].append(dict(name=sname, sites=sites, **effect(ids, base, deltas, answer)))
        # The CLT's own graph on the same prompt and the PLT features that carry each unit's write into its own layer.
        graph = attribute(prompt, model, max_n_logits=10, desired_logit_prob=0.95, batch_size=128)
        scores = prune_graph(graph).cumulative_scores
        order = torch.argsort(scores[:len(graph.selected_features)])[:args.units]
        for rank, i in enumerate(order.tolist()):
            l, p, f = (int(v) for v in graph.active_features[graph.selected_features[i]])
            w = write_of(C, l, p, [f], cache, n)
            target = -w[l][p]
            resid, top, share = best_reconstruction(P, False, zp, target, l, p)
            for t in top:
                t['label'] = neuronpedia(f"{t['layer']}-gemmascope-transcoder-16k", t['index'], labels)['explanations']
            later = sum(float((w[k][p] ** 2).sum()) for k in w if k > l)
            entry['clt_units'].append(dict(rank=rank + 1, layer=l, position=p, token=toks[p], index=f,
                                           activation=float(zc[l][p, f]),
                                           label=neuronpedia(f'{l}-clt-hp', f, labels)['explanations'],
                                           later_write_share=later / max(later + float((w[l][p] ** 2).sum()), 1e-12),
                                           plt_residual=resid, plt_top=top, plt_top_share=share,
                                           **effect(ids, base, w, answer)))
        readings.append(entry)
        log(stage='graph', graph=key, supernodes=len(entry['supernodes']), units=len(entry['clt_units']))
    # The clt-hp source on Neuronpedia should be this CLT: compare its recorded maximum activation with ours on the same
    # text for a few of the units read above.
    checks = []
    for entry in readings[:2]:
        for u in entry['clt_units'][:3]:
            info = neuronpedia(f"{u['layer']}-clt-hp", u['index'], labels)
            if not info['top_tokens']:
                continue
            text = ''.join(info['top_tokens'])
            ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.to(dev)
            with torch.no_grad():
                _, cache = model.run_with_cache(ids, names_filter=lambda n_: n_ == f"blocks.{u['layer']}.{C.input_hook}")
                ours = float(C.encode(cache[f"blocks.{u['layer']}.{C.input_hook}"][0], u['layer'], [u['index']]).float().max())
            checks.append(dict(layer=u['layer'], index=u['index'], neuronpedia_max=info['max_value'], ours_max=ours))
    (args.out / 'READINGS.json').write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(),
                                                            description=DESCRIPTION, clt_hp_check=checks,
                                                            readings=readings), indent=1, ensure_ascii=False))
    log(stage='done', checks=checks)


if __name__ == '__main__':
    main()
