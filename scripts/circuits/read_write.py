import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from common import Dictionary, add_hooks, choose_prompt, filter_items, gemma_items, load_target, replacement_model

DESCRIPTION = ('Read-write agreement of explanation units, with the model as referee. For each unit (features at '
               'layer l and position p of one dictionary) two interventions are compared on the final-position '
               'next-token distribution: deleting what the unit writes (its decoder writes into every layer it writes '
               'to, the semantics under which attribution graphs are validated) and removing what it reads (the '
               'smallest change of its input at its own hook, layer and position that switches every member feature '
               'off, with the rest of the model running as usual). A decomposition that is faithful about a unit gives '
               'similar effects. Units are published supernodes, or single-feature units from a registry or from the '
               'converse rows of a CLT graph.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--model', default='google/gemma-2-2b')
    parser.add_argument('--dictionary', required=True, help='dictionary whose units are tested')
    parser.add_argument('--registry', type=Path, required=True, help='registry with supernodes (published or own-graph units)')
    parser.add_argument('--registry-model', default='gemma-2-2b')
    parser.add_argument('--units-from', type=Path, default=None, help='REVERSE_ROWS.json of a CLT graph: its single-feature units')
    parser.add_argument('--prompts-from', type=Path, required=True, help='rows with the chosen prompt per graph')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'config.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    t0 = time.time()
    dev = torch.device('cuda')
    model = replacement_model(args.model, args.dictionary, dev, torch.bfloat16)
    model.requires_grad_(False)
    D = Dictionary(model.transcoders)
    D.t.requires_grad_(False)
    L = model.cfg.n_layers
    prompts = {}
    for r in json.loads(args.prompts_from.read_text())['rows']:
        prompts.setdefault(r['graph'], r['prompt'])
    if args.units_from is None:
        items, _ = filter_items(gemma_items(args.registry, args.registry_model), L, D.t.d_transcoder)
        units = {k: sn for k, _, sn in items}
    else:
        units = {}
        for r in json.loads(args.units_from.read_text())['rows']:
            if r['features'] == 1:
                l, rest = r['unit'][1:].split('F')
                f, p = rest.split('P')
                units.setdefault(r['graph'], {})[r['unit']] = [(int(l), int(p), int(f))]

    def encoder(l, feats):
        # Encoder rows (n, d_model) and biases (n,) of features at layer l.
        if hasattr(D.t, 'transcoders'):
            tc = D.t.transcoders[l]
            return tc.W_enc[feats].float(), tc.b_enc[feats].float()
        return D.t.W_enc[l][feats].float(), D.t.b_enc[l][feats].float()

    def run(ids, hooks):
        with torch.no_grad():
            logits = model.run_with_hooks(ids, fwd_hooks=hooks)
        return torch.log_softmax(logits[0, -1].float(), -1)

    rows = []
    for key, sn in units.items():
        if key not in prompts:
            continue
        prompt = prompts[key]
        ids = model.tokenizer(prompt, return_tensors='pt').input_ids.to(dev)
        n = ids.shape[1]
        names = [f'blocks.{l}.{D.input_hook}' for l in range(L)]
        with torch.no_grad():
            _, cache = model.run_with_cache(ids, names_filter=lambda n_: n_ in names)
        base = run(ids, [])
        answer = int(base.argmax())
        w0 = base.exp()
        for uname, fs in sn.items():
            fs = [(l, p, f) for l, p, f in fs if p < n]
            if not fs:
                continue
            writes, reads = {}, {}
            for l, p in sorted({(l, p) for l, p, _ in fs}):
                feats = [f for ll, pp, f in fs if ll == l and pp == p]
                x = cache[f'blocks.{l}.{D.input_hook}'][0, p].float()
                # Some CLT encoders keep a leading batch axis for a single position.
                z = D.encode(cache[f'blocks.{l}.{D.input_hook}'][0, p], l, feats).float().reshape(len(feats))
                dec = D.decoders(l, feats)
                for k in range(dec.shape[1]):
                    writes.setdefault(l + k, torch.zeros(n, dec.shape[2], device=dev))[p] -= (z[:, None] * dec[:, k]).sum(0)
                # Smallest input change that sets every member's pre-activation to zero at this site.
                W, b = encoder(l, feats)
                pre = W @ x + b
                delta = -W.T @ torch.linalg.solve(W @ W.T + 1e-6 * torch.eye(len(feats), device=dev), pre)
                reads.setdefault(l, {})[p] = delta
            read_hooks = []
            for l, by_pos in reads.items():
                def hook(act, hook, by_pos=by_pos):
                    for p, delta in by_pos.items():
                        act[0, p] = act[0, p] + delta.to(act.dtype)
                    return act
                read_hooks.append((f'blocks.{l}.{D.input_hook}', hook))
            ew = run(ids, add_hooks(writes)) - base
            er = run(ids, read_hooks) - base
            energy_w = float((w0 * ew ** 2).sum())
            energy_r = float((w0 * er ** 2).sum())
            top = lambda e: model.tokenizer.decode(int((base + e).argmax()))
            rows.append(dict(graph=key, unit=uname, features=len(fs), layers=sorted({l for l, _, _ in fs}),
                             positions=sorted({p for _, p, _ in fs}), drop_write=float(-ew[answer]),
                             drop_read=float(-er[answer]), energy_write=energy_w, energy_read=energy_r,
                             read_vs_write=float((w0 * (er - ew) ** 2).sum() / max(energy_w, 1e-12)) ** .5,
                             top_base=top(torch.zeros_like(base)), top_write=top(ew), top_read=top(er)))
        print(json.dumps(dict(elapsed=round(time.time() - t0, 1), graph=key, units=len(sn))), flush=True)
    (args.out / 'READ_WRITE_ROWS.json').write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(),
                                                                   description=DESCRIPTION, dictionary=args.dictionary,
                                                                   rows=rows), indent=1))
    print(json.dumps(dict(stage='done', rows=len(rows))), flush=True)


if __name__ == '__main__':
    main()
