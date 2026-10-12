import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from common import Dictionary, replacement_model

DESCRIPTION = ('Correspondence between two models\' per-layer transcoders on unlabeled paired text, for porting annotated '
               'circuit roles. Each text is tokenized by both models; positions where a token of each model ends at the '
               'same character are aligned. For every source feature of the listed roles (supernodes of published '
               'source graphs), the Pearson correlation with every target feature of every layer is accumulated over '
               'the aligned positions, and the --top best-correlated target features are stored.')


def roles_from(registry, specs):
    # specs: 'graph=name|name' -> {graph: [(layer, feature), ...]} from the published supernodes.
    out = {}
    graphs = json.loads(registry.read_text())['graphs']
    for spec in specs:
        g, names = spec.split('=')
        sn = graphs[g]['variants'][0]['supernodes']
        feats = sorted({(l, f) for name in names.split('|') for l, _, f in sn[name]})
        out[g] = feats
    return out


def aligned(tok_a, tok_b, text, max_tokens):
    # Token positions of the two tokenizations whose tokens end at the same character, within the first max_tokens
    # tokens of the first tokenizer; position 0 (BOS) is excluded.
    ea = tok_a(text, return_offsets_mapping=True, truncation=True, max_length=max_tokens)
    cut = ea['offset_mapping'][-1][1]
    eb = tok_b(text[:cut], return_offsets_mapping=True)
    ends_b = {e: j for j, (s, e) in enumerate(eb['offset_mapping']) if e > s}
    pairs = [(i, ends_b[e]) for i, (s, e) in enumerate(ea['offset_mapping']) if e > s and e in ends_b]
    return ea['input_ids'], eb['input_ids'], pairs


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--source-model', default='google/gemma-2-2b')
    parser.add_argument('--source-dict', default='gemma')
    parser.add_argument('--target-model', default='google/gemma-3-1b-pt')
    parser.add_argument('--target-dict', default='mwhanna/gemma-scope-2-1b-pt/transcoder_all/width_16k_l0_small_affine')
    parser.add_argument('--registry', type=Path, required=True, help='PUBLISHED_EXPLANATIONS.json')
    parser.add_argument('--roles', nargs='+', required=True, help='graph=supernode|supernode')
    parser.add_argument('--text', type=Path, required=True, help='natural_fineweb_slice.json (train documents)')
    parser.add_argument('--docs', type=int, default=4096)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--top', type=int, default=16)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'config.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    t0 = time.time()
    dev = torch.device('cuda')
    src = replacement_model(args.source_model, args.source_dict, dev, torch.bfloat16)
    tgt = replacement_model(args.target_model, args.target_dict, dev, torch.bfloat16)
    for m in (src, tgt):
        m.requires_grad_(False)
    S, T = Dictionary(src.transcoders), Dictionary(tgt.transcoders)
    roles = roles_from(args.registry, args.roles)
    feats = sorted({x for v in roles.values() for x in v})
    src_layers = sorted({l for l, _ in feats})
    n_src, L_t = len(feats), tgt.cfg.n_layers
    width = T.encode(torch.zeros(1, tgt.cfg.d_model, device=dev), 0).shape[-1]
    n = torch.zeros((), dtype=torch.float64, device=dev)
    s1 = torch.zeros(n_src, dtype=torch.float64, device=dev)
    s2 = torch.zeros(n_src, dtype=torch.float64, device=dev)
    t1 = torch.zeros(L_t, width, dtype=torch.float64, device=dev)
    t2 = torch.zeros(L_t, width, dtype=torch.float64, device=dev)
    st = torch.zeros(L_t, n_src, width, dtype=torch.float64, device=dev)
    s_active = torch.zeros(n_src, dtype=torch.float64, device=dev)
    docs = json.loads(args.text.read_text())['train'][:args.docs]
    names_s = [f'blocks.{l}.{S.input_hook}' for l in src_layers]
    names_t = [f'blocks.{l}.{T.input_hook}' for l in range(L_t)]
    for d, text in enumerate(docs):
        ids_a, ids_b, pairs = aligned(src.tokenizer, tgt.tokenizer, text, args.max_tokens)
        if len(pairs) < 2:
            continue
        ia = torch.tensor([i for i, _ in pairs], device=dev)
        ib = torch.tensor([j for _, j in pairs], device=dev)
        with torch.no_grad():
            _, ca = src.run_with_cache(torch.tensor([ids_a], device=dev), names_filter=lambda x: x in names_s)
            _, cb = tgt.run_with_cache(torch.tensor([ids_b], device=dev), names_filter=lambda x: x in names_t)
            sa = torch.stack([S.encode(ca[f'blocks.{l}.{S.input_hook}'][0, ia], l, [f]).reshape(-1)
                              for l, f in feats], 1).double()
            n += len(pairs)
            s1 += sa.sum(0)
            s2 += (sa ** 2).sum(0)
            s_active += (sa > 0).double().sum(0)
            for l in range(L_t):
                z = T.encode(cb[f'blocks.{l}.{T.input_hook}'][0, ib], l).double()
                t1[l] += z.sum(0)
                t2[l] += (z ** 2).sum(0)
                st[l] += sa.T @ z
        if d % 200 == 0:
            print(json.dumps(dict(elapsed=round(time.time() - t0, 1), docs=d + 1, aligned=int(n))), flush=True)
    cov = st / n - (s1 / n)[None, :, None] * (t1 / n)[:, None, :]
    var_s = s2 / n - (s1 / n) ** 2
    var_t = t2 / n - (t1 / n) ** 2
    corr = cov / torch.sqrt(var_s[None, :, None] * var_t[:, None, :]).clamp_min(1e-12)
    corr = torch.where(var_t[:, None, :] > 0, corr, torch.zeros_like(corr))
    flat = corr.permute(1, 0, 2).reshape(n_src, -1)
    vals, idx = flat.topk(args.top, dim=1)
    out = []
    for k, (l, f) in enumerate(feats):
        out.append(dict(layer=l, feature=f, active_share=float(s_active[k] / n),
                        top=[dict(layer=int(i) // width, feature=int(i) % width, corr=float(v))
                             for v, i in zip(vals[k].tolist(), idx[k].tolist())]))
    (args.out / 'PORT_CORR.json').write_text(json.dumps(dict(
        written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, aligned=int(n),
        roles={g: [list(x) for x in v] for g, v in roles.items()}, width=width, source_features=out), indent=1))
    print(json.dumps(dict(stage='done', aligned=int(n), source_features=n_src)), flush=True)


if __name__ == '__main__':
    main()
