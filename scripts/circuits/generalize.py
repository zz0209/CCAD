import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from circuit_tracer import ReplacementModel
from common import Dictionary, add_hooks, filter_items, gemma_items

DESCRIPTION = ('CIRCUITS_ACROSS_DICTIONARIES P2 (H-inv): does the causal role of a published supernode carry over to '
               'concept-preserving variants of its prompt? Each supernode is deleted at its published positions '
               'through the dictionary that defined it, on the published prompt and on every variant of equal token '
               'length; the readout is the drop of the log-probability of the answer token (first token of the '
               'answer the explanation predicts), and retention is the drop on a variant divided by the drop on the '
               'published prompt. Feature activity on the variant is recorded alongside.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--families', type=Path, required=True)
    parser.add_argument('--prompts-from', type=Path, required=True, help='ORACLE_ROWS.json with the chosen prompts')
    parser.add_argument('--source', default='gemma')
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
    dev = torch.device('cuda')
    model = ReplacementModel.from_pretrained('google/gemma-2-2b', args.source, dtype=torch.bfloat16, backend='transformerlens')
    model.requires_grad_(False)
    A = Dictionary(model.transcoders)
    A.t.requires_grad_(False)
    L = model.cfg.n_layers
    items, _ = filter_items(gemma_items(args.registry), L, A.t.d_transcoder)
    items = {k: sn for k, _, sn in items}
    chosen = {}
    for r in json.loads(args.prompts_from.read_text())['rows']:
        chosen.setdefault(r['graph'], r['prompt'])
    families = json.loads(args.families.read_text())['families']
    names = [f'blocks.{l}.{A.input_hook}' for l in range(L)]
    tok = model.tokenizer

    def answer_id(text):
        return tok(text, add_special_tokens=False).input_ids[0]

    def drop(ids, sn_feats, answer):
        with torch.no_grad():
            _, cache = model.run_with_cache(ids, names_filter=lambda n: n in names)
            base = torch.log_softmax(model(ids)[0, -1].float(), -1)
            deltas, active = {}, []
            for l, p, f in sn_feats:
                z = A.encode(cache[f'blocks.{l}.{A.input_hook}'][0, p], l, [f]).float()
                active.append(float(z) > 0)
                deltas.setdefault(l, torch.zeros(ids.shape[1], model.cfg.d_model, device=dev))[p] -= z[0] * A.decoders(l, [f])[0, 0]
            edited = torch.log_softmax(model.run_with_hooks(ids, fwd_hooks=add_hooks(deltas))[0, -1].float(), -1)
        return float(base[answer] - edited[answer]), float(np.mean(active)), float(base[answer])

    rows = []
    for key, fam in families.items():
        if key not in items or key not in chosen:
            log(stage='skip', graph=key)
            continue
        prompt = chosen[key]
        ids0 = tok(prompt, return_tensors='pt').input_ids.to(dev)
        variants = []
        for fill in fam['fill']:
            text = fam['template'].format(fill)
            ids = tok(text, return_tensors='pt').input_ids.to(dev)
            if ids.shape[1] == ids0.shape[1] and text.strip() != prompt.strip():
                variants.append((text, ids))
        a0 = answer_id(fam['answer'])
        for sname, feats in items[key].items():
            feats = [(l, p, f) for l, p, f in feats if p < ids0.shape[1]]
            d0, act0, lp0 = drop(ids0, feats, a0)
            per = []
            for text, ids in variants:
                dv, actv, lpv = drop(ids, feats, a0)
                per.append(dict(prompt=text, drop=dv, active=actv, base_logprob=lpv))
            rows.append(dict(graph=key, supernode=sname, features=len(feats), prompt=prompt, answer=fam['answer'],
                             drop_original=d0, active_original=act0, base_logprob_original=lp0, variants=per))
        log(stage='graph', graph=key, variants=len(variants))
    (args.out / 'GENERALIZE_ROWS.json').write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(),
                                                                  description=DESCRIPTION, rows=rows), indent=1))
    log(stage='done', rows=len(rows))


if __name__ == '__main__':
    main()
