import argparse
import json
import time
from pathlib import Path

import torch
from jlens.hooks import ActivationRecorder

from common import cap_gpu_memory, first_token, load_model, write_json

DESCRIPTION = ('Count the two-hop prompts one checkpoint answers correctly (top-1 token is an answer first token, the '
               'swap_eval.py rule and forward path) on each given prompt set. The step-2 driver uses the counts to '
               'decide whether an OLMo-1B intermediate checkpoint can enter any claim (n >= 20 on v1 or v2).')
MIN_N = 20


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--tasks', nargs='+', required=True, help='name=task.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu_cap_gib', type=float, default=14.0)
    args = parser.parse_args()
    if args.device == 'cuda':
        cap_gpu_memory(args.gpu_cap_gib)
    t0 = time.time()
    row, hf, tok, lm = load_model(args.manifest, args.label, args.device)
    final = lm.n_layers - 1
    out = dict(label=args.label, revision=row['revision'], commit=row['commit'], min_n=MIN_N, sets={})
    for spec in args.tasks:
        name, path = spec.split('=')
        items = json.load(open(path))['items']
        rows = []
        for it in items:
            ids = lm.encode(it['two_hop'], max_length=512)
            if ids[0, 0].item() != tok.bos_token_id:
                raise ValueError('missing BOS')
            with torch.no_grad(), ActivationRecorder(lm.layers, at=[final]) as rec:
                lm.forward(ids)
                logits = lm.unembed(rec.activations[final][0, -1].float()).float()
            A = {first_token(tok, a) for a in it['answers']}
            rows.append(dict(id=it['id'], pair=it['pair'], held_out=it['held_out'], bridge=it['bridge'],
                             correct=logits.argmax().item() in A))
        n = sum(r['correct'] for r in rows)
        n_ho = sum(r['correct'] for r in rows if r['held_out'])
        out['sets'][name] = dict(task=path, n_items=len(rows), n_correct=n, n_correct_held_out=n_ho, rows=rows)
        print(f'{args.label} {name}: {n}/{len(rows)} correct, held-out {n_ho}', flush=True)
    out['eligible_any'] = any(s['n_correct'] >= MIN_N for s in out['sets'].values())
    out['seconds'] = time.time() - t0
    write_json(args.out, out)
    print('ELIGIBILITY_DONE', args.label, out['eligible_any'], f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
