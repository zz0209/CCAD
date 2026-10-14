import argparse
import json
import time

import numpy as np
import torch

from p1common import (DATASETS, LAYERS, capture, encode_pair, load_model, load_tokenizer, pad_batch, prompt_text,
                      read_jsonl, run_dir, stable_hash, token_batches, write_jsonl)

DESCRIPTION = ('P1 self-contrast inputs: for the first n examples of each training set, the base model answers the '
               'same prompts by sampling at temperature 1 (no top-k/top-p, no repetition penalty), then both the '
               "dataset responses and the model's own responses are teacher-forced and the residual stream after each "
               'grid layer is averaged over response tokens (per example, then over examples). Writes '
               'forecast/own_responses/<dataset>.jsonl and forecast/sc_means.pt.')


@torch.no_grad()
def generate(lm, tok, users, gen, batch, seed):
    tok.padding_side = 'left'
    texts = [prompt_text(tok, u) for u in users]
    torch.manual_seed(seed)
    eos = set(lm.generation_config.eos_token_id)
    out = []
    for s in range(0, len(texts), batch):
        enc = tok(texts[s:s + batch], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        g = lm.generate(**enc, do_sample=True, temperature=gen['temperature'], top_p=gen['top_p'], top_k=gen['top_k'],
                        repetition_penalty=gen['repetition_penalty'], max_new_tokens=gen['max_new_tokens'],
                        pad_token_id=tok.pad_token_id)
        new = g[:, enc['input_ids'].shape[1]:]
        for row in new:
            ids = row.tolist()
            finished = bool(eos & set(ids))
            out.append(dict(response=tok.decode(ids, skip_special_tokens=True).strip(), finished=finished))
    tok.padding_side = 'right'
    return out


@torch.no_grad()
def response_means(lm, tok, pairs, max_tokens=24576):
    enc = [encode_pair(tok, u, r) for u, r in pairs]
    seqs = [p + q for p, q in enc]
    spans = [(len(p), len(p) + len(q) - 1) for p, q in enc]  # response tokens, <|im_end|> excluded
    sums = {l: torch.zeros(lm.config.hidden_size, dtype=torch.float64, device='cuda') for l in LAYERS}
    norms = {l: 0.0 for l in LAYERS}
    # an empty response has no response tokens and is left out of the mean (counted in n_empty)
    keep = [i for i in range(len(seqs)) if spans[i][1] > spans[i][0]]
    n_tok = 0
    for local in token_batches(np.array([len(seqs[i]) for i in keep]), max_tokens, 64):
        idx = [keep[j] for j in local]
        ids, mask = pad_batch([seqs[i] for i in idx], tok.pad_token_id)
        with capture(lm, LAYERS) as st:
            lm.model(input_ids=ids, attention_mask=mask)
        for l in LAYERS:
            for r, i in enumerate(idx):
                a, b = spans[i]
                h = st[l][r, a:b].float()
                sums[l] += h.mean(0).double()
                norms[l] += float(h.norm(dim=-1).sum())
        n_tok += sum(spans[i][1] - spans[i][0] for i in idx)
    return ({l: (sums[l] / len(keep)).float().cpu() for l in LAYERS},
            {l: norms[l] / n_tok for l in LAYERS}, len(seqs) - len(keep))


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run', required=True)
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    args = parser.parse_args()
    rd = run_dir(args.run)
    cfg = json.loads((rd / 'config.json').read_text())
    sc = cfg['selfcontrast']
    out_path = rd / 'forecast' / 'sc_means.pt'
    res = torch.load(out_path) if out_path.exists() else {}
    tok = load_tokenizer()
    lm = load_model()
    t0 = time.time()
    for k, name in enumerate(args.datasets):
        if name in res:
            continue
        rows = read_jsonl(rd / 'data' / f'{name}.jsonl')[:sc['n_examples']]
        own_path = rd / 'forecast' / 'own_responses' / f'{name}.jsonl'
        if own_path.exists():
            own = read_jsonl(own_path)
        else:
            own = generate(lm, tok, [r['user'] for r in rows], sc['gen'], sc['gen_batch'], stable_hash(name) % 2 ** 31)
            write_jsonl(own_path, [dict(user=r['user'], **o) for r, o in zip(rows, own)])
        data_mean, data_norm, data_empty = response_means(lm, tok, [(r['user'], r['response']) for r in rows])
        own_mean, own_norm, own_empty = response_means(lm, tok, [(r['user'], o['response']) for r, o in zip(rows, own)])
        assert data_empty == 0
        res[name] = dict(data_mean=data_mean, own_mean=own_mean, data_token_norm=data_norm, own_token_norm=own_norm,
                         own_finished=float(np.mean([o['finished'] for o in own])), own_empty=own_empty)
        torch.save(res, out_path)
        print(f'{name}: {k + 1}/{len(args.datasets)} done, {time.time() - t0:.0f}s elapsed; own finished '
              f'{res[name]["own_finished"]:.2f}; |sc| by layer '
              + ' '.join(f'{l}:{float((data_mean[l] - own_mean[l]).norm()):.2f}/{own_norm[l]:.1f}' for l in LAYERS),
              flush=True)


if __name__ == '__main__':
    main()
