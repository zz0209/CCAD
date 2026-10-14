import argparse
import json
import time

import numpy as np
import torch
from peft import get_peft_model

from battery import Battery
from p1common import ANIMALS, DATASETS, load_model, load_tokenizer, read_jsonl, run_dir, save_json, token_batches
from train import lora_config, make_batch

DESCRIPTION = ('P1 gradient baseline: with the LoRA adapter at its seed-0 initialization (the same as the seed-0 '
               'fine-tune), the gradient of each behaviour score on the dev split and the gradient of the mean '
               'assistant-token loss on the first n examples of each training set, both over the LoRA parameters. '
               'First-order forecast of a score change under a gradient step: -<g_behaviour, g_data> (variant inner) '
               'and the same with cosine normalization (variant cosine). Writes forecast/ga.json.')


def flat_grad(params):
    return torch.cat([p.grad.reshape(-1) for p in params])


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run', required=True)
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--max_tokens', type=int, default=1024)
    args = parser.parse_args()
    rd = run_dir(args.run)
    cfg = json.loads((rd / 'config.json').read_text())
    tok = load_tokenizer()
    bat = Battery(tok, json.loads((rd / 'battery.json').read_text()), 'dev')
    lm = load_model()
    torch.manual_seed(0)
    model = get_peft_model(lm, lora_config(cfg))
    inner = model.base_model.model
    params = [p for p in model.parameters() if p.requires_grad]
    n_par = sum(p.numel() for p in params)
    t0 = time.time()
    # behaviour gradients are kept in host memory (81 x 36.9M fp32, about 12 GB)
    G = torch.zeros(len(bat.names), n_par)
    for bi, name in enumerate(bat.names):
        if bat.kinds[bi] == 'animal':
            continue
        idx = list(np.where(bat.lt_beh == bi)[0])
        model.zero_grad(set_to_none=False)
        for local in token_batches(np.array([len(bat.lt_seqs[i]) for i in idx]), args.max_tokens, 64):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                v = bat.last_token_tensor(inner, [idx[j] for j in local])
            (v.sum() / len(idx)).backward()
        G[bi] = flat_grad(params).cpu()
    print(f'behaviour gradients (non-animal) {time.time() - t0:.0f}s', flush=True)
    P = len(bat.animal_prompts)
    for pi in range(P):
        seq_ids = [i for i, k in enumerate(bat.an_key) if k[0] == pi]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            L = bat.animal_matrix(inner, seq_ids)[pi]
        f = L - torch.logsumexp(L, 0)
        for a in ANIMALS:
            bi = bat.animal_index[a]
            g = torch.autograd.grad(f[ANIMALS.index(a)] / P, params, retain_graph=True)
            G[bi] += torch.cat([x.reshape(-1) for x in g]).cpu()
    print(f'behaviour gradients (all) {time.time() - t0:.0f}s', flush=True)
    n = cfg['selfcontrast']['n_examples']
    pred = dict(inner={}, cosine={})
    gnorm = G.norm(dim=1)
    data_norms = {}
    for d in args.datasets:
        rows = read_jsonl(rd / 'data' / f'{d}.jsonl')[:n]
        model.zero_grad(set_to_none=False)
        total_tokens = 0
        for s in range(0, n, args.batch):
            ids, mask, lab = make_batch(tok, rows[s:s + args.batch])
            k = int((lab[:, 1:] != -100).sum())
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=mask, labels=lab).loss
            (loss * k).backward()
            total_tokens += k
        g = (flat_grad(params) / total_tokens).cpu()
        ip = -(G @ g)
        pred['inner'][d] = {nm: float(x) for nm, x in zip(bat.names, ip.cpu().numpy())}
        pred['cosine'][d] = {nm: float(x) for nm, x in zip(bat.names, (ip / (gnorm * g.norm())).cpu().numpy())}
        data_norms[d] = float(g.norm())
        print(f'{d}: data gradient done, {time.time() - t0:.0f}s elapsed', flush=True)
    save_json(rd / 'forecast' / 'ga.json', dict(pred=pred, lora_params=n_par, data_grad_norm=data_norms,
                                                behaviour_grad_norm={nm: float(x) for nm, x in
                                                                     zip(bat.names, gnorm.cpu().numpy())}))


if __name__ == '__main__':
    main()
