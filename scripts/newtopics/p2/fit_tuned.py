import argparse
import json
import math
import time
from pathlib import Path

import torch
from jlens.hooks import ActivationRecorder
from torch import nn

from common import band_layers, cap_gpu_memory, load_model, write_json

DESCRIPTION = ('Train a tuned lens (Belrose et al. 2023) for one checkpoint on the band layers: per-layer affine '
               'translator h + A h + b initialised at zero, read out with the model final norm and unembedding, '
               'trained to minimise KL(final-layer distribution || lens distribution) with the tuned-lens package '
               'defaults (SGD Nesterov, lr 1.0*(1-0.9), momentum 0.9, weight decay 1e-3, linear decay to zero, no '
               'warmup, 250 steps, gradient-norm clip 1.0, bf16 model) on FineWeb train documents disjoint from the '
               'J-lens fitting documents. Reports held-out KL of the tuned and logit lens on FineWeb eval documents.')


def chunks(tok, docs, n_seq, seq_len, wrap=False):
    # Returns n_seq sequences of BOS + (seq_len - 1) document tokens and the number of distinct sequences. With
    # wrap, a tokenizer that yields fewer sequences than n_seq cycles through them again from the start.
    out = []
    for doc in docs:
        ids = tok.encode(doc, add_special_tokens=False)
        for s in range(0, len(ids) - (seq_len - 1) + 1, seq_len - 1):
            out.append([tok.bos_token_id] + ids[s:s + seq_len - 1])
            if len(out) == n_seq:
                return torch.tensor(out), n_seq
    if not wrap or not out:
        raise ValueError(f'only {len(out)} sequences of {seq_len} tokens available, need {n_seq}')
    return torch.tensor([out[i % len(out)] for i in range(n_seq)]), len(out)


def kl_rows(lm, h, target_logp):
    logq = torch.log_softmax(lm.unembed(h).float(), -1)
    return (target_logp.exp() * (target_logp - logq)).sum(-1)


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--fineweb', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--skip_docs', type=int, default=200, help='train documents reserved for the J-lens fit')
    parser.add_argument('--steps', type=int, default=250)
    parser.add_argument('--seqs_per_step', type=int, default=32)
    parser.add_argument('--micro', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--eval_docs', type=int, default=64)
    parser.add_argument('--state_gib', type=float, default=1.3, help='translator params+grads+momentum per group')
    args = parser.parse_args()
    cap_gpu_memory()
    torch.manual_seed(0)
    row, hf, tok, lm = load_model(args.manifest, args.label)
    fw = json.load(open(args.fineweb))
    data, n_unique = chunks(tok, fw['train'][args.skip_docs:], args.steps * args.seqs_per_step, args.seq_len,
                            wrap=True)
    eval_data, _ = chunks(tok, fw['eval'][:args.eval_docs], args.eval_docs, args.seq_len)
    band = band_layers(lm.n_layers)
    final = lm.n_layers - 1
    d = lm.d_model
    per_layer = 3 * (d * d + d) * 4
    group_size = max(1, int(args.state_gib * 2 ** 30 // per_layer))
    groups = [band[i:i + group_size] for i in range(0, len(band), group_size)]
    print(f'{args.label}: band {band}, groups {groups}, {data.shape[0]} train seqs ({n_unique} distinct), '
          f'{len(eval_data)} eval seqs', flush=True)
    state, log = {}, []
    t0 = time.time()
    for gi, group in enumerate(groups):
        tr = nn.ModuleList([nn.Linear(d, d) for _ in group]).cuda()
        for m in tr:
            m.weight.data.zero_()
            m.bias.data.zero_()
        opt = torch.optim.SGD(tr.parameters(), lr=1.0 * (1 - 0.9), momentum=0.9, nesterov=True, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / args.steps))
        for step in range(args.steps):
            batch = data[step * args.seqs_per_step:(step + 1) * args.seqs_per_step]
            losses = torch.zeros(len(group))
            for mb in batch.split(args.micro):
                with torch.no_grad(), ActivationRecorder(lm.layers, at=group + [final]) as rec:
                    lm.forward(mb.cuda())
                    acts = {l: rec.activations[l][:, 1:].detach() for l in group + [final]}
                    target = torch.log_softmax(lm.unembed(acts[final]).float(), -1)
                n_mb = mb.shape[0] / batch.shape[0]
                for j, l in enumerate(group):
                    h = acts[l].float()
                    loss = kl_rows(lm, h + tr[j](h), target).mean() * n_mb
                    loss.backward()
                    losses[j] += loss.item()
                del acts, target
            for m in tr:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            if step % 25 == 0 or step == args.steps - 1:
                msg = dict(group=gi, step=step, kl=[round(x, 4) for x in losses.tolist()],
                           elapsed_s=round(time.time() - t0, 1))
                log.append(msg)
                print(json.dumps(msg), flush=True)
        for j, l in enumerate(group):
            state[l] = dict(weight=tr[j].weight.detach().cpu(), bias=tr[j].bias.detach().cpu())
        del tr, opt
        torch.cuda.empty_cache()
    # Held-out KL to the model's own final distribution, per band layer.
    ev = {l: dict(tuned=0.0, logit=0.0) for l in band}
    with torch.no_grad():
        for mb in eval_data.split(args.micro):
            with ActivationRecorder(lm.layers, at=band + [final]) as rec:
                lm.forward(mb.cuda())
                acts = {l: rec.activations[l][:, 1:] for l in band + [final]}
            target = torch.log_softmax(lm.unembed(acts[final]).float(), -1)
            for l in band:
                h = acts[l].float()
                W, b = state[l]['weight'].cuda(), state[l]['bias'].cuda()
                ev[l]['tuned'] += kl_rows(lm, h + h @ W.T + b, target).sum().item()
                ev[l]['logit'] += kl_rows(lm, h, target).sum().item()
    n_tok = eval_data.shape[0] * (eval_data.shape[1] - 1)
    ev = {l: {k: v / n_tok for k, v in e.items()} for l, e in ev.items()}
    print('held-out KL (nats/token):', json.dumps({l: {k: round(v, 3) for k, v in e.items()} for l, e in ev.items()}),
          flush=True)
    torch.save(dict(translators=state, band=band, label=args.label, revision=row['revision']), args.out)
    write_json(str(args.out) + '.json', dict(label=args.label, band=band, groups=groups, steps=args.steps,
                                             seqs_per_step=args.seqs_per_step, seq_len=args.seq_len,
                                             train_tokens=int(data.numel()), distinct_train_sequences=n_unique,
                                             heldout_kl=ev, log=log,
                                             seconds=time.time() - t0, finite=all(
            math.isfinite(v) for e in ev.values() for v in e.values())))
    print('TUNED_DONE', args.label, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
