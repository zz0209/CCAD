import argparse
import json
import math
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from jlens.hooks import ActivationRecorder
from muon import SingleDeviceMuonWithAuxAdam
from torch import nn

from common import band_layers, cap_gpu_memory, load_model, write_json
from fit_tuned import chunks, kl_rows

DESCRIPTION = ('Train the converged tuned lens (TL-full of the P2 formal study) for one checkpoint on the band '
               'layers: per-layer affine translator h + A h + b initialised at zero, read out with the model final norm '
               'and unembedding, loss KL(final-layer distribution || lens distribution) over all positions after BOS. '
               'Optimizer: Muon (KellerJordan/Muon SingleDeviceMuonWithAuxAdam) on the translator matrices (lr 0.02, '
               'momentum 0.95, weight decay 0) and Adam on the biases (lr 3e-4, betas 0.9/0.95, eps 1e-10), constant '
               'learning rate, 32 sequences of 128 tokens per step. Data: FineWeb sample-10BT parquet documents in file '
               'order, excluding the P2 slice documents; the first 32 sequences are the convergence set, the next ones '
               'are training data. Every 32 steps the mean held-out KL over band layers is measured; at steps that are '
               'multiples of 128 and at least 256, training stops when that KL improved by less than 1% since 75% of '
               'the current step count. Cap: 1024 steps (2^22 tokens). Resumable from the .ckpt file.')
def parquet_sequences(tok, path, exclude_ids, n_seq, seq_len, batch_rows=2000):
    out, n_docs, n_excluded, last_row = [], 0, 0, -1
    row = -1
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=['id', 'text']):
        for doc_id, text in zip(batch.column('id').to_pylist(), batch.column('text').to_pylist()):
            row += 1
            if doc_id in exclude_ids:
                n_excluded += 1
                continue
            n_docs += 1
            ids = tok.encode(text, add_special_tokens=False)
            for s in range(0, len(ids) - (seq_len - 1) + 1, seq_len - 1):
                out.append([tok.bos_token_id] + ids[s:s + seq_len - 1])
                if len(out) == n_seq:
                    return torch.tensor(out), dict(docs_used=n_docs, docs_excluded=n_excluded, last_row=row)
    raise ValueError(f'{path}: only {len(out)} sequences available, need {n_seq}')


def mean_kl(lm, tr, group, final, data, micro, device):
    tot = torch.zeros(len(group))
    with torch.no_grad():
        for mb in data.split(micro):
            with ActivationRecorder(lm.layers, at=group + [final]) as rec:
                lm.forward(mb.to(device))
                acts = {l: rec.activations[l][:, 1:] for l in group + [final]}
            target = torch.log_softmax(lm.unembed(acts[final]).float(), -1)
            for j, l in enumerate(group):
                h = acts[l].float()
                tot[j] += kl_rows(lm, h + tr[j](h), target).sum().item()
    return (tot / (data.shape[0] * (data.shape[1] - 1))).tolist()


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--fineweb', required=True, help='natural_fineweb_slice.json (exclusion ids, eval docs)')
    parser.add_argument('--parquet', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max_steps', type=int, default=1024)
    parser.add_argument('--seqs_per_step', type=int, default=32)
    parser.add_argument('--conv_seqs', type=int, default=32)
    parser.add_argument('--micro', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--eval_docs', type=int, default=64)
    parser.add_argument('--state_gib', type=float, default=4.0, help='translator params+grads+momentum per group')
    parser.add_argument('--eval_every', type=int, default=32)
    parser.add_argument('--check_every', type=int, default=128, help='must be a multiple of 4 * eval_every')
    parser.add_argument('--min_steps', type=int, default=256)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu_cap_gib', type=float, default=14.0)
    parser.add_argument('--train_data', choices=['parquet', 'jdocs'], default='parquet',
                        help='jdocs (TL-match): the J-lens fitting documents, slice train 0-199, each document that '
                             'has at least seq_len - 1 tokens giving BOS + its first seq_len - 1 tokens; training '
                             'cycles through them in order for max_steps steps without the convergence stop')
    args = parser.parse_args()
    if args.check_every % (4 * args.eval_every):
        raise ValueError('check_every must be a multiple of 4 * eval_every so that 75% of a check step is evaluated')
    if args.device == 'cuda':
        cap_gpu_memory(args.gpu_cap_gib)
    torch.manual_seed(0)
    row, hf, tok, lm = load_model(args.manifest, args.label, args.device)
    fw = json.load(open(args.fineweb))
    if args.train_data == 'parquet':
        data, data_info = parquet_sequences(tok, args.parquet, set(fw['document_ids']),
                                            args.conv_seqs + args.max_steps * args.seqs_per_step, args.seq_len)
        conv, train = data[:args.conv_seqs], data[args.conv_seqs:]
    else:
        conv, data_info = parquet_sequences(tok, args.parquet, set(fw['document_ids']), args.conv_seqs, args.seq_len)
        seqs = [tok.encode(doc, add_special_tokens=False)[:args.seq_len - 1] for doc in fw['train'][:200]]
        seqs = [[tok.bos_token_id] + s for s in seqs if len(s) == args.seq_len - 1]
        idx = [i % len(seqs) for i in range(args.max_steps * args.seqs_per_step)]
        train = torch.tensor(seqs)[idx]
        data_info = dict(conv=data_info, jdocs_sequences=len(seqs), jdocs_tokens=len(seqs) * args.seq_len)
        args.min_steps = args.max_steps + 1
    eval_data, _ = chunks(tok, fw['eval'][:args.eval_docs], args.eval_docs, args.seq_len)
    band = band_layers(lm.n_layers)
    final = lm.n_layers - 1
    d = lm.d_model
    group_size = max(1, int(args.state_gib * 2 ** 30 // (3 * (d * d + d) * 4)))
    groups = [band[i:i + group_size] for i in range(0, len(band), group_size)]
    print(f'{args.label}: band {band}, groups {groups}, {train.shape[0]} train seqs available, data {data_info}',
          flush=True)
    ckpt_path = Path(str(args.out) + '.ckpt')
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False) if ckpt_path.exists() else None
    state = ck['state'] if ck else {}
    group_meta = ck['group_meta'] if ck else []
    t0 = time.time()
    for gi, group in enumerate(groups):
        if gi < len(group_meta) and group_meta[gi]['finished']:
            continue
        tr = nn.ModuleList([nn.Linear(d, d) for _ in group]).to(args.device)
        for m in tr:
            m.weight.data.zero_()
            m.bias.data.zero_()
        opt = SingleDeviceMuonWithAuxAdam([
            dict(params=[m.weight for m in tr], use_muon=True, lr=0.02, momentum=0.95, weight_decay=0),
            dict(params=[m.bias for m in tr], use_muon=False, lr=3e-4, betas=(0.9, 0.95), eps=1e-10,
                 weight_decay=0)])
        hist, start = {}, 0
        if ck and ck.get('open_group') == gi:
            tr.load_state_dict(ck['translators'])
            opt.load_state_dict(ck['optimizer'])
            hist, start = {int(k): v for k, v in ck['hist'].items()}, ck['step']
            print(f'resumed group {gi} at step {start}', flush=True)
        if 0 not in hist:
            hist[0] = mean_kl(lm, tr, group, final, conv, args.micro, args.device)
        converged, step = False, start
        while step < args.max_steps and not converged:
            batch = train[step * args.seqs_per_step:(step + 1) * args.seqs_per_step]
            for mb in batch.split(args.micro):
                with torch.no_grad(), ActivationRecorder(lm.layers, at=group + [final]) as rec:
                    lm.forward(mb.to(args.device))
                    acts = {l: rec.activations[l][:, 1:].detach() for l in group + [final]}
                    target = torch.log_softmax(lm.unembed(acts[final]).float(), -1)
                n_mb = mb.shape[0] / batch.shape[0]
                for j, l in enumerate(group):
                    h = acts[l].float()
                    (kl_rows(lm, h + tr[j](h), target).mean() * n_mb).backward()
                del acts, target
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.eval_every == 0:
                hist[step] = mean_kl(lm, tr, group, final, conv, args.micro, args.device)
                msg = dict(group=gi, step=step, tokens=step * args.seqs_per_step * args.seq_len,
                           conv_kl_mean=round(sum(hist[step]) / len(group), 4), elapsed_s=round(time.time() - t0, 1))
                print(json.dumps(msg), flush=True)
            if step % args.check_every == 0 and step >= args.min_steps:
                ref = sum(hist[step * 3 // 4]) / len(group)
                now = sum(hist[step]) / len(group)
                converged = (ref - now) / ref < 0.01
            if step % args.check_every == 0 or converged or step == args.max_steps:
                torch.save(dict(state=state, group_meta=group_meta, open_group=gi, step=step, hist=hist,
                                translators=tr.state_dict(), optimizer=opt.state_dict()), str(ckpt_path) + '.tmp')
                Path(str(ckpt_path) + '.tmp').replace(ckpt_path)
        for j, l in enumerate(group):
            state[l] = dict(weight=tr[j].weight.detach().cpu(), bias=tr[j].bias.detach().cpu())
        group_meta.append(dict(group=group, steps=step, converged=converged, finished=True,
                               train_tokens=step * args.seqs_per_step * args.seq_len,
                               conv_kl={str(k): v for k, v in sorted(hist.items())}))
        torch.save(dict(state=state, group_meta=group_meta, open_group=None), str(ckpt_path) + '.tmp')
        Path(str(ckpt_path) + '.tmp').replace(ckpt_path)
        print(f'group {gi} done: steps {step}, converged {converged}', flush=True)
        del tr, opt
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    ev = {l: dict(tuned=0.0, logit=0.0) for l in band}
    with torch.no_grad():
        for mb in eval_data.split(args.micro):
            with ActivationRecorder(lm.layers, at=band + [final]) as rec:
                lm.forward(mb.to(args.device))
                acts = {l: rec.activations[l][:, 1:] for l in band + [final]}
            target = torch.log_softmax(lm.unembed(acts[final]).float(), -1)
            for l in band:
                h = acts[l].float()
                W, b = state[l]['weight'].to(args.device), state[l]['bias'].to(args.device)
                ev[l]['tuned'] += kl_rows(lm, h + h @ W.T + b, target).sum().item()
                ev[l]['logit'] += kl_rows(lm, h, target).sum().item()
    n_tok = eval_data.shape[0] * (eval_data.shape[1] - 1)
    ev = {l: {k: v / n_tok for k, v in e.items()} for l, e in ev.items()}
    print('held-out KL (nats/token):', json.dumps({l: {k: round(v, 3) for k, v in e.items()} for l, e in ev.items()}),
          flush=True)
    torch.save(dict(translators=state, band=band, label=args.label, revision=row['revision']), args.out)
    write_json(str(args.out) + '.json', dict(
        label=args.label, recipe='muon_converged' if args.train_data == 'parquet' else 'muon_jdocs_matched',
        train_data=args.train_data, band=band, groups=groups, group_meta=group_meta,
        seqs_per_step=args.seqs_per_step, seq_len=args.seq_len, max_steps=args.max_steps, eval_every=args.eval_every,
        check_every=args.check_every, min_steps=args.min_steps, data=data_info,
        parquet=str(args.parquet), heldout_kl=ev, all_converged=all(g['converged'] for g in group_meta),
        seconds=time.time() - t0, finite=all(math.isfinite(v) for e in ev.values() for v in e.values())))
    print('TUNED_DONE', args.label, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
