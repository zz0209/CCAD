import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from p3_common import LETTERS, MODEL_PATH, READER_PATH, user_text

DESCRIPTION = ('P3 text-reader baseline: Qwen2.5-0.5B with a 4-way classification head on the last token, fully '
               'fine-tuned to read the prefix as text (question, options and the decoded reasoning so far) and predict '
               'the rollout\'s final answer, on the same training pairs, validation split, epochs, batch size and size '
               'of learning-rate grid as the probe; the learning rate and epoch with the lowest validation '
               'cross-entropy are kept and predict the answer distribution at every test boundary.')
LRS = [1e-5, 3e-5, 1e-4]


def reader_text(q, prefix_text):
    return f'{user_text(q)}\n\nReasoning so far:\n{prefix_text}'


def logits_for(model, seqs, pad_id, bs):
    out = []
    for b in range(0, len(seqs), bs):
        chunk = seqs[b:b + bs]
        T = max(map(len, chunk))
        ids = torch.full((len(chunk), T), pad_id, dtype=torch.long)
        att = torch.zeros((len(chunk), T), dtype=torch.long)
        for k, s in enumerate(chunk):
            ids[k, :len(s)] = torch.tensor(s)
            att[k, :len(s)] = 1
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out.append(model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits.float())
    return torch.cat(out)


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--micro-batch', type=int, default=4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--lrs', type=float, nargs='+', default=LRS)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    items = json.loads((args.run_dir / 'items.json').read_text())
    src_tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    tok = AutoTokenizer.from_pretrained(READER_PATH)
    assert tok.pad_token_id is not None
    enc = lambda q, ids: tok(reader_text(q, src_tok.decode(ids)), add_special_tokens=False)['input_ids']
    tr = [p for p in items['pairs'] if p['split'] == 'train']
    va = [p for p in items['pairs'] if p['split'] == 'val']
    x_tr = [enc(p, p['prefix_ids']) for p in tr]
    x_va = [enc(p, p['prefix_ids']) for p in va]
    y_tr = torch.tensor([LETTERS.index(p['label']) for p in tr], device='cuda')
    y_va = torch.tensor([LETTERS.index(p['label']) for p in va], device='cuda')
    print(f'[reader] train {len(tr)} val {len(va)}; tokens per train pass {sum(map(len, x_tr))}, '
          f'longest {max(map(len, x_tr + x_va))}', flush=True)
    accum = args.batch_size // args.micro_batch
    total = math.ceil(len(tr) / args.batch_size) * args.epochs
    grid, best, t0 = [], None, time.time()
    for lr in args.lrs:
        torch.manual_seed(args.seed)
        gen = np.random.default_rng(args.seed)
        model = AutoModelForSequenceClassification.from_pretrained(READER_PATH, num_labels=4, torch_dtype=torch.float32,
                                                                   attn_implementation='sdpa')
        model.config.pad_token_id = tok.pad_token_id
        model.gradient_checkpointing_enable()
        model.cuda().train()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        warm = max(1, total // 10)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
        curve = []
        for ep in range(args.epochs):
            order = gen.permutation(len(tr))
            for b in range(0, len(tr), args.batch_size):
                idx = order[b:b + args.batch_size]
                opt.zero_grad()
                for m in range(0, len(idx), args.micro_batch):
                    sub = idx[m:m + args.micro_batch]
                    logits = logits_for(model, [x_tr[i] for i in sub], tok.pad_token_id, len(sub))
                    (F.cross_entropy(logits, y_tr[sub], reduction='sum') / len(idx)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
            model.eval()
            with torch.no_grad():
                lv = logits_for(model, x_va, tok.pad_token_id, args.micro_batch)
            model.train()
            ce_va = float(F.cross_entropy(lv, y_va))
            acc_va = float((lv.argmax(1) == y_va).float().mean())
            curve.append(dict(epoch=ep + 1, val_ce=ce_va, val_acc=acc_va))
            print(f'[reader] lr {lr:g} epoch {ep + 1}/{args.epochs}: val CE {ce_va:.4f} acc {acc_va:.3f}, '
                  f'{time.time() - t0:.0f}s', flush=True)
            if best is None or ce_va < best['val_ce']:
                best = dict(lr=lr, epoch=ep + 1, val_ce=ce_va, val_acc=acc_va,
                            state={k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()})
        grid.append(dict(lr=lr, curve=curve))
        del model, opt
        torch.cuda.empty_cache()
    print(f'[reader] selected lr {best["lr"]:g} epoch {best["epoch"]} val CE {best["val_ce"]:.4f}', flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(READER_PATH, num_labels=4, torch_dtype=torch.float32,
                                                               attn_implementation='sdpa')
    model.config.pad_token_id = tok.pad_token_id
    model.load_state_dict(best['state'])
    model.cuda().eval()
    preds = []
    with torch.no_grad():
        for t in items['test']:
            seqs = [enc(t, t['reasoning_ids'][:c]) for c in t['cuts']]
            preds.append(dict(id=t['id'], probs=logits_for(model, seqs, tok.pad_token_id, 1).softmax(-1).tolist()))
    (args.run_dir / 'reader.json').write_text(json.dumps(dict(
        selected={k: v for k, v in best.items() if k != 'state'}, epochs=args.epochs, batch_size=args.batch_size,
        weight_decay=args.weight_decay, lrs=args.lrs, n_train=len(tr), n_val=len(va), grid=grid, traces=preds)))


if __name__ == '__main__':
    main()
