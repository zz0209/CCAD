import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from p3_common import LETTERS

DESCRIPTION = ('P3 foresight probe: an attention-pooled probe (per-token LayerNorm without affine parameters, 4 learned '
               'query vectors with softmax pooling over all prefix tokens except the first, a linear map to 4 logits) '
               'on the stored residual stream of each prefix, trained with cross-entropy on the final answers of the '
               'cut rollouts. Each candidate layer and learning rate is trained for a fixed number of epochs; the layer, '
               'learning rate and epoch with the lowest validation cross-entropy are kept and that probe predicts the '
               'answer distribution at every test boundary.')
LRS = [3e-4, 1e-3, 3e-3]


class AttnPoolProbe(nn.Module):
    def __init__(self, d, heads=4, classes=4):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.query = nn.Parameter(torch.zeros(heads, d))
        self.out = nn.Linear(heads * d, classes)

    def forward(self, x, mask):
        h = self.norm(x.float())
        s = torch.einsum('btd,hd->bht', h, self.query) / math.sqrt(h.shape[-1])
        a = s.masked_fill(~mask[:, None, :], float('-inf')).softmax(-1)
        return self.out(torch.einsum('bht,btd->bhd', a, h).flatten(1))


def batch(X, spans):
    T = max(n for _, n in spans)
    x = torch.zeros(len(spans), T, X.shape[1], dtype=X.dtype, device=X.device)
    mask = torch.zeros(len(spans), T, dtype=torch.bool, device=X.device)
    for k, (s, n) in enumerate(spans):
        x[k, :n] = X[s:s + n]
        mask[k, 1:n] = True
    return x, mask


def evaluate(probe, X, spans, y, bs=32):
    probe.eval()
    logits = []
    with torch.no_grad():
        for b in range(0, len(spans), bs):
            logits.append(probe(*batch(X, spans[b:b + bs])))
    probe.train()
    logits = torch.cat(logits)
    return float(F.cross_entropy(logits, y)), float((logits.argmax(1) == y).float().mean())


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--layers', type=int, nargs='+', default=None)
    parser.add_argument('--lrs', type=float, nargs='+', default=LRS)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    rd = args.run_dir / 'resid'
    index = json.loads((rd / 'index.json').read_text())
    layers = args.layers or index['layers']
    tr = [p for p in index['pairs'] if p['split'] == 'train']
    va = [p for p in index['pairs'] if p['split'] == 'val']
    y_tr = torch.tensor([LETTERS.index(p['label']) for p in tr], device='cuda')
    y_va = torch.tensor([LETTERS.index(p['label']) for p in va], device='cuda')
    sp_tr = [(p['start'], p['length']) for p in tr]
    sp_va = [(p['start'], p['length']) for p in va]
    steps_per_epoch = math.ceil(len(tr) / args.batch_size)
    total = steps_per_epoch * args.epochs
    grid, best, t0 = [], None, time.time()
    for layer in layers:
        X = torch.from_numpy(np.load(rd / f'pairs_L{layer}.npy', mmap_mode='r')[:]).cuda()
        for lr in args.lrs:
            torch.manual_seed(args.seed)
            gen = np.random.default_rng(args.seed)
            probe = AttnPoolProbe(X.shape[1]).cuda()
            opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=args.weight_decay)
            warm = max(1, total // 10)
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
            curve = []
            for ep in range(args.epochs):
                order = gen.permutation(len(tr))
                for b in range(0, len(tr), args.batch_size):
                    idx = order[b:b + args.batch_size]
                    loss = F.cross_entropy(probe(*batch(X, [sp_tr[i] for i in idx])), y_tr[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sched.step()
                ce_va, acc_va = evaluate(probe, X, sp_va, y_va)
                ce_tr, acc_tr = evaluate(probe, X, sp_tr, y_tr)
                curve.append(dict(epoch=ep + 1, train_ce=ce_tr, train_acc=acc_tr, val_ce=ce_va, val_acc=acc_va))
                if best is None or ce_va < best['val_ce']:
                    best = dict(layer=layer, lr=lr, epoch=ep + 1, val_ce=ce_va, val_acc=acc_va,
                                state={k: v.detach().clone() for k, v in probe.state_dict().items()})
            grid.append(dict(layer=layer, lr=lr, curve=curve))
            print(f'[probe] layer {layer} lr {lr:g}: best val CE {min(c["val_ce"] for c in curve):.4f} '
                  f'(train CE at end {curve[-1]["train_ce"]:.4f}), {time.time() - t0:.0f}s', flush=True)
        del X
        torch.cuda.empty_cache()
    print(f'[probe] selected layer {best["layer"]} lr {best["lr"]:g} epoch {best["epoch"]} '
          f'val CE {best["val_ce"]:.4f} acc {best["val_acc"]:.3f}', flush=True)
    X = torch.from_numpy(np.load(rd / f'test_L{best["layer"]}.npy', mmap_mode='r')[:]).cuda()
    probe = AttnPoolProbe(X.shape[1]).cuda()
    probe.load_state_dict(best['state'])
    probe.eval()
    preds = []
    with torch.no_grad():
        for t in index['test']:
            spans = [(t['start'], n) for n in t['prefix_lengths']]
            preds.append(dict(id=t['id'], probs=probe(*batch(X, spans)).softmax(-1).tolist()))
    torch.save(best['state'], args.run_dir / 'probe_best.pt')
    (args.run_dir / 'probe.json').write_text(json.dumps(dict(
        selected={k: v for k, v in best.items() if k != 'state'}, hidden=X.shape[1], epochs=args.epochs,
        batch_size=args.batch_size, weight_decay=args.weight_decay, lrs=args.lrs, layers=layers,
        n_train=len(tr), n_val=len(va), grid=grid, traces=preds)))


if __name__ == '__main__':
    main()
