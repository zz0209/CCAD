import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

DESCRIPTION = ('Ground truth for the unit test. A hand-built residual model with three MLP layers and one position '
               'holds four circuits in separate dimensions: (A) a two-step circuit, (a XOR b) AND (c XOR d), computed '
               'through intermediate features at layer 0 and an output at layer 1 (Lange et al.); (B) a feature written '
               'at layer 0 and amplified in the same direction by layers 1 and 2; (C) a constant written by layer 2 for '
               'every input; (D) an input stored in two directions of which layer 2 reads only one; (E) the same '
               'feature of one input computed in parallel by every layer (cross-layer superposition); (F) an input '
               'stored in two directions, the first erased by layer 1, so that layer 2 computes from the second. A per-layer '
               'transcoder (PLT) and a cross-layer transcoder (CLT) are trained on its MLPs. Every active CLT feature '
               'is labeled by the circuit that receives most of its later writes, and the tests of the paper are '
               'applied to it: removing its input (one feature), and removing the inputs of the active PLT features at '
               'later layers. The removed fraction and the bounded explained share of the claimed later writes are '
               'reported per circuit, at inputs where the model\'s own later output in that circuit is not zero.')

D_MODEL, L = 20, 3


def inputs():
    rows = []
    for a, b, c, d in itertools.product([-1., 1.], repeat=4):
        for s, w, q, v in itertools.product([0., 1.], repeat=4):
            x = torch.zeros(D_MODEL)
            x[0:4] = torch.tensor([a, b, c, d])
            x[9] = s
            x[12] = w
            x[13] = w
            x[16] = q
            x[18] = v
            x[19] = v
            rows.append(x)
    return torch.stack(rows)


def mlp(l, r):
    # The hand-set MLP of layer l on residual states r (batch, D_MODEL); returns its output.
    out = torch.zeros_like(r)
    relu = torch.relu
    if l == 0:
        a, b, c, d = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
        out[:, 4] = relu(a - b - 1)
        out[:, 5] = relu(b - a - 1)
        out[:, 6] = relu(c - d - 1)
        out[:, 7] = relu(d - c - 1)
        out[:, 10] = relu(r[:, 9])
    elif l == 1:
        out[:, 8] = relu(r[:, 4] + r[:, 5] + r[:, 6] + r[:, 7] - 1)
        out[:, 10] = 2 * relu(r[:, 10])
        # (F) erases the first of two copies of an input, so that layer 2 can only read the second.
        out[:, 18] = -relu(r[:, 18])
    else:
        out[:, 10] = relu(r[:, 10])
        out[:, 11] = 1.
        out[:, 14] = relu(r[:, 13] - .5)
        out[:, 15] = relu(r[:, 19] - .5)
    # (E) cross-layer superposition: every layer computes the same feature of the same input into dimension 17.
    out[:, 17] = relu(r[:, 16])
    return out


def forward(x, edits=None):
    # edits: {layer: function(r) -> r} applied to the MLP input of that layer only (the residual stream is not changed).
    r, ins, outs = x.clone(), [], []
    for l in range(L):
        rin = edits[l](r) if edits and l in edits else r
        m = mlp(l, rin)
        ins.append(rin)
        outs.append(m)
        r = r + m
    return ins, outs


class Transcoder(torch.nn.Module):
    # Per-layer (cross=False) or cross-layer (cross=True) transcoder with ReLU features, no decoder bias, and the
    # tanh sparsity penalty on activation times decoder norm.
    def __init__(self, n, cross, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.cross = cross
        self.W_enc = torch.nn.Parameter(torch.randn(L, n, D_MODEL, generator=g) * .3)
        self.b_enc = torch.nn.Parameter(torch.zeros(L, n))
        n_out = L if cross else 1
        self.W_dec = torch.nn.Parameter(torch.randn(L, n, n_out, D_MODEL, generator=g) * .1)

    def encode(self, l, r):
        return torch.relu(r @ self.W_enc[l].T + self.b_enc[l])

    def dec(self, l, k):
        # Decoder of layer-l features into the MLP output of layer k, or None.
        if self.cross:
            return self.W_dec[l, :, k] if k >= l else None
        return self.W_dec[l, :, 0] if k == l else None

    def reconstruct(self, ins):
        zs = [self.encode(l, ins[l]) for l in range(L)]
        outs = []
        for k in range(L):
            o = 0
            for l in range(L):
                dk = self.dec(l, k)
                if dk is not None:
                    o = o + zs[l] @ dk
            outs.append(o)
        return zs, outs

    def penalty(self, zs, c=4.):
        p = 0
        for l in range(L):
            norm = self.W_dec[l].norm(dim=-1) if not self.cross else self.W_dec[l, :, l:].flatten(1).norm(dim=-1)
            p = p + torch.tanh(c * zs[l] * norm.reshape(1, -1) if not self.cross else c * zs[l] * norm).sum(-1).mean()
        return p


def train(cross, n, lam, steps, seed, x):
    ins, outs = forward(x)
    tc = Transcoder(n, cross, seed)
    opt = torch.optim.Adam(tc.parameters(), lr=1e-2)
    for step in range(steps):
        zs, rec = tc.reconstruct(ins)
        mse = sum(((rec[k] - outs[k]) ** 2).sum(-1).mean() for k in range(L))
        loss = mse + lam * tc.penalty(zs)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        zs, rec = tc.reconstruct(ins)
        fvu = [float(((rec[k] - outs[k]) ** 2).sum() / max(float((outs[k] ** 2).sum()), 1e-9)) for k in range(L)]
        l0 = [float((zs[l] > 0).float().sum(-1).mean()) for l in range(L)]
    return tc, dict(fvu=fvu, l0=l0, mse=float(mse))


def removal(W, b, x):
    # Smallest change of x that sets the listed pre-activations to zero.
    pre = W @ x + b
    return -W.T @ torch.linalg.solve(W @ W.T + 1e-6 * torch.eye(len(b)), pre)


CIRCUITS = {'A two-step': [8], 'B amplified': [10], 'C constant': [11], 'D redundant': [14], 'E parallel': [17],
            'F erased copy': [15]}


def bounded(claim, removed):
    # Same measure as summarize_depth_referee.bounded: 1 - sum_k (|c_k| - p_k)^2 / sum_k |c_k|^2, cut at 0.
    c = torch.tensor(claim)
    keep = c > 0
    p = torch.tensor(removed)[keep] / c[keep].sqrt()
    return float(max(0., 1 - float(((c[keep].sqrt() - p) ** 2).sum() / c.sum())))


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--features', type=int, default=12)
    parser.add_argument('--lam', type=float, default=.05)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'config.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    torch.manual_seed(0)
    x = inputs()
    results, rows = [], []
    for seed in args.seeds:
        clt, qc = train(True, args.features, args.lam, args.steps, seed, x)
        plt, qp = train(False, args.features, args.lam, args.steps, seed, x)
        results.append(dict(seed=seed, clt=qc, plt=qp))
        with torch.no_grad():
            ins, outs = forward(x)
            for l in range(L - 1):
                z = clt.encode(l, ins[l])
                for f in range(args.features):
                    later = torch.stack([clt.dec(l, k)[f] for k in range(l + 1, L)])
                    if later.norm() < 1e-3 or not (z[:, f] > 0).any():
                        continue
                    # Label by the circuit whose output dimensions receive most of the later write energy.
                    share = {name: float((later[:, dims] ** 2).sum()) for name, dims in CIRCUITS.items()}
                    total = float((later ** 2).sum())
                    label = max(share, key=share.get)
                    for i in torch.nonzero(z[:, f] > 0).flatten().tolist():
                        a = z[i, f]
                        claim = {k: a * clt.dec(l, k)[f] for k in range(l, L)}
                        def run(edits):
                            _, o = forward(x[i:i + 1], edits)
                            return {k: o[k][0] - outs[k][i] for k in range(L)}
                        W, b = clt.W_enc[l, f:f + 1], clt.b_enc[l, f:f + 1]
                        # Own input: change of the residual read by layer l and every later layer (the CLT reads the
                        # residual stream, so the change stays in it).
                        d_u = removal(W, b, ins[l][i])
                        res = {}
                        res['u'] = run({k: (lambda r, d=d_u: r + d) for k in range(l, L)})
                        # The same direction moved until the read projection is zero, past the threshold.
                        w = W[0]
                        d_z = -(w @ ins[l][i]) / (w @ w) * w
                        res['u_zero'] = run({k: (lambda r, d=d_z: r + d) for k in range(l, L)})
                        # PLT units: the active PLT features of every later layer, removed at their own MLP input.
                        def plt_edit(k):
                            def edit(r):
                                zk = plt.encode(k, r)
                                on = torch.nonzero(zk[0] > 0).flatten()
                                if len(on) == 0:
                                    return r
                                return r + removal(plt.W_enc[k, on], plt.b_enc[k, on], r[0])
                            return edit
                        res['s'] = run({k: plt_edit(k) for k in range(l + 1, L)})
                        dims = CIRCUITS[label]
                        row = dict(seed=seed, layer=l, feature=f, input=i, label=label,
                                   label_share=share[label] / total, activation=float(a),
                                   claim_energy=[float((claim[k] ** 2).sum()) for k in range(l + 1, L)],
                                   # Energy of the model's own later output in the circuit's dimensions at this input.
                                   target_energy=sum(float((outs[k][i, dims] ** 2).sum()) for k in range(l + 1, L)))
                        for name, ch in res.items():
                            row[f'removed_{name}_by_layer'] = [float(-(ch[k] * claim[k]).sum()) for k in range(l + 1, L)]
                            row[f'removed_{name}'] = sum(row[f'removed_{name}_by_layer']) / sum(row['claim_energy'])
                        rows.append(row)
        print(json.dumps(dict(seed=seed, clt=qc, plt=qp, rows=len(rows))), flush=True)
    for r in rows:
        r['bounded_u'] = bounded(r['claim_energy'], r['removed_u_by_layer'])
        r['bounded_u_zero'] = bounded(r['claim_energy'], r['removed_u_zero_by_layer'])
    summary = {}
    for label in CIRCUITS:
        # Features whose later writes go mostly (more than 60%) to one circuit, at inputs where the model's own later
        # output in that circuit is not zero.
        sel = [r for r in rows if r['label'] == label and r['label_share'] > .6 and r['target_energy'] > 1e-6]
        if not sel:
            continue
        q = lambda k: [float(torch.quantile(torch.tensor([r[k] for r in sel]), p)) for p in (.5, .25, .75)]
        summary[label] = dict(rows=len(sel), features=len({(r['seed'], r['layer'], r['feature']) for r in sel}),
                              layers=sorted({r['layer'] for r in sel}),
                              removed_u=q('removed_u'), removed_s=q('removed_s'), bounded_u=q('bounded_u'),
                              removed_u_zero=q('removed_u_zero'), bounded_u_zero=q('bounded_u_zero'))
    print(json.dumps(summary, indent=1))
    (args.out / 'TOY_REFEREE.json').write_text(json.dumps(dict(
        written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, training=results,
        summary=summary, rows=rows), indent=1))


if __name__ == '__main__':
    main()
