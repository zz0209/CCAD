from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import platform
import random
import shutil
import time
import traceback

import numpy as np
import pyarrow.parquet as pq
import scipy
from scipy.optimize import linear_sum_assignment, minimize
from sklearn.covariance import LedoitWolf
import torch
import transformers

DESCRIPTION = ('Carry the published SFC Gemma-2-2B SHIFT explanation from Gemma Scope 16k dictionaries to the 65k '
               'dictionaries at the same sites, compare label-free matching with response-trained correspondences, '
               'and use each target-only deletion in the SHIFT classifier procedure.')
PROFESSOR, NURSE = 21, 13
PRONOUNS = {'he', 'she', 'him', 'her', 'his', 'hers', 'himself', 'herself'}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


class Run:
    def __init__(self, cfg, config_path, resume):
        self.dir = Path(cfg['run_root']) / cfg['run_id']
        if self.dir.exists():
            status = json.loads((self.dir / 'status.json').read_text())['status']
            assert resume and status != 'PASS', f'{self.dir} exists with status {status}'
        (self.dir / 'code').mkdir(parents=True, exist_ok=True)
        shutil.copy2(__file__, self.dir / 'code' / Path(__file__).name)
        resolved = dict(cfg, config_path=str(config_path), script_sha256=sha256(__file__), python=platform.python_version(),
                        torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
                        scipy=scipy.__version__, cuda_device=torch.cuda.get_device_name(0), resolved_at_utc=now())
        (self.dir / 'config.resolved.json').write_text(json.dumps(resolved, indent=1))
        self.started = now()
        self.status('RUNNING')
        self.t0 = time.time()

    def status(self, value, error=None):
        ended = now() if value in ('PASS', 'FAIL') else None
        (self.dir / 'status.json').write_text(json.dumps(dict(status=value, started_at_utc=self.started,
                                                              ended_at_utc=ended, error=error), indent=1))

    def progress(self, **kw):
        row = dict(written_at_utc=now(), elapsed=round(time.time() - self.t0, 1), **kw)
        (self.dir / 'progress.json').write_text(json.dumps(row))
        with (self.dir / 'progress.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    def records(self, rows):
        with (self.dir / 'metrics.raw.jsonl').open('a') as f:
            f.write(''.join(json.dumps(dict(run_id=self.dir.name, **r)) + '\n' for r in rows))


def read_split(cfg, split):
    path, = Path(cfg['dataset_dir']).glob(f'data/{split}-*.parquet')
    table = pq.read_table(path).to_pydict()
    return table['hard_text'], table['profession'], table['gender']


def sfc_ambiguous(data, seed):
    text, prof, gen = data
    neg = [x for x, p, g in zip(text, prof, gen) if p == PROFESSOR and g == 0]
    pos = [x for x, p, g in zip(text, prof, gen) if p == NURSE and g == 1]
    n = min(len(neg), len(pos))
    texts, labels = neg[:n] + pos[:n], [0] * n + [1] * n
    idx = list(range(2 * n))
    random.Random(seed).shuffle(idx)
    lab = np.array([labels[i] for i in idx])
    return [texts[i] for i in idx], lab, lab.copy()


def sfc_balanced(data, seed):
    text, prof, gen = data
    groups = [[x for x, p, g in zip(text, prof, gen) if p == pp and g == gg]
              for pp, gg in [(PROFESSOR, 0), (PROFESSOR, 1), (NURSE, 0), (NURSE, 1)]]
    n = min(len(g) for g in groups)
    texts = sum((g[:n] for g in groups), [])
    true = [0] * (2 * n) + [1] * (2 * n)
    spur = [0] * n + [1] * n + [0] * n + [1] * n
    idx = list(range(4 * n))
    random.Random(seed).shuffle(idx)
    return [texts[i] for i in idx], np.array([true[i] for i in idx]), np.array([spur[i] for i in idx])


def jumprelu(pre, threshold):
    return pre * (pre > threshold)


class Model:
    def __init__(self, cfg, sites, dev):
        self.dev = dev
        self.tok = transformers.AutoTokenizer.from_pretrained(cfg['model_dir'])
        self.tok.padding_side = 'right'
        full = transformers.AutoModelForCausalLM.from_pretrained(cfg['model_dir'], dtype=torch.bfloat16,
                                                                 attn_implementation='eager', local_files_only=True)
        self.backbone = full.model
        if cfg.get('full_model'):
            # Natural-text families need the next-token distribution, so every layer and the output head are kept.
            self.full = full.to(dev).eval().requires_grad_(False)
        else:
            # Only the classifier layer output is read, so later layers are never run.
            self.backbone.layers = self.backbone.layers[:cfg['probe_layer'] + 1]
        self.backbone.to(dev).eval().requires_grad_(False)
        self.mode, self.captured = None, None
        layers = self.backbone.layers
        for site in sites:
            kind, layer = site.split('_')
            if kind == 'resid':
                layers[int(layer)].register_forward_hook(self.resid_hook(site))
            else:
                assert kind == 'attn', site
                layers[int(layer)].self_attn.o_proj.register_forward_pre_hook(self.attn_hook(site))
        layers[cfg['probe_layer']].register_forward_hook(self.capture)

    def resid_hook(self, site):
        def hook(module, args, output):
            assert isinstance(output, torch.Tensor)
            return None if self.mode is None else self.mode(site, output)
        return hook

    def attn_hook(self, site):
        def hook(module, args):
            return None if self.mode is None else (self.mode(site, args[0]),)
        return hook

    def capture(self, module, args, output):
        self.captured = output

    def encode(self, texts):
        batch = self.tok(texts, padding=True, return_tensors='pt')
        assert batch['input_ids'].shape[1] <= 1024
        return batch['input_ids'].to(self.dev), batch['attention_mask'].to(self.dev)

    def states(self, ids, mask, mode=None):
        # Token states at the classifier layer, the representation that the classifier and the later layers read.
        self.mode, self.ids, self.mask = mode, ids, mask
        self.backbone(input_ids=ids, attention_mask=mask, use_cache=False)
        self.mode = None
        h, self.captured = self.captured.float(), None
        return h

    def pooled(self, ids, mask, mode=None):
        h = self.states(ids, mask, mode)
        return (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)

    def logprobs(self, ids, mask, mode=None):
        self.mode = mode
        logits = self.full(input_ids=ids, attention_mask=mask, use_cache=False).logits
        self.mode, self.captured = None, None
        return logits.float().log_softmax(-1)


class Probe(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = torch.nn.Linear(dim, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_probe(acts, labels, seed, lr, batch, dev):
    torch.manual_seed(seed)
    probe = Probe(acts.shape[1]).to(dev)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for i in range(0, len(acts), batch):
        x = torch.tensor(acts[i:i + batch], device=dev, dtype=torch.float32)
        y = torch.tensor(labels[i:i + batch], device=dev, dtype=torch.float32)
        opt.zero_grad()
        loss_fn(probe(x), y).backward()
        opt.step()
    return probe


@torch.no_grad()
def probe_scores(probe, acts, dev):
    return probe(torch.tensor(acts, device=dev, dtype=torch.float32)).cpu().numpy()


def accuracies(logits, true, spur):
    pred = (logits > 0).astype(int)
    out = dict(profession=float((pred == true).mean()), gender=float((pred == spur).mean()))
    for t in (0, 1):
        for s in (0, 1):
            sel = (true == t) & (spur == s)
            out[f'group_{t}{s}'] = float((pred[sel] == true[sel]).mean())
    out['worst_group'] = min(out[f'group_{t}{s}'] for t in (0, 1) for s in (0, 1))
    return out


def pick_option(options, source_l0, choice):
    # Options are the average_l0_<k> folders of one width; choices other than the extremes are relative to the
    # average L0 of the source dictionary at the same site.
    l0 = lambda o: int(o.split('_')[-1])
    ranked = sorted(options, key=l0)
    if choice == 'lowest_l0':
        return ranked[0]
    if choice == 'highest_l0':
        return ranked[-1]
    if choice == 'next_lower_l0':
        return [o for o in ranked if l0(o) < source_l0][-1]
    assert choice == 'next_higher_l0', choice
    return [o for o in ranked if l0(o) > source_l0][0]


class Dictionaries:
    def __init__(self, cfg, features, dev):
        rule = json.loads(Path(cfg['l0_rule']).read_text())
        self.sites = list(dict.fromkeys(f['site'] for f in features))
        self.rows = {s: [i for i, f in enumerate(features) if f['site'] == s] for s in self.sites}
        self.src, self.tgt_cpu, self.paths = {}, {}, {}
        for site in self.sites:
            root = Path(cfg['res_root'] if site.startswith('resid') else cfg['att_root'])
            paths = dict(source=root / rule[f'{site}/{cfg["source_width"]}']['sfc_rule'] / 'params.npz')
            entry = rule[f'{site}/{cfg["target_width"]}']
            if cfg['target_choice'] == 'sfc_rule':
                paths['target'] = root / entry['sfc_rule'] / 'params.npz'
            else:
                source_l0 = int(rule[f'{site}/{cfg["source_width"]}']['sfc_rule'].split('_')[-1])
                option = pick_option(entry['options'], source_l0, cfg['target_choice'])
                paths['target'] = root / f'layer_{site.split("_")[1]}/width_{cfg["target_width"]}/{option}/params.npz'
            assert paths['target'] != paths['source'], site
            self.paths[site] = {k: str(v) for k, v in paths.items()}
            p = np.load(paths['source'])
            feats = [features[i]['index'] for i in self.rows[site]]
            self.src[site] = dict(W_enc=torch.tensor(p['W_enc'][:, feats], device=dev),
                                  b_enc=torch.tensor(p['b_enc'][feats], device=dev),
                                  threshold=torch.tensor(p['threshold'][feats], device=dev),
                                  W_dec=torch.tensor(p['W_dec'][feats], device=dev))
            t = np.load(paths['target'])
            self.tgt_cpu[site] = {k: t[k] for k in ['W_enc', 'b_enc', 'threshold', 'W_dec', 'b_dec']}
        self.dev = dev
        self.full = None

    def load_full_target(self):
        self.full = {s: dict(W_enc=torch.tensor(t['W_enc'], device=self.dev, dtype=torch.bfloat16),
                             b_enc=torch.tensor(t['b_enc'], device=self.dev),
                             threshold=torch.tensor(t['threshold'], device=self.dev),
                             W_dec=torch.tensor(t['W_dec'], device=self.dev, dtype=torch.bfloat16),
                             b_dec=torch.tensor(t['b_dec'], device=self.dev)) for s, t in self.tgt_cpu.items()}

    def drop_full_target(self):
        self.full = None
        torch.cuda.empty_cache()

    def target_slice(self, site, idx):
        t = self.tgt_cpu[site]
        return dict(W_enc=torch.tensor(t['W_enc'][:, idx], device=self.dev),
                    b_enc=torch.tensor(t['b_enc'][idx], device=self.dev),
                    threshold=torch.tensor(t['threshold'][idx], device=self.dev),
                    W_dec=torch.tensor(t['W_dec'][idx], device=self.dev))


def source_mode(dicts, q):
    def mode(site, x):
        s, rows = dicts.src[site], dicts.rows[site]
        x32 = x.float()
        z = jumprelu(x32 @ s['W_enc'] + s['b_enc'], s['threshold'])
        return (x32 - (z * q[:, None, rows]) @ s['W_dec']).to(x.dtype)
    return mode


def readout_mode(dicts, q):
    def mode(site, x):
        s, t, rows = dicts.src[site], dicts.full[site], dicts.rows[site]
        zt = jumprelu((x.to(torch.bfloat16) @ t['W_enc']).float() + t['b_enc'], t['threshold'])
        recon = (zt.to(torch.bfloat16) @ t['W_dec']).float() + t['b_dec']
        z = jumprelu(recon @ s['W_enc'] + s['b_enc'], s['threshold'])
        return (x.float() - (z * q[:, None, rows]) @ s['W_dec']).to(x.dtype)
    return mode


class TargetMap:
    # cap bounds the deletion fraction of each target feature; None lets a one-to-one gain exceed one.
    def __init__(self, dicts, idx, weights, cap=1.):
        self.dicts, self.idx, self.weights, self.cap = dicts, idx, weights, cap
        self.slices = {s: dicts.target_slice(s, idx[s]) for s in dicts.sites}

    def mode(self, q, weights=None):
        weights = self.weights if weights is None else weights

        def run(site, x):
            t, rows = self.slices[site], self.dicts.rows[site]
            a = q[:, rows] @ weights[site]
            a = a if self.cap is None else torch.clamp(a, max=self.cap)
            x32 = x.float()
            z = jumprelu(x32 @ t['W_enc'] + t['b_enc'], t['threshold'])
            return (x32 - (z * a[:, None, :]) @ t['W_dec']).to(x.dtype)
        return run

    def export(self):
        return {s: dict(idx=self.idx[s].tolist(), weights=self.weights[s].detach().cpu().numpy().tolist())
                for s in self.dicts.sites}


class PartMap:
    # A rediscovered explanation: each target feature belongs to one named part and is deleted by the fraction of that
    # source part which a request deletes.
    def __init__(self, dicts, red, features):
        assert red['dictionaries'] == dicts.paths
        self.part_rows = {p: [i for i, f in enumerate(features) if f['part'] == p] for p in sorted({f['part'] for f in features})}
        self.slices = {s: dicts.target_slice(s, np.array(v['idx'])) for s, v in red['sites'].items() if v['idx']}
        self.parts = {s: v['part'] for s, v in red['sites'].items() if v['idx']}

    def mode(self, q):
        frac = {p: q[:, rows].mean(1) for p, rows in self.part_rows.items()}

        def run(site, x):
            if site not in self.slices:
                return x
            t = self.slices[site]
            a = torch.stack([frac[p] for p in self.parts[site]], 1)
            z = jumprelu(x.float() @ t['W_enc'] + t['b_enc'], t['threshold'])
            return (x.float() - (z * a[:, None, :]) @ t['W_dec']).to(x.dtype)
        return run


def request_masks(cfg, features):
    n = len(features)
    parts = sorted({f['part'] for f in features})
    masks = {'complete': np.ones(n, dtype=np.float32)}
    for p in parts:
        masks[p] = np.array([f['part'] == p for f in features], dtype=np.float32)
    new = {}
    for band, (lo, hi) in cfg['layer_bands'].items():
        band_mask = np.array([lo <= int(f['site'].split('_')[1]) <= hi for f in features], dtype=np.float32)
        new[f'band_{band}'] = band_mask
        for p in parts:
            m = band_mask * masks[p]
            if m.sum() > 0:
                new[f'band_{band}_{p}'] = m
    rng = np.random.default_rng(cfg['member_seed'])
    for k in range(cfg['member_subsets']):
        m = np.zeros(n, dtype=np.float32)
        m[rng.choice(n, n // 2, replace=False)] = 1.
        new[f'members_{k}'] = m
    masks.update(new)
    families = dict(complete=['complete'], parts=parts, new=list(new))
    return masks, families


def batches(n, size):
    return [list(range(i, min(i + size, n))) for i in range(0, n, size)]


@torch.no_grad()
def collect_pooled(model, texts, batch, mode_fn, run, label):
    out = []
    for k, ix in enumerate(batches(len(texts), batch)):
        ids, mask = model.encode([texts[i] for i in ix])
        out.append(model.pooled(ids, mask, None if mode_fn is None else mode_fn(len(ix))).cpu().numpy())
        if k % 100 == 0:
            run.progress(stage='pooled', method=label, step=f'{k}/{len(texts) // batch + 1}')
    return np.concatenate(out)


@torch.no_grad()
def natural_statistics(model, dicts, texts, cfg, run, candidates=None):
    dev = dicts.dev
    pron = torch.zeros(len(model.tok), dtype=torch.bool, device=dev)
    for tid, piece in enumerate(model.tok.convert_ids_to_tokens(list(range(len(model.tok))))):
        if piece is not None and piece.replace('▁', '').lower() in PRONOUNS:
            pron[tid] = True
    acc = {}
    for site in dicts.sites:
        r = len(dicts.rows[site])
        if candidates is None:
            d = dicts.tgt_cpu[site]['W_enc'].shape[1]
            acc[site] = dict(n=torch.zeros((), dtype=torch.float64, device=dev),
                             zs=torch.zeros(r, dtype=torch.float64, device=dev),
                             zs2=torch.zeros(r, dtype=torch.float64, device=dev),
                             zt=torch.zeros(d, dtype=torch.float64, device=dev),
                             zt2=torch.zeros(d, dtype=torch.float64, device=dev),
                             ft=torch.zeros(d, dtype=torch.float64, device=dev),
                             zst=torch.zeros(r, d, dtype=torch.float64, device=dev),
                             pt=torch.zeros(d, dtype=torch.float64, device=dev),
                             ps=torch.zeros(r, dtype=torch.float64, device=dev),
                             ntok_pron=torch.zeros((), dtype=torch.float64, device=dev),
                             zs_hat=torch.zeros(r, dtype=torch.float64, device=dev),
                             zs_hat2=torch.zeros(r, dtype=torch.float64, device=dev),
                             res=torch.zeros((), dtype=torch.float64, device=dev),
                             xsum=torch.zeros(dicts.src[site]['W_enc'].shape[0], dtype=torch.float64, device=dev),
                             x2=torch.zeros((), dtype=torch.float64, device=dev))
        else:
            u = len(candidates[site])
            acc[site] = dict(n=torch.zeros((), dtype=torch.float64, device=dev),
                             zs2=torch.zeros(r, dtype=torch.float64, device=dev),
                             gram=torch.zeros(u, u, dtype=torch.float64, device=dev),
                             cross=torch.zeros(u, r, dtype=torch.float64, device=dev))
    slices = None if candidates is None else {s: dicts.target_slice(s, candidates[s]) for s in dicts.sites}

    def mode(site, x):
        valid = model.mask.bool().clone()
        valid[:, 0] = False
        flat = x[valid].float()
        is_pron = pron[model.ids[valid]]
        s, a = dicts.src[site], acc[site]
        zs = jumprelu(flat @ s['W_enc'] + s['b_enc'], s['threshold'])
        a['n'] += flat.shape[0]
        if candidates is None:
            t = dicts.full[site]
            zt = jumprelu((flat.to(torch.bfloat16) @ t['W_enc']).float() + t['b_enc'], t['threshold'])
            a['zs'] += zs.sum(0).double()
            a['zs2'] += (zs ** 2).sum(0).double()
            a['zt'] += zt.sum(0).double()
            a['zt2'] += (zt ** 2).sum(0).double()
            a['ft'] += (zt > 0).sum(0).double()
            a['zst'] += (zs.T @ zt).double()
            a['pt'] += zt[is_pron].sum(0).double()
            a['ps'] += zs[is_pron].sum(0).double()
            a['ntok_pron'] += is_pron.sum()
            # Source activation read back from the target reconstruction, and the target's reconstruction error.
            recon = (zt.to(torch.bfloat16) @ t['W_dec']).float() + t['b_dec']
            zs_hat = jumprelu(recon @ s['W_enc'] + s['b_enc'], s['threshold'])
            a['zs_hat'] += (zs * zs_hat).sum(0).double()
            a['zs_hat2'] += (zs_hat ** 2).sum(0).double()
            a['res'] += ((flat - recon) ** 2).sum().double()
            a['xsum'] += flat.sum(0).double()
            a['x2'] += (flat ** 2).sum().double()
        else:
            t = slices[site]
            zt = jumprelu(flat @ t['W_enc'] + t['b_enc'], t['threshold'])
            a['zs2'] += (zs ** 2).sum(0).double()
            a['gram'] += (zt.T @ zt).double()
            a['cross'] += (zt.T @ zs).double()
        return x

    for k, ix in enumerate(batches(len(texts), cfg['natural_batch'])):
        ids, mask = model.encode([texts[i] for i in ix])
        model.pooled(ids, mask, mode)
        if k % 20 == 0:
            run.progress(stage='natural' if candidates is None else 'natural_gram', step=f'{k}/{len(texts) // cfg["natural_batch"] + 1}')
    return {s: {k: v.cpu().numpy() for k, v in a.items()} for s, a in acc.items()}


def exclusive_top(score, per_source):
    chosen, used = [], set()
    for j in range(score.shape[1]):
        picked = []
        for idx in np.argsort(-score[:, j], kind='stable'):
            if int(idx) not in used:
                picked.append(int(idx))
                used.add(int(idx))
            if len(picked) == per_source:
                break
        chosen.append(picked)
    return chosen


def map_from_lists(dicts, lists, dev):
    idx, weights = {}, {}
    for site in dicts.sites:
        union = sorted({t for row in lists[site] for t in row})
        pos = {t: k for k, t in enumerate(union)}
        w = np.zeros((len(lists[site]), len(union)), dtype=np.float32)
        for r, row in enumerate(lists[site]):
            for t in row:
                w[r, pos[t]] = 1.
        idx[site], weights[site] = np.array(union), torch.tensor(w, device=dev)
    return TargetMap(dicts, idx, weights)


def native_fit(dicts, candidates, cand_rows, gram_stats):
    weights, quality = {}, {}
    for site in dicts.sites:
        st = gram_stats[site]
        n = st['n']
        dec_t = dicts.tgt_cpu[site]['W_dec'][candidates[site]].astype(np.float64)
        dec_s = dicts.src[site]['W_dec'].cpu().numpy().astype(np.float64)
        w = np.zeros((len(dicts.rows[site]), len(candidates[site])), dtype=np.float32)
        for r in range(len(dicts.rows[site])):
            k = np.array(cand_rows[site][r])
            Q = st['gram'][np.ix_(k, k)] / n * (dec_t[k] @ dec_t[k].T)
            c = st['cross'][k, r] / n * (dec_t[k] @ dec_s[r])
            base = st['zs2'][r] / n * float(dec_s[r] @ dec_s[r])
            res = minimize(lambda a: (a @ Q @ a - 2 * c @ a, 2 * Q @ a - 2 * c), np.zeros(len(k)), jac=True,
                           method='L-BFGS-B', bounds=[(0., 1.)] * len(k))
            assert res.success, res.message
            w[r, k] = res.x
            quality[f'{site}:{r}'] = float(1 - (res.fun + base) / base)
        weights[site] = w
    return weights, quality


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    run = Run(cfg, args.config, args.resume)
    try:
        execute(cfg, run)
    except BaseException as exc:
        run.status('FAIL', error=f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}')
        raise
    run.status('PASS')


def execute(cfg, run):
    dev = torch.device('cuda:0')
    torch.manual_seed(cfg['seed'])
    assert cfg['subset_inclusion'] in ('half', 'uniform'), cfg['subset_inclusion']
    manifest = json.loads(Path(cfg['features']).read_text())
    features = [dict(site=site, index=int(idx), part=part, comment=comment)
                for site, rows in manifest['features'].items() for idx, part, comment in rows]
    assert len(features) == cfg['expected_features'], len(features)
    masks, families = request_masks(cfg, features)
    (run.dir / 'requests.json').write_text(json.dumps(dict(features=features, masks={k: v.tolist() for k, v in masks.items()},
                                                           families=families), indent=1))
    dicts = Dictionaries(cfg, features, dev)
    (run.dir / 'dictionaries.json').write_text(json.dumps(dicts.paths, indent=1))
    model = Model(cfg, dicts.sites, dev)
    run.progress(stage='loaded', sites=len(dicts.sites), features=len(features))

    train, test = read_split(cfg, 'train'), read_split(cfg, cfg['eval_split'])
    amb_train, amb_train_y, _ = sfc_ambiguous(train, cfg['data_seed'])
    bal_test, bal_true, bal_spur = sfc_balanced(test, cfg['data_seed'])
    bal_train, bal_train_true, _ = sfc_balanced(train, cfg['data_seed'])
    if cfg.get('limit_use'):
        amb_train, amb_train_y = amb_train[:cfg['limit_use']], amb_train_y[:cfg['limit_use']]
        bal_test, bal_true, bal_spur = bal_test[:cfg['limit_use']], bal_true[:cfg['limit_use']], bal_spur[:cfg['limit_use']]
        bal_train, bal_train_true = bal_train[:cfg['limit_use']], bal_train_true[:cfg['limit_use']]
    text, prof, _ = train
    natural = [x for x, p in zip(text, prof) if p not in (PROFESSOR, NURSE)]
    random.Random(cfg['data_seed'] + 1).shuffle(natural)
    natural = natural[:cfg['natural_bios']]
    fit_texts = amb_train[:cfg['fit_bios']]
    eval_texts = bal_test[:cfg['eval_bios']]
    eval_true, eval_spur = bal_true[:cfg['eval_bios']], bal_spur[:cfg['eval_bios']]
    run.records([dict(kind='data', ambiguous_train=len(amb_train), balanced_test=len(bal_test), balanced_train=len(bal_train),
                      natural_bios=len(natural), fit_bios=len(fit_texts), eval_bios=len(eval_texts))])

    # Classifier recipe of the SFC notebook, trained on clean ambiguous biographies.
    probe_path = run.dir / 'probe.pt'
    clean_train = collect_pooled(model, amb_train, cfg['use_batch'], None, run, 'clean_train')
    probe = train_probe(clean_train, amb_train_y, cfg['probe_seed'], cfg['probe_lr'], cfg['probe_batch'], dev)
    torch.save(probe.state_dict(), probe_path)
    probe.requires_grad_(False)
    clean_test = collect_pooled(model, bal_test, cfg['use_batch'], None, run, 'clean_test')
    oracle = train_probe(collect_pooled(model, bal_train, cfg['use_batch'], None, run, 'clean_balanced_train'),
                         bal_train_true, cfg['probe_seed'], cfg['probe_lr'], cfg['probe_batch'], dev)
    use = dict(clean=dict(original=accuracies(probe_scores(probe, clean_test, dev), bal_true, bal_spur)),
               oracle=dict(original=accuracies(probe_scores(oracle, clean_test, dev), bal_true, bal_spur)))
    run.records([dict(kind='use', method=k, **v['original']) for k, v in use.items()])
    del clean_train

    # Label-free statistics on natural biographies of other professions.
    dicts.load_full_target()
    nat = natural_statistics(model, dicts, natural, cfg, run)
    cos_lists, act_lists, top2_lists, candidates, cand_rows, corr_store = {}, {}, {}, {}, {}, {}
    cap_candidates, cap_allowed, cap_init, capacity_stats = {}, {}, {}, {}
    crng = np.random.default_rng(cfg['seed'] + 11)
    for site in dicts.sites:
        st = nat[site]
        n = st['n']
        dec_t = dicts.tgt_cpu[site]['W_dec']
        dec_t = dec_t / np.linalg.norm(dec_t, axis=1, keepdims=True)
        dec_s = dicts.src[site]['W_dec'].cpu().numpy()
        dec_s = dec_s / np.linalg.norm(dec_s, axis=1, keepdims=True)
        cos = dec_s @ dec_t.T
        ms, mt = st['zs'] / n, st['zt'] / n
        sd_s = np.sqrt(np.maximum(st['zs2'] / n - ms ** 2, 0))
        sd_t = np.sqrt(np.maximum(st['zt2'] / n - mt ** 2, 0))
        corr = (st['zst'] / n - ms[:, None] * mt[None, :]) / np.maximum(sd_s[:, None] * sd_t[None, :], 1e-12)
        corr[:, sd_t == 0] = 0.
        rows_c, cols_c = linear_sum_assignment(-cos)
        rows_a, cols_a = linear_sum_assignment(-corr)
        cos_lists[site] = [[int(cols_c[list(rows_c).index(r)])] for r in range(cos.shape[0])]
        act_lists[site] = [[int(cols_a[list(rows_a).index(r)])] for r in range(corr.shape[0])]
        top2_lists[site] = exclusive_top(cos.T, 2)
        per_row = [sorted(set(np.argsort(-cos[r])[:cfg['candidates_cos']].tolist())
                          | set(np.argsort(-corr[r])[:cfg['candidates_corr']].tolist())) for r in range(cos.shape[0])]
        union = sorted({t for row in per_row for t in row})
        pos = {t: k for k, t in enumerate(union)}
        candidates[site] = np.array(union)
        cand_rows[site] = [[pos[t] for t in row] for row in per_row]
        if any('capacity' in f for f in cfg['fit_families']):
            # Capacity control: each candidate of each source feature is replaced by a random target feature from the
            # same activation-frequency bin, outside every real candidate, so the map keeps its size and firing rates
            # but loses the features chosen for the explanation. The replacement of the Hungarian match starts at one.
            rate = st['ft'] / n
            bins = np.empty(len(rate), dtype=int)
            bins[np.argsort(rate, kind='stable')] = np.arange(len(rate)) * cfg['capacity_bins'] // len(rate)
            free = ~np.isin(np.arange(len(rate)), union)
            pools = [np.flatnonzero((bins == b) & free) for b in range(cfg['capacity_bins'])]
            rep_rows, rep_match = [], []
            for r, row in enumerate(per_row):
                chosen = {}
                for t in row:
                    chosen[t] = int(crng.choice([u for u in pools[bins[t]] if u not in chosen.values()]))
                rep_rows.append(list(chosen.values()))
                rep_match.append(chosen[cos_lists[site][r][0]])
            cap_union = sorted({u for row in rep_rows for u in row})
            cpos = {t: k for k, t in enumerate(cap_union)}
            cap_candidates[site] = np.array(cap_union)
            cap_allowed[site] = torch.zeros(len(per_row), len(cap_union), device=dev)
            cap_init[site] = np.zeros((len(per_row), len(cap_union)), dtype=np.float32)
            for r in range(len(per_row)):
                cap_allowed[site][r, [cpos[u] for u in rep_rows[r]]] = 1.
                cap_init[site][r, cpos[rep_match[r]]] = 1.
            capacity_stats[site] = dict(candidates=len(cap_union), real_candidates=len(union),
                                        real_rate=float(rate[union].mean()), replacement_rate=float(rate[cap_union].mean()))
        corr_store[site] = dict(cos=cos[:, union], corr=corr[:, union], pron_share_target=(st['pt'] / np.maximum(st['zt'], 1e-12))[union],
                                pron_share_source=st['ps'] / np.maximum(st['zs'], 1e-12),
                                pron_token_share=float(st['ntok_pron'] / n), tokens=float(n),
                                hungarian_decoder_cos=[float(cos[r, cos_lists[site][r][0]]) for r in range(cos.shape[0])],
                                hungarian_activation_corr=[float(corr[r, act_lists[site][r][0]]) for r in range(cos.shape[0])],
                                pron_share_all_target=st['pt'] / np.maximum(st['zt'], 1e-12),
                                source_retention=st['zs_hat'] / np.maximum(st['zs2'], 1e-12),
                                # Squared error of the source activation read back from the target reconstruction.
                                source_readback_error=(st['zs2'] - 2 * st['zs_hat'] + st['zs_hat2']) / np.maximum(st['zs2'], 1e-12),
                                target_fve=float(1 - st['res'] / (st['x2'] - (st['xsum'] ** 2).sum() / n)))
    np.savez_compressed(run.dir / 'natural_statistics.npz', **{f'{s}__{k}': np.asarray(v) for s, d in corr_store.items()
                                                                for k, v in d.items()}, **{f'{s}__candidates': candidates[s] for s in dicts.sites},
                        **{f'{s}__capacity_candidates': cap_candidates[s] for s in cap_candidates})
    if capacity_stats:
        (run.dir / 'CAPACITY.json').write_text(json.dumps(dict(bins=cfg['capacity_bins'], sites=capacity_stats), indent=1))
    del nat
    gram = natural_statistics(model, dicts, natural, cfg, run, candidates)
    native_w, native_quality = native_fit(dicts, candidates, cand_rows, gram)
    run.records([dict(kind='native_quality', key=k, explained=v) for k, v in native_quality.items()])

    maps = dict(hungarian_decoder=map_from_lists(dicts, cos_lists, dev),
                hungarian_activation=map_from_lists(dicts, act_lists, dev),
                top2_decoder=map_from_lists(dicts, top2_lists, dev),
                native=TargetMap(dicts, candidates, {s: torch.tensor(native_w[s], device=dev) for s in dicts.sites}))
    allowed = {s: torch.zeros(len(dicts.rows[s]), len(candidates[s]), device=dev) for s in dicts.sites}
    for s in dicts.sites:
        for r, row in enumerate(cand_rows[s]):
            allowed[s][r, row] = 1.

    # Source responses on the fitting and evaluation biographies.
    names = list(masks)
    qmat = torch.tensor(np.stack([masks[k] for k in names]), device=dev)

    @torch.no_grad()
    def clean_logits(texts):
        return np.concatenate([probe(model.pooled(*model.encode([texts[i] for i in ix]))).cpu().numpy()
                               for ix in batches(len(texts), cfg['eval_batch'])])

    @torch.no_grad()
    def responses(texts, base, make_mode, label, request_names):
        rows = [(qi, bi) for qi in range(len(request_names)) for bi in range(len(texts))]
        out = np.zeros((len(request_names), len(texts)), dtype=np.float32)
        qsel = torch.tensor(np.stack([masks[k] for k in request_names]), device=dev)
        for k, ix in enumerate(batches(len(rows), cfg['eval_batch'])):
            q = qsel[[rows[i][0] for i in ix]]
            ids, mask = model.encode([texts[rows[i][1]] for i in ix])
            logit = probe(model.pooled(ids, mask, make_mode(q))).cpu().numpy()
            for j, i in enumerate(ix):
                out[rows[i][0], rows[i][1]] = logit[j] - base[rows[i][1]]
            if k % 100 == 0:
                run.progress(stage='responses', method=label, step=f'{k}/{len(rows) // cfg["eval_batch"] + 1}')
        return out

    fit_base, eval_base = clean_logits(fit_texts), clean_logits(eval_texts)
    train_requests = families['complete'] + families['parts']
    fit_source = responses(fit_texts, fit_base, lambda q: source_mode(dicts, q), 'source_fit', train_requests)
    norm = (fit_source ** 2).mean(1)
    run.records([dict(kind='fit_source_energy', request=r, mean_square=float(v)) for r, v in zip(train_requests, norm)])

    hung_w = {}
    for s in dicts.sites:
        w = np.zeros((len(dicts.rows[s]), len(candidates[s])), dtype=np.float32)
        for r, row in enumerate(cos_lists[s]):
            pos = np.flatnonzero(candidates[s] == row[0])
            assert len(pos) == 1 and pos[0] in cand_rows[s][r], (s, r)
            w[r, pos[0]] = 1.
        hung_w[s] = w
    inits = dict(native=native_w, hungarian=hung_w)
    # One-to-one control: the Hungarian matches with gains trained by part supervision, same data and updates.
    matched_only = {s: torch.tensor((hung_w[s] > 0).astype(np.float32), device=dev) for s in dicts.sites}
    # Gains of one-to-one maps may exceed one when matched_gain_max is set, so a match can delete more than itself.
    gain_max = cfg.get('matched_gain_max', 1.)
    natural_families = {'natural', 'natural_matched', 'natural_capacity'} & set(cfg['fit_families'])
    assert not (natural_families or cfg.get('natural_lm_texts')) or cfg.get('full_model'), 'natural text needs full_model'
    if natural_families or cfg.get('natural_lm_texts'):
        corpus = json.loads(Path(cfg['natural_lm_texts']).read_text())
        nat_ids, nat_mask = {}, {}
        for split in ('train', 'eval'):
            enc = model.tok(corpus[split], truncation=True, max_length=cfg['natural_lm_len'], padding='max_length',
                            return_tensors='pt')
            nat_ids[split], nat_mask[split] = enc['input_ids'].to(dev), enc['attention_mask'].to(dev)
        run.records([dict(kind='natural_lm_data', train=len(corpus['train']), eval=len(corpus['eval']),
                          source_sha256=corpus['source_sha256'], length=cfg['natural_lm_len'])])

    whiten = None
    if any(f.endswith(('_state', '_statemixed')) for f in cfg['fit_families']):
        # Reader-complete families fit the layer-22 token states that every downstream reader sees. Their error is
        # whitened by the covariance of clean states on unlabeled biographies of other professions, either with the
        # Ledoit-Wolf estimate or by standardizing each dimension. The first position is left out everywhere.
        chunks = []
        with torch.no_grad():
            for ix in batches(min(len(natural), cfg['whiten_bios']), cfg['natural_batch']):
                ids, mask = model.encode([natural[i] for i in ix])
                valid = mask.bool().clone()
                valid[:, 0] = False
                chunks.append(model.states(ids, mask)[valid].cpu().numpy())
        states = np.concatenate(chunks)
        states = states[np.random.default_rng(cfg['seed']).permutation(len(states))[:cfg['whiten_tokens']]]
        if cfg['state_reader'] == 'ledoit_wolf':
            lw = LedoitWolf().fit(states)
            vals, vecs = np.linalg.eigh(lw.covariance_)
            w = vecs @ np.diag(vals ** -.5) @ vecs.T
            info = dict(shrinkage=float(lw.shrinkage_), eigenvalue_min=float(vals.min()), eigenvalue_max=float(vals.max()))
        else:
            assert cfg['state_reader'] == 'standardized', cfg['state_reader']
            w = np.diag(1. / states.std(0))
            info = dict(std_min=float(states.std(0).min()), std_max=float(states.std(0).max()))
        whiten = torch.tensor(w, device=dev, dtype=torch.float32)
        (run.dir / 'WHITENING.json').write_text(json.dumps(dict(reader=cfg['state_reader'], tokens=len(states), **info), indent=1))
        del states, chunks

    def state_loss(ids, mask, q, tmap, weights, requests):
        # Whitened error of the layer-22 token states against the source edit, normalized per request by the mean
        # source effect energy of its rows in the batch; all drawn subsets share one normalizer.
        m = mask.float().clone()
        m[:, 0] = 0.
        with torch.no_grad():
            h0 = model.states(ids, mask)
            hs = model.states(ids, mask, source_mode(dicts, q))
            eff = ((((hs - h0) @ whiten) ** 2).sum(-1) * m).sum(1)
        ht = model.states(ids, mask, tmap.mode(q, weights))
        err = ((((ht - hs) @ whiten) ** 2).sum(-1) * m).sum(1)
        groups = {}
        for k, r in enumerate(requests):
            groups.setdefault('drawn' if r in ('random', 'single') else r, []).append(k)
        scale = torch.empty_like(eff)
        for ix in groups.values():
            scale[ix] = eff[ix].mean().clamp_min(1e-8)
        return (err / scale).mean()

    def natural_q(n, rng):
        # Each sequence draws an inclusion probability uniformly and each feature joins with it.
        draw = rng.random((n, len(features))) < rng.random((n, 1))
        draw[np.arange(n), rng.integers(0, len(features), n)] |= ~draw.any(1)
        return torch.tensor(draw.astype(np.float32), device=dev)

    def natural_error(ids, mask, q, tmap, weights):
        # Probability-weighted squared error of next-token log-probability changes, against the source edit.
        with torch.no_grad():
            lp0 = model.logprobs(ids, mask)
            ds = model.logprobs(ids, mask, source_mode(dicts, q)) - lp0
            w = lp0.exp() * mask[..., None]
        dt = model.logprobs(ids, mask, tmap.mode(q, weights)) - lp0
        return (w * (dt - ds) ** 2).sum(), (w * ds ** 2).sum()

    def train_natural(family, init, seed):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        matched = family == 'natural_matched'
        capacity = family == 'natural_capacity'
        mask_of = matched_only if matched else (cap_allowed if capacity else allowed)
        hi = gain_max if matched else 1.
        cand_of, init_of = (cap_candidates, cap_init) if capacity else (candidates, inits[init])
        params = {s: torch.nn.Parameter(torch.tensor(init_of[s], device=dev)) for s in dicts.sites}
        opt = torch.optim.Adam(params.values(), lr=cfg['fit_lr'])
        tmap = TargetMap(dicts, cand_of, {s: params[s] for s in dicts.sites}, cap=None if hi > 1 else 1.)
        curve = []
        for step in range(cfg['natural_steps']):
            ix = rng.choice(len(nat_ids['train']), cfg['natural_lm_batch'], replace=False)
            q = natural_q(len(ix), rng)
            weights = {s: params[s] * mask_of[s] for s in dicts.sites}
            err, eff = natural_error(nat_ids['train'][ix], nat_mask['train'][ix], q, tmap, weights)
            fit_loss = err / eff.clamp_min(1e-12)
            l1 = sum(w.sum() for w in weights.values())
            loss = fit_loss + cfg['fit_l1'] * l1
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for p in params.values():
                    p.clamp_(0., hi)
            curve.append(dict(step=step, fit_loss=float(fit_loss), l1=float(l1)))
            if step % 25 == 0 or step == cfg['natural_steps'] - 1:
                run.progress(stage='fit', method=f'{family}_{init}_s{seed}', step=step, fit_loss=round(float(fit_loss), 5),
                             l1=round(float(l1), 2))
        run.records([dict(kind='fit_curve', family=family, init=init, seed=seed, **c) for c in curve])
        return TargetMap(dicts, cand_of, {s: (params[s] * mask_of[s]).detach() for s in dicts.sites},
                         cap=None if hi > 1 else 1.)

    @torch.no_grad()
    def natural_eval(tmap):
        rng = np.random.default_rng(cfg['natural_eval_seed'])
        num = den = 0.
        for ix in batches(len(nat_ids['eval']), cfg['natural_lm_batch']):
            q = natural_q(len(ix), rng)
            err, eff = natural_error(nat_ids['eval'][ix], nat_mask['eval'][ix], q, tmap, tmap.weights)
            num += float(err)
            den += float(eff)
        return float(np.sqrt(num / den))

    def train_map(family, init, seed):
        if family in ('natural', 'natural_matched', 'natural_capacity'):
            return train_natural(family, init, seed)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        # A family ending in _state fits the layer-22 token states with the same requests, biographies and draws as
        # the family it extends, which fits the classifier response. A capacity_ prefix uses the frequency-matched
        # random candidates, a matched_ prefix keeps only the Hungarian match of each source feature.
        # A family ending in _statemixed alternates these biography steps with generic-text steps of random subsets,
        # both read by the whitened layer-22 states, with the biography step count of the family it extends.
        mixed = family.endswith('_statemixed')
        reader = 'state' if family.endswith(('_state', '_statemixed')) else 'classifier'
        base = family[:family.rindex('_')] if reader == 'state' else family
        capacity = base.startswith('capacity_')
        base = base[len('capacity_'):] if capacity else base
        one_to_one = base.startswith('matched_') or base == 'parts_matched'
        base = base[len('matched_'):] if base.startswith('matched_') else base
        cand_of, init_of = (cap_candidates, cap_init) if capacity else (candidates, inits[init])
        params = {s: torch.nn.Parameter(torch.tensor(init_of[s], device=dev)) for s in dicts.sites}
        opt = torch.optim.Adam(params.values(), lr=cfg['fit_lr'])
        # The subset family adds one random half of the features per biography, whose source response is computed live.
        # The label-free families keep the complete request and replace the two part rows by two rows of random feature
        # subsets of uniformly drawn size, or of single features, so they see no annotation at equal budget.
        req = dict(complete=['complete'], parts=train_requests, subsets=train_requests + ['random'],
                   parts_matched=train_requests, random=['complete', 'random', 'random'],
                   singletons=['complete', 'single', 'single'])[base]
        label_free = base in ('random', 'singletons')
        mask_of = matched_only if one_to_one else (cap_allowed if capacity else allowed)
        hi = gain_max if one_to_one else 1.
        per = cfg['fit_sequences'] // len(req)
        # The training forward uses the same deletion cap as the returned map: none when one-to-one gains may exceed one.
        tmap = TargetMap(dicts, cand_of, {s: params[s] for s in dicts.sites}, cap=None if hi > 1 else 1.)
        curve = []
        steps = cfg['fit_steps'] * (2 if mixed else 1)

        def update(fit_loss, weights, step):
            l1 = sum(w.sum() for w in weights.values())
            loss = fit_loss + cfg['fit_l1'] * l1
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for p in params.values():
                    p.clamp_(0., hi)
            curve.append(dict(step=step, fit_loss=float(fit_loss), l1=float(l1)))
            if step % 25 == 0 or step == steps - 1:
                run.progress(stage='fit', method=f'{family}_{init}_s{seed}', step=step, fit_loss=round(float(fit_loss), 5),
                             l1=round(float(l1), 2))

        for step in range(steps):
            if mixed and step % 2 == 0:
                ix = rng.choice(len(nat_ids['train']), cfg['natural_lm_batch'], replace=False)
                q = natural_q(len(ix), rng)
                weights = {s: params[s] * mask_of[s] for s in dicts.sites}
                update(state_loss(nat_ids['train'][ix], nat_mask['train'][ix], q, tmap, weights, ['random'] * len(ix)),
                       weights, step)
                continue
            bios = rng.choice(len(fit_texts), per, replace=False)
            rows = [(r, b) for r in req for b in bios]
            if label_free:
                # Every row draws its own subset; a singleton row starts empty and receives exactly one feature.
                inclusion = rng.random((len(rows), 1)) if base == 'random' else 0.
                draw = rng.random((len(rows), len(features))) < inclusion
                draw[np.arange(len(rows)), rng.integers(0, len(features), len(rows))] |= ~draw.any(1)
                random_q = torch.tensor(draw.astype(np.float32), device=dev)
                q = torch.stack([random_q[k] if r in ('random', 'single') else qmat[names.index(r)]
                                 for k, (r, _) in enumerate(rows)])
            else:
                # Each feature joins with probability one half, or with a per-row probability drawn uniformly so that
                # small subsets, whose responses do not saturate, also appear.
                inclusion = .5 if cfg['subset_inclusion'] == 'half' else rng.random((per, 1))
                draw = rng.random((per, len(features))) < inclusion
                draw[np.arange(per), rng.integers(0, len(features), per)] |= ~draw.any(1)
                random_q = torch.tensor(draw.astype(np.float32), device=dev)
                q = torch.stack([random_q[k % per] if r == 'random' else qmat[names.index(r)]
                                 for k, (r, _) in enumerate(rows)])
            ids, mask = model.encode([fit_texts[b] for _, b in rows])
            weights = {s: params[s] * mask_of[s] for s in dicts.sites}
            if reader == 'state':
                fit_loss = state_loss(ids, mask, q, tmap, weights, [r for r, _ in rows])
            else:
                drawn = ('random', 'single')
                target = np.array([np.nan if r in drawn else fit_source[train_requests.index(r), b] for r, b in rows],
                                  dtype=np.float32)
                live = [k for k, (r, _) in enumerate(rows) if r in drawn]
                if live:
                    with torch.no_grad():
                        src = probe(model.pooled(ids[live], mask[live], source_mode(dicts, q[live]))).cpu().numpy()
                    target[live] = src - fit_base[[rows[k][1] for k in live]]
                scale = np.array([norm[train_requests.index('complete' if r in drawn else r)] for r, _ in rows])
                if label_free:
                    # Drawn rows are normalized by their own mean source energy in the batch, as each part row is by its
                    # part energy, so small subsets keep the same weight as the complete request.
                    scale[live] = max(float(np.mean(target[live] ** 2)), 1e-8)
                logit = probe(model.pooled(ids, mask, tmap.mode(q, weights)))
                delta = logit - torch.tensor(fit_base[[b for _, b in rows]], device=dev)
                fit_loss = ((delta - torch.tensor(target, device=dev)) ** 2 / torch.tensor(scale, device=dev)).mean()
            update(fit_loss, weights, step)
        run.records([dict(kind='fit_curve', family=family, init=init, seed=seed, **c) for c in curve])
        return TargetMap(dicts, cand_of, {s: (params[s] * mask_of[s]).detach() for s in dicts.sites},
                         cap=None if hi > 1 else 1.)

    labels = dict(complete='complete_sup', parts='part_sup', subsets='subset_sup', parts_matched='matched_part_sup',
                  random='random_sup', singletons='singleton_sup', natural='natural_sup', natural_matched='natural_matched',
                  natural_capacity='natural_capacity', subsets_state='subset_state_sup', random_state='random_state_sup',
                  matched_subsets_state='matched_subset_state_sup', capacity_subsets_state='capacity_subset_state_sup',
                  subsets_statemixed='subset_statemixed_sup')
    for init in cfg['fit_inits']:
        for seed in cfg['fit_seeds']:
            for family in cfg['fit_families']:
                maps[f'{labels[family]}_{init}_s{seed}'] = train_map(family, init, seed)
    (run.dir / 'maps.json').write_text(json.dumps({k: m.export() for k, m in maps.items()}))

    # Response fidelity on balanced test biographies for every request family.
    eval_resp = {'source': responses(eval_texts, eval_base, lambda q: source_mode(dicts, q), 'source', names),
                 'readout': responses(eval_texts, eval_base, lambda q: readout_mode(dicts, q), 'readout', names)}
    for name, tmap in maps.items():
        eval_resp[name] = responses(eval_texts, eval_base, tmap.mode, name, names)
    np.savez_compressed(run.dir / 'responses.npz', request_names=np.array(names), eval_true=eval_true, eval_spur=eval_spur,
                        eval_base=eval_base, **eval_resp)
    if cfg.get('natural_lm_texts'):
        # Natural-text fidelity of every map under random feature subsets on held-out generic text.
        natural_errors = {}
        for name, tmap in maps.items():
            natural_errors[name] = natural_eval(tmap)
            run.progress(stage='natural_eval', method=name, error=round(natural_errors[name], 4))
        (run.dir / 'NATURAL.json').write_text(json.dumps(natural_errors, indent=1))
    summary = {}
    src = eval_resp['source']
    for method, resp in eval_resp.items():
        if method == 'source':
            continue
        row = {}
        for fam, reqs in families.items():
            ix = [names.index(r) for r in reqs]
            row[fam] = float(np.sqrt(((resp[ix] - src[ix]) ** 2).sum() / (src[ix] ** 2).sum()))
            row[f'{fam}_effect_ratio'] = float(np.sqrt((resp[ix] ** 2).sum() / (src[ix] ** 2).sum()))
        summary[method] = row
    run.records([dict(kind='nrmse', method=k, **v) for k, v in summary.items()])
    (run.dir / 'NRMSE.json').write_text(json.dumps(summary, indent=1))

    # SHIFT use with the complete request: original classifier and a classifier retrained after deletion.
    complete = torch.tensor(masks['complete'], device=dev)
    executors = {'source': lambda b: source_mode(dicts, complete.expand(b, -1)),
                 'readout': lambda b: readout_mode(dicts, complete.expand(b, -1))}
    for name, tmap in maps.items():
        executors[name] = (lambda m: lambda b: m.mode(complete.expand(b, -1)))(tmap)
    if cfg['skip_use']:
        executors = {}
    pooled_dir = run.dir / 'pooled'
    pooled_dir.mkdir(exist_ok=True)
    for name, make in executors.items():
        test_acts = collect_pooled(model, bal_test, cfg['use_batch'], make, run, f'use_test_{name}')
        train_acts = collect_pooled(model, amb_train, cfg['use_batch'], make, run, f'use_train_{name}')
        retrained = train_probe(train_acts, amb_train_y, cfg['probe_seed'], cfg['probe_lr'], cfg['probe_batch'], dev)
        use[name] = dict(original=accuracies(probe_scores(probe, test_acts, dev), bal_true, bal_spur),
                         retrained=accuracies(probe_scores(retrained, test_acts, dev), bal_true, bal_spur))
        np.savez_compressed(pooled_dir / f'{name}.npz', test=test_acts.astype(np.float16), train=train_acts.astype(np.float16))
        run.records([dict(kind='use', method=name, classifier=k, **v) for k, v in use[name].items()])
        (run.dir / 'USE.json').write_text(json.dumps(use, indent=1))
    dicts.drop_full_target()
    counts = {k: {s: int((m.weights[s] > cfg['active_weight']).any(0).sum()) for s in dicts.sites} for k, m in maps.items()}
    result = dict(written_utc=now(), nrmse=summary, use=use, families=families,
                  natural=json.loads((run.dir / 'NATURAL.json').read_text()) if cfg.get('natural_lm_texts') else None,
                  target_features=dict((k, sum(v.values())) for k, v in counts.items()),
                  native_quality=native_quality, natural_tokens=float(sum(corr_store[s]['tokens'] for s in dicts.sites) / len(dicts.sites)))
    (run.dir / 'RESULTS.json').write_text(json.dumps(result, indent=1))
    run.progress(stage='done')


if __name__ == '__main__':
    main()
