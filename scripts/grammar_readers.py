import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers

from cross_seed_transfer import Side, margins

DESCRIPTION = ('Held-out readers for saved grammar correspondences: every map, the source edit and the clean model on '
               'BLiMP paradigms that no correspondence or explanation selection used, read by each paradigm\'s '
               'good-minus-bad margin and by the full next-token distribution of its sentences from the edited position '
               'on, plus the next-token distribution of held-out generic text.')


def pairs_of(path, tok, prefix, eos, count):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if len(rows) == count:
                break
            r = json.loads(line)
            good = prefix + tok.encode(r['sentence_good'], add_special_tokens=False) + [eos]
            bad = prefix + tok.encode(r['sentence_bad'], add_special_tokens=False) + [eos]
            diff = next(i for i, (a, b) in enumerate(zip(good, bad)) if a != b)
            rows.append(dict(good=good, bad=bad, position=diff - 1, task=r['UID'], row_id=int(r['pairID'])))
    return rows


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    out = Path(c['run_root']) / c['run_id']
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.resolved.json').write_text(json.dumps(c, indent=1))
    (out / 'status.json').write_text(json.dumps(dict(status='RUNNING', started_at_utc=datetime.now(timezone.utc).isoformat())))
    (out / 'code').mkdir()
    (out / 'code' / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    torch.manual_seed(0)
    torch.set_float32_matmul_precision('highest')
    device = torch.device(c['device'])
    sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
    from dictionary_learning.trainers.top_k import AutoEncoderTopK

    class Log:
        def __init__(self):
            self.t0 = time.time()

        def checked(self, path, label=''):
            return path

        def progress(self, **kw):
            row = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), elapsed=round(time.time() - self.t0, 1), **kw)
            (out / 'progress.json').write_text(json.dumps(row))
            with (out / 'progress.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
    log = Log()
    source = Side(log, c['source'], AutoEncoderTopK, device)
    target = Side(log, c['target'], AutoEncoderTopK, device, source.model)
    tok = transformers.AutoTokenizer.from_pretrained(c['source']['model_dir'], local_files_only=True)
    eos = c['eos_token_id']
    prefix = tok.encode(c['sentence_prefix'], add_special_tokens=False) if c.get('sentence_prefix') else []
    gate = torch.tensor(np.load(c['source']['gate'])['gate'], device=device)
    members = gate.sum(1).nonzero().flatten()
    parts = gate[members].argmax(1)
    m = len(members)
    dec_s = source.sae.decoder.weight[:, members].T
    executors = {}
    for spec in c['executors']:
        z = np.load(spec['maps'])
        assert np.array_equal(z['members'], members.cpu().numpy()), spec
        cand = torch.tensor(z['candidates'], device=device)
        executors[spec['name']] = (cand, target.sae.decoder.weight[:, cand].T,
                                   torch.tensor(z[spec['key']], device=device, dtype=torch.float32), spec['cap'])

    names = list(c['part_names'])
    masks = {'complete': torch.ones(m, device=device)}
    masks.update({p: (parts == k).float() for k, p in enumerate(names)})
    rng = np.random.default_rng(c['random_seed'])
    randoms = []
    for k in range(c['random_requests']):
        draw = rng.random(m) < rng.random()
        draw[rng.integers(m)] |= not draw.any()
        masks[f'random_{k}'] = torch.tensor(draw.astype(np.float32), device=device)
        randoms.append(f'random_{k}')
    families = dict(complete=['complete'], parts=names, random=randoms)
    requests = [r for f in ['complete', 'parts', 'random'] for r in families[f]]
    (out / 'requests.json').write_text(json.dumps(dict(families=families, part_names=names), indent=1))

    def edit_for(name, q):
        if name == 'source':
            def edit(h):
                qq = q[:, None, :] if q.dim() == 2 else q
                return -(source.sae.encode(h)[..., members] * qq) @ dec_s
            return source, edit
        cand, dec, weights, cap = executors[name]

        def edit(h):
            load = q @ weights.T
            load = load if cap is None else load.clamp(max=cap)
            load = load[:, None, :] if q.dim() == 2 else load
            return -(target.sae.encode(h)[..., cand] * load) @ dec
        return target, edit

    pairs = [r for path in c['heldout_paradigms'] for r in pairs_of(path, tok, prefix, eos, c['pairs_per_paradigm'])]
    tasks = np.array([r['task'] for r in pairs])
    tokens = np.memmap(c['natural_eval_tokens'], dtype=c.get('token_dtype', '<u2'), mode='r').reshape(-1, c['context'])
    # Confirmation readers start after the documents that development readers and transfer evaluations already used.
    doff = c.get('natural_doc_offset', 0)
    docs = np.asarray(tokens[doff:doff + c['natural_docs']], dtype=np.int64)
    assert len(docs) == c['natural_docs'], (doff, len(docs))
    execs = ['source'] + list(executors)
    store = {}

    def put(key, shape, ri, ix, value):
        if key not in store:
            store[key] = np.full(shape, np.nan, dtype=np.float32)
        store[key][ri, ix] = value

    @torch.no_grad()
    def sentence_lp(side, rr, edit):
        length = max(len(r[k]) for r in rr for k in ['good', 'bad'])
        ids = torch.full((2 * len(rr), length), eos, dtype=torch.long, device=device)
        pad = torch.zeros_like(ids)
        for i, row in enumerate(rr):
            for j, k in enumerate(['good', 'bad']):
                ids[2 * i + j, :len(row[k])] = torch.tensor(row[k], device=device)
                pad[2 * i + j, :len(row[k])] = 1
        positions = torch.tensor([r['position'] for r in rr], device=device).repeat_interleave(2)
        keep = ((torch.arange(length, device=device)[None, :] >= positions[:, None]) & pad.bool()).float()
        side.positions, side.edit = positions, edit
        lp = side.logprobs(ids, pad)
        side.positions = side.edit = None
        return lp, keep

    batches = [list(range(i, min(i + c['batch_pairs'], len(pairs)))) for i in range(0, len(pairs), c['batch_pairs'])]
    for bi, ix in enumerate(batches):
        rr = [pairs[i] for i in ix]
        with torch.no_grad():
            base = margins(source, rr, eos).cpu().numpy()
            lp0, keep = sentence_lp(source, rr, None)
        put('pair|none|margin', (1, len(pairs)), 0, ix, base)
        w0 = lp0.exp() * keep[..., None]
        for ri, req in enumerate(requests):
            # One request vector for every sentence of the batch.
            q = masks[req]
            side, edit = edit_for('source', q)
            with torch.no_grad():
                side.edit = edit
                ms = margins(side, rr, eos).cpu().numpy()
                side.edit = None
            lps, _ = sentence_lp(source, rr, edit)
            ds = lps - lp0
            eff = (w0 * ds ** 2).sum((1, 2)).reshape(-1, 2).sum(1)
            for name in execs:
                if name == 'source':
                    me, lpe = ms, lps
                else:
                    side, edit = edit_for(name, q)
                    with torch.no_grad():
                        side.edit = edit
                        me = margins(side, rr, eos).cpu().numpy()
                        side.edit = None
                    lpe, _ = sentence_lp(side, rr, edit)
                err = (w0 * (lpe - lp0 - ds) ** 2).sum((1, 2)).reshape(-1, 2).sum(1)
                shape = (len(requests), len(pairs))
                put(f'pair|{name}|margin', shape, ri, ix, me)
                put(f'pair|{name}|nt_err', shape, ri, ix, err.cpu().numpy())
                put(f'pair|{name}|nt_eff', shape, ri, ix, eff.cpu().numpy())
        log.progress(stage='pairs', items=bi + 1, total=len(batches))
    for bi in range(0, len(docs), c['natural_batch']):
        ids = torch.tensor(docs[bi:bi + c['natural_batch']], device=device)
        with torch.no_grad():
            lp0 = source.logprobs(ids)
        w0 = lp0.exp()
        for ri, req in enumerate(requests):
            q = masks[req]
            side, edit = edit_for('source', q)
            with torch.no_grad():
                side.edit = edit
                ds = side.logprobs(ids) - lp0
                side.edit = None
                eff = (w0 * ds ** 2).sum((1, 2))
                for name in execs:
                    side, edit = edit_for(name, q)
                    side.edit = edit
                    err = (w0 * (side.logprobs(ids) - lp0 - ds) ** 2).sum((1, 2))
                    side.edit = None
                    ix = list(range(bi, bi + len(ids)))
                    put(f'doc|{name}|nt_err', (len(requests), len(docs)), ri, ix, err.cpu().numpy())
                    put(f'doc|{name}|nt_eff', (len(requests), len(docs)), ri, ix, eff.cpu().numpy())
        log.progress(stage='docs', items=bi // c['natural_batch'] + 1, total=(len(docs) + c['natural_batch'] - 1) // c['natural_batch'])
    np.savez_compressed(out / 'responses.npz', tasks=tasks, request_names=np.array(requests), **store)

    paradigms = sorted(set(tasks))
    base = store['pair|none|margin'][0]
    summary = {}
    for name in execs:
        fam_out = {}
        for fam, reqs in families.items():
            rows = [requests.index(r) for r in reqs]
            src = store['pair|source|margin'][rows] - base
            mine = store[f'pair|{name}|margin'][rows] - base
            cell = {f'margin_{p}': float(np.sqrt(((mine - src)[:, tasks == p] ** 2).sum() / (src[:, tasks == p] ** 2).sum()))
                    for p in paradigms}
            cell['heldout_next_token'] = float(np.sqrt(store[f'pair|{name}|nt_err'][rows].sum() / store[f'pair|{name}|nt_eff'][rows].sum()))
            cell['fineweb_next_token'] = float(np.sqrt(store[f'doc|{name}|nt_err'][rows].sum() / store[f'doc|{name}|nt_eff'][rows].sum()))
            cell['E2'] = float(np.mean([cell[f'margin_{p}'] for p in paradigms] + [cell['heldout_next_token']]))
            fam_out[fam] = cell
        summary[name] = fam_out
    (out / 'READERS.json').write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(),
                                                      description=DESCRIPTION, pairs=len(pairs), docs=len(docs),
                                                      paradigms=paradigms, summary=summary), indent=1))
    (out / 'status.json').write_text(json.dumps(dict(status='PASS', ended_at_utc=datetime.now(timezone.utc).isoformat())))
    log.progress(stage='done')


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        cfg = json.loads(Path(sys.argv[sys.argv.index('--config') + 1]).read_text())
        run = Path(cfg['run_root']) / cfg['run_id']
        if run.exists():
            (run / 'status.json').write_text(json.dumps(dict(status='FAIL', error=traceback.format_exc())))
        raise
