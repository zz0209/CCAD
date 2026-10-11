from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import shutil
import sys
import traceback

import numpy as np
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gemma_shift_transfer import PRONOUNS, Run, batches, read_split, sfc_balanced
from gemma_task_use import later_train
from run_shift_explanation import source_groups
from train_shift_dictionaries import site_module

DESCRIPTION = ('Held-out readers for saved Pythia-70M SHIFT correspondences: every relation, the source edit and the '
               'clean model on the held-out development biographies and generic text, read by the original classifier, '
               'later classifiers and a gender probe that never supervised any relation, the next-token distribution, '
               'word-class probability mass and the final hidden states.')


def train_head(acts, labels, cfg, dev):
    torch.manual_seed(cfg['probe_seed'])
    head = torch.nn.Linear(acts.shape[1], 1).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg['probe_lr'])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for epoch in range(cfg['head_epochs']):
        order = np.random.default_rng(cfg['probe_seed'] + epoch).permutation(len(acts))
        for i in range(0, len(acts), cfg['probe_batch']):
            ix = order[i:i + cfg['probe_batch']]
            x = torch.tensor(acts[ix], device=dev, dtype=torch.float32)
            y = torch.tensor(labels[ix], device=dev, dtype=torch.float32)
            opt.zero_grad()
            loss_fn(head(x).squeeze(-1), y).backward()
            opt.step()
    return head.requires_grad_(False)


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    run = Run(cfg, args.config, args.resume)
    shutil.copy2(__file__, run.dir / 'code' / Path(__file__).name)
    try:
        execute(cfg, run)
    except BaseException as exc:
        run.status('FAIL', error=f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}')
        raise
    run.status('PASS')


def execute(cfg, run):
    dev = torch.device('cuda:0')
    torch.manual_seed(cfg['seed'])
    sys.path.extend([cfg['dictionary_source_dir'], cfg['dictionary_overlay_dir']])
    from dictionary_learning.trainers.top_k import AutoEncoderTopK
    source = json.loads(Path(cfg['source_manifest']).read_text())
    members = source['members']
    sites = list(members)
    groups, _ = source_groups(cfg['notebook'], members)
    parts = list(groups)
    sb = np.load(cfg['source_parameters'])
    sp = {s: {k: torch.tensor(sb[f'{s}__{k}'], device=dev) for k in ['encoder', 'encoder_bias', 'decoder', 'center']} for s in sites}
    targets = {}
    for s in sites:
        state = torch.load(Path(cfg['target_directory']) / f'{s}_seed{cfg["target_seed"]}.pt', map_location=dev, weights_only=True)
        sae = AutoEncoderTopK(512, state['encoder.weight'].shape[0], int(state['k'])).to(dev)
        sae.load_state_dict(state)
        targets[s] = sae.requires_grad_(False)
    executors = {}
    for spec in cfg['executors']:
        rel = np.load(spec['relation'])
        executors[spec['name']] = {s: torch.tensor(rel[f'{s}__{spec["key"]}'], device=dev, dtype=torch.float32) for s in sites}

    # Requests over the 55 explanation features, as one vector in site order.
    flat = [(s, i) for s in sites for i in members[s]]
    def vector(selected):
        return np.array([any(i in groups[g].get(s, []) for g in selected) for s, i in flat], dtype=np.float32)
    masks = {'full': vector(parts)}
    masks.update({p: vector([p]) for p in parts})
    unions = [f'{a}+{b}' for k, a in enumerate(parts) for b in parts[k + 1:]]
    masks.update({u: vector(u.split('+')) for u in unions})
    rng = np.random.default_rng(cfg['random_seed'])
    randoms = []
    for k in range(cfg['random_requests']):
        m = rng.random(len(flat)) < rng.random()
        m[rng.integers(len(flat))] |= not m.any()
        masks[f'random_{k}'] = m.astype(np.float32)
        randoms.append(f'random_{k}')
    families = dict(complete=['full'], parts=parts, new=unions, random=randoms)
    requests = [r for f in ['complete', 'parts', 'new', 'random'] for r in families[f]]
    natural_requests = [r for f in ['complete', 'parts', 'random'] for r in families[f]]
    (run.dir / 'requests.json').write_text(json.dumps(dict(families=families, masks={r: masks[r].tolist() for r in requests}), indent=1))
    offsets = np.cumsum([0] + [len(members[s]) for s in sites])

    model = transformers.AutoModelForCausalLM.from_pretrained(cfg['model_local_dir'], local_files_only=True, dtype=torch.float32,
                                                              attn_implementation='eager').eval().to(dev)
    model.requires_grad_(False)
    model.config.use_cache = False
    tok = transformers.AutoTokenizer.from_pretrained(cfg['model_local_dir'], local_files_only=True)
    head_out = model.get_output_embeddings()
    state = dict(mode=None, q=None, mask=None, pooled=None)

    def hook(site):
        def f(module, inputs, out):
            x = out[0] if isinstance(out, tuple) else out
            mode, q = state['mode'], state['q']
            if mode is not None:
                qs = q[:, offsets[sites.index(site)]:offsets[sites.index(site) + 1]]
                if mode == 'source':
                    s = sp[site]
                    z = torch.relu((x - s['center']) @ s['encoder'].T + s['encoder_bias'])
                    x = x - (z * qs[:, None, :]) @ s['decoder']
                else:
                    t = targets[site]
                    coeff = qs @ executors[mode][site].T
                    x = x - (t.encode(x) * coeff[:, None, :]) @ t.decoder.weight.T
            if site == 'resid_4':
                m = state['mask'].float()
                state['pooled'] = (x * m[..., None]).sum(1) / m.sum(1, keepdim=True)
            return (x, *out[1:]) if isinstance(out, tuple) else x
        return f
    for s in sites:
        site_module(model, s).register_forward_hook(hook(s))

    @torch.no_grad()
    def forward(ids, mask, mode=None, q=None):
        state.update(mode=mode, q=q, mask=mask)
        hidden = model.gpt_neox(ids, attention_mask=mask, use_cache=False).last_hidden_state
        state['mode'] = None
        return hidden, state['pooled'], head_out(hidden).float().log_softmax(-1)

    def encode(token_lists):
        n = max(len(t) for t in token_lists)
        ids = torch.zeros((len(token_lists), n), dtype=torch.long, device=dev)
        mask = torch.zeros_like(ids)
        for k, t in enumerate(token_lists):
            ids[k, :len(t)] = torch.tensor(t, device=dev)
            mask[k, :len(t)] = 1
        return ids, mask

    def pooled_clean(texts):
        out = []
        for ix in batches(len(texts), cfg['batch']):
            ids, mask = encode([tok(texts[i])['input_ids'][:cfg['max_length']] for i in ix])
            out.append(forward(ids, mask)[1].cpu().numpy())
        return np.concatenate(out)

    probe = np.load(Path(cfg['frozen_source_run']) / 'probe.npz')
    original = torch.nn.Linear(512, 1).to(dev)
    original.weight.data = torch.tensor(probe['weight'], device=dev)
    original.bias.data = torch.tensor(probe['bias'], device=dev)
    heads = {'original': original.requires_grad_(False)}
    train = read_split(cfg, 'train')
    for pair in cfg['later_pairs']:
        for orientation in (0, 1):
            texts, labels = later_train(train, pair['negative'], pair['positive'], orientation, cfg['data_seed'])
            texts, labels = texts[:cfg['head_train_bios']], labels[:cfg['head_train_bios']]
            heads[f'{pair["name"]}_o{orientation}'] = train_head(pooled_clean(texts), labels, cfg, dev)
            run.progress(stage='heads', head=f'{pair["name"]}_o{orientation}', bios=len(texts))
    bal, _, bal_gender = sfc_balanced(train, cfg['data_seed'])
    heads['gender'] = train_head(pooled_clean(bal[:cfg['head_train_bios']]), bal_gender[:cfg['head_train_bios']], cfg, dev)
    torch.save({k: v.state_dict() for k, v in heads.items()}, run.dir / 'heads.pt')
    readers = list(heads)

    def word_ids(words):
        ids = set()
        for w in words:
            for form in (' ' + w, ' ' + w.capitalize()):
                t = tok(form)['input_ids']
                if len(t) == 1:
                    ids.add(t[0])
        return sorted(ids)
    classes = dict(pronoun=word_ids(sorted(PRONOUNS)), female=word_ids(cfg['female_words']), male=word_ids(cfg['male_words']))
    (run.dir / 'word_classes.json').write_text(json.dumps(dict(token_ids=classes), indent=1))
    class_ix = {k: torch.tensor(v, device=dev) for k, v in classes.items()}

    rows = json.loads(Path(cfg['evaluation_membership']).read_text())['rows']
    bios = [r['tokens'] for r in rows]
    true = np.array([r['label'] for r in rows])
    gender = np.array([r['gender'] for r in rows])
    docs = np.memmap(cfg['natural_lm_eval_tokens'], dtype='<u2', mode='r').reshape(-1, 128)[:cfg['natural_docs']]
    names = ['source'] + list(executors)
    store = {}
    total = dict(bio=len(bios), doc=len(docs))
    jobs = [('bio', ix) for ix in batches(len(bios), cfg['batch'])] + [('doc', ix) for ix in batches(len(docs), cfg['batch'])]
    for k, (kind, ix) in enumerate(jobs):
        if kind == 'bio':
            ids, mask = encode([bios[i] for i in ix])
            reqs = requests
        else:
            ids = torch.tensor(np.array(docs[ix], dtype=np.int64), device=dev)
            mask = torch.ones_like(ids)
            reqs = natural_requests
        m = mask.float()
        h0, p0, lp0 = forward(ids, mask)
        w0 = lp0.exp() * m[..., None]
        mass0 = {c: torch.logsumexp(lp0[..., v], -1) for c, v in class_ix.items()}
        if kind == 'bio':
            for r_name, head in heads.items():
                store.setdefault(f'bio|none|{r_name}', np.full((1, total['bio']), np.nan, dtype=np.float32))[0, ix] = \
                    head(p0).squeeze(-1).cpu().numpy()
        for ri, req in enumerate(reqs):
            q = torch.tensor(masks[req], device=dev).expand(len(ix), -1)
            hs, ps, lps = forward(ids, mask, 'source', q)
            ds = lps - lp0
            mass_s = {c: torch.logsumexp(lps[..., v], -1) - mass0[c] for c, v in class_ix.items()}
            for name in names:
                he, pe, lpe = (hs, ps, lps) if name == 'source' else forward(ids, mask, name, q)
                row = dict(nt_err=(w0 * (lpe - lp0 - ds) ** 2).sum((1, 2)), nt_eff=(w0 * ds ** 2).sum((1, 2)))
                if kind == 'bio':
                    for r_name, head in heads.items():
                        row[f'logit_{r_name}'] = head(pe).squeeze(-1)
                    for c, v in class_ix.items():
                        delta = torch.logsumexp(lpe[..., v], -1) - mass0[c]
                        row[f'mass_{c}_err'] = (((delta - mass_s[c]) ** 2) * m).sum(1)
                        row[f'mass_{c}_eff'] = ((mass_s[c] ** 2) * m).sum(1)
                    row['state_err'] = (((he - hs) ** 2).sum(-1) * m).sum(1)
                    row['state_eff'] = (((hs - h0) ** 2).sum(-1) * m).sum(1)
                for key, value in row.items():
                    full_key = f'{kind}|{name}|{key}'
                    if full_key not in store:
                        store[full_key] = np.full((len(reqs), total[kind]), np.nan, dtype=np.float32)
                    store[full_key][ri, ix] = value.float().cpu().numpy()
        run.progress(stage='readers', kind=kind, items=k + 1, total=len(jobs))
    np.savez_compressed(run.dir / 'responses.npz', bio_true=true, bio_gender=gender, request_names=np.array(requests),
                        natural_request_names=np.array(natural_requests), **store)
    summary = {}
    clean = {r: store[f'bio|none|{r}'][0] for r in readers}
    for name in names:
        out = {}
        for fam in ['complete', 'parts', 'new', 'random']:
            rows_f = [requests.index(r) for r in families[fam]]
            cell = {}
            for r in readers:
                src = store[f'bio|source|logit_{r}'][rows_f] - clean[r]
                mine = store[f'bio|{name}|logit_{r}'][rows_f] - clean[r]
                cell[r] = float(np.sqrt(((mine - src) ** 2).sum() / (src ** 2).sum()))
            for key in ['nt', 'state'] + [f'mass_{c}' for c in classes]:
                cell['bio_next_token' if key == 'nt' else key] = float(np.sqrt(
                    store[f'bio|{name}|{key}_err'][rows_f].sum() / store[f'bio|{name}|{key}_eff'][rows_f].sum()))
            if fam != 'new':
                nrows = [natural_requests.index(r) for r in families[fam]]
                cell['fineweb_next_token'] = float(np.sqrt(store[f'doc|{name}|nt_err'][nrows].sum() /
                                                           store[f'doc|{name}|nt_eff'][nrows].sum()))
            cell['E2'] = float(np.mean([cell[r] for r in readers if r != 'original'] + [cell['bio_next_token']]))
            out[fam] = cell
        summary[name] = out
    (run.dir / 'READERS.json').write_text(json.dumps(dict(
        written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, bios=len(bios), docs=len(docs),
        readers=readers, word_class_tokens={k: len(v) for k, v in classes.items()}, summary=summary), indent=1))
    run.progress(stage='done')


if __name__ == '__main__':
    main()
