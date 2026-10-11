from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import shutil
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemma_shift_transfer import (PRONOUNS, Dictionaries, Model, Probe, Run, TargetMap, batches, collect_pooled,
                                  read_split, readout_mode, request_masks, sfc_balanced, source_mode)
from gemma_task_use import later_train, train_head

DESCRIPTION = ('Held-out readers for saved Gemma SHIFT correspondences: every map, the source edit and the readout are '
               'applied on development biographies and generic text, and each request is read by the original '
               'classifier, later classifiers and a gender probe that never supervised any map, the model\'s own '
               'next-token distribution, word-class probability mass and the layer-22 states.')


def word_ids(tok, words):
    ids = set()
    for w in words:
        for piece in ('▁' + w, '▁' + w.capitalize()):
            i = tok.convert_tokens_to_ids(piece)
            if i is not None and i != tok.unk_token_id:
                ids.add(int(i))
    return sorted(ids)


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
    manifest = json.loads(Path(cfg['features']).read_text())
    features = [dict(site=site, index=int(idx), part=part, comment=comment)
                for site, rows in manifest['features'].items() for idx, part, comment in rows]
    assert len(features) == cfg['expected_features'], len(features)
    masks, families = request_masks(cfg, features)
    # Fixed random subsets: each request draws its own inclusion probability, as the natural-text family does.
    rng = np.random.default_rng(cfg['random_seed'])
    families['random'] = []
    for k in range(cfg['random_requests']):
        m = rng.random(len(features)) < rng.random()
        m[rng.integers(len(features))] |= not m.any()
        masks[f'random_{k}'] = m.astype(np.float32)
        families['random'].append(f'random_{k}')
    requests = [r for fam in ['complete', 'parts', 'new', 'random'] for r in families[fam]]
    natural_requests = [r for fam in ['complete', 'parts', 'random'] for r in families[fam]]
    (run.dir / 'requests.json').write_text(json.dumps(dict(families=families, masks={r: masks[r].tolist() for r in requests}), indent=1))

    dicts = Dictionaries(cfg, features, dev)
    executors = {}
    for spec in cfg['map_runs']:
        source_run = Path(spec['run'])
        assert json.loads((source_run / 'status.json').read_text())['status'] == 'PASS', source_run
        assert dicts.paths == json.loads((source_run / 'dictionaries.json').read_text()), source_run
        gain_max = json.loads((source_run / 'config.resolved.json').read_text()).get('matched_gain_max', 1.)
        saved = json.loads((source_run / 'maps.json').read_text())
        for name in spec['methods']:
            # One-to-one maps trained with gains above one were applied without the deletion cap.
            cap = None if gain_max > 1 and name.startswith(('natural_matched', 'matched_')) else 1.
            executors[f"{spec['prefix']}{name}"] = TargetMap(
                dicts, {s: np.array(v['idx']) for s, v in saved[name].items()},
                {s: torch.tensor(np.array(v['weights'], dtype=np.float32), device=dev) for s, v in saved[name].items()}, cap)
    dicts.load_full_target()
    model = Model(cfg, dicts.sites, dev)
    probe = Probe(2304).to(dev)
    probe.load_state_dict(torch.load(Path(cfg['probe_run']) / 'probe.pt', map_location=dev, weights_only=True))
    probe.requires_grad_(False)

    # Held-out readers: later classifiers on biased pairs of other professions, and a gender probe, all trained on
    # clean pooled states before any map is applied.
    train = read_split(cfg, 'train')
    heads = {'original': probe}
    for pair in cfg['later_pairs']:
        for orientation in (0, 1):
            texts, labels = later_train(train, pair['negative'], pair['positive'], orientation, cfg['data_seed'])
            acts = collect_pooled(model, texts, cfg['use_batch'], None, run, f'later_{pair["name"]}_{orientation}')
            heads[f'{pair["name"]}_o{orientation}'] = train_head(acts, labels, cfg, cfg['later_epochs'], dev)
    bal_train, _, bal_train_gender = sfc_balanced(train, cfg['data_seed'])
    acts = collect_pooled(model, bal_train, cfg['use_batch'], None, run, 'gender_probe')
    heads['gender'] = train_head(acts, bal_train_gender, cfg, cfg['later_epochs'], dev)
    del acts
    for h in heads.values():
        h.requires_grad_(False)
    torch.save({k: v.state_dict() for k, v in heads.items()}, run.dir / 'heads.pt')
    # The large pooled batches above leave cached blocks that the reader loop never needs; other jobs share the GPU.
    torch.cuda.empty_cache()
    readers = list(heads)

    classes = dict(pronoun=word_ids(model.tok, sorted(PRONOUNS)), female=word_ids(model.tok, cfg['female_words']),
                   male=word_ids(model.tok, cfg['male_words']))
    (run.dir / 'word_classes.json').write_text(json.dumps(dict(words=dict(pronoun=sorted(PRONOUNS), female=cfg['female_words'],
                                                                          male=cfg['male_words']), token_ids=classes), indent=1))
    class_ix = {k: torch.tensor(v, device=dev) for k, v in classes.items()}

    data = read_split(cfg, cfg['eval_split'])
    bios, true, gender = sfc_balanced(data, cfg['data_seed'])
    # Confirmation readers start after the biographies and documents that development readers already used.
    off, doff = cfg.get('eval_offset', 0), cfg.get('natural_doc_offset', 0)
    bios, true, gender = (x[off:off + cfg['eval_bios']] for x in (bios, true, gender))
    corpus = json.loads(Path(cfg['natural_lm_texts']).read_text())
    docs = corpus['eval'][doff:doff + cfg['natural_docs']]
    assert len(bios) and len(docs), (len(bios), len(docs))

    @torch.no_grad()
    def forward(ids, mask, mode):
        model.mode, model.ids, model.mask = mode, ids, mask
        logits = model.full(input_ids=ids, attention_mask=mask, use_cache=False).logits
        model.mode = None
        h, model.captured = model.captured.float(), None
        return h, logits.float().log_softmax(-1)

    names = ['source', 'readout'] + list(executors)

    def make_mode(name, q):
        if name == 'source':
            return source_mode(dicts, q)
        if name == 'readout':
            return readout_mode(dicts, q)
        return executors[name].mode(q)

    state_path = run.dir / 'partial.npz'
    store = dict(np.load(state_path)) if state_path.exists() else {}
    done = set(json.loads((run.dir / 'partial.json').read_text())['done']) if (run.dir / 'partial.json').exists() else set()
    total = dict(bio=len(bios), doc=len(docs))
    jobs = [('bio', ix) for ix in batches(len(bios), cfg['batch'])] + [('doc', ix) for ix in batches(len(docs), cfg['batch'])]
    for k, (kind, ix) in enumerate(jobs):
        if f'{kind}:{ix[0]}' in done:
            continue
        if kind == 'bio':
            ids, mask = model.encode([bios[i] for i in ix])
            reqs = requests
        else:
            enc = model.tok([docs[i] for i in ix], truncation=True, max_length=cfg['natural_lm_len'], padding=True,
                            return_tensors='pt')
            ids, mask = enc['input_ids'].to(dev), enc['attention_mask'].to(dev)
            reqs = natural_requests
        m = mask.float()
        h0, lp0 = forward(ids, mask, None)
        w0 = lp0.exp() * m[..., None]
        mass0 = {c: torch.logsumexp(lp0[..., v], -1) for c, v in class_ix.items()}
        pooled0 = (h0 * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        if kind == 'bio':
            for r_name, head in heads.items():
                key = f'bio|none|{r_name}'
                if key not in store:
                    store[key] = np.full((1, total['bio']), np.nan, dtype=np.float32)
                store[key][0, ix] = head(pooled0).cpu().numpy()
        for ri, req in enumerate(reqs):
            q = torch.tensor(masks[req], device=dev).expand(len(ix), -1)
            hs, lps = forward(ids, mask, source_mode(dicts, q))
            ds = lps - lp0
            mass_s = {c: torch.logsumexp(lps[..., v], -1) - mass0[c] for c, v in class_ix.items()}
            pooled_s = (hs * m[..., None]).sum(1) / m.sum(1, keepdim=True)
            eff_nt = (w0 * ds ** 2).sum((1, 2))
            eff_h = (((hs - h0) ** 2).sum(-1) * m).sum(1)
            for name in names:
                if name == 'source':
                    he, lpe, pooled_e = hs, lps, pooled_s
                else:
                    he, lpe = forward(ids, mask, make_mode(name, q))
                    pooled_e = (he * m[..., None]).sum(1) / m.sum(1, keepdim=True)
                row = dict(nt_err=(w0 * (lpe - lp0 - ds) ** 2).sum((1, 2)), nt_eff=eff_nt)
                if kind == 'bio':
                    for r_name, head in heads.items():
                        row[f'logit_{r_name}'] = head(pooled_e)
                    for c, v in class_ix.items():
                        delta = torch.logsumexp(lpe[..., v], -1) - mass0[c]
                        row[f'mass_{c}_err'] = (((delta - mass_s[c]) ** 2) * m).sum(1)
                        row[f'mass_{c}_eff'] = ((mass_s[c] ** 2) * m).sum(1)
                    row['state_err'] = (((he - hs) ** 2).sum(-1) * m).sum(1)
                    row['state_eff'] = eff_h
                for key, value in row.items():
                    full_key = f'{kind}|{name}|{key}'
                    if full_key not in store:
                        store[full_key] = np.full((len(reqs), total[kind]), np.nan, dtype=np.float32)
                    store[full_key][ri, ix] = value.float().cpu().numpy()
        done.add(f'{kind}:{ix[0]}')
        if k % 4 == 3 or k == len(jobs) - 1:
            np.savez_compressed(state_path, **store)
            (run.dir / 'partial.json').write_text(json.dumps(dict(done=sorted(done))))
        run.progress(stage='readers', kind=kind, items=k + 1, total=len(jobs))
    np.savez_compressed(run.dir / 'responses.npz', bio_true=true, bio_gender=gender, request_names=np.array(requests),
                        natural_request_names=np.array(natural_requests), **store)

    # Summary: effect nRMSE pooled over items and the requests of each family, for every executor and reader.
    summary = {}
    clean = {r_name: store[f'bio|none|{r_name}'][0] for r_name in readers}
    for name in names:
        out = {}
        for fam in ['complete', 'parts', 'new', 'random']:
            rows = [requests.index(r) for r in families[fam]]
            cell = {}
            src = {r_name: store[f'bio|source|logit_{r_name}'][rows] - clean[r_name] for r_name in readers}
            for r_name in readers:
                mine = store[f'bio|{name}|logit_{r_name}'][rows] - clean[r_name]
                cell[r_name] = float(np.sqrt(((mine - src[r_name]) ** 2).sum() / (src[r_name] ** 2).sum()))
            for key in ['nt', 'state'] + [f'mass_{c}' for c in classes]:
                cell[key if key != 'nt' else 'bio_next_token'] = float(np.sqrt(
                    store[f'bio|{name}|{key}_err'][rows].sum() / store[f'bio|{name}|{key}_eff'][rows].sum()))
            if fam != 'new':
                nrows = [natural_requests.index(r) for r in families[fam]]
                cell['fineweb_next_token'] = float(np.sqrt(store[f'doc|{name}|nt_err'][nrows].sum() /
                                                           store[f'doc|{name}|nt_eff'][nrows].sum()))
            held = [r for r in readers if r != 'original'] + ['bio_next_token']
            cell['E2'] = float(np.mean([cell[r] for r in held]))
            out[fam] = cell
        summary[name] = out
    record = dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, eval_split=cfg['eval_split'],
                  bios=len(bios), docs=len(docs), eval_offset=off, natural_doc_offset=doff, readers=readers, word_class_tokens={k: len(v) for k, v in classes.items()},
                  e2=('mean of the later classifiers, the gender probe and the biography next-token error; the original '
                      'classifier supervised the part maps and is reported separately'), summary=summary)
    (run.dir / 'READERS.json').write_text(json.dumps(record, indent=1))
    state_path.unlink()
    run.progress(stage='done')


if __name__ == '__main__':
    main()
