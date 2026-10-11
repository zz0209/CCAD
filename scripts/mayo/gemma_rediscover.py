from pathlib import Path
import argparse
import json
import shutil
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemma_shift_transfer import (PRONOUNS, Dictionaries, Model, Probe, Run, batches, jumprelu, read_split,
                                  sfc_ambiguous)
from gemma_readers import word_ids

DESCRIPTION = ('Rediscover the SHIFT explanation in a target dictionary with a fixed procedure, without the source '
               'explanation: rank every target feature at the explanation sites by activation-times-gradient attribution '
               'of the original classifier logit on the fitting biographies, and accept features in rank order whose '
               'activation mass on these biographies falls at least half on pronoun tokens (pronoun part) or on female '
               'and male words (description part), until as many features are accepted as the transferred maps use.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    run = Run(cfg, args.config, False)
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
    dicts = Dictionaries(cfg, features, dev)
    dicts.load_full_target()
    model = Model(cfg, dicts.sites, dev)
    probe = Probe(2304).to(dev)
    probe.load_state_dict(torch.load(Path(cfg['probe_run']) / 'probe.pt', map_location=dev, weights_only=True))
    probe.requires_grad_(False)

    # The number of features to accept: target features with a nonzero load in the transferred maps, averaged over seeds.
    saved = json.loads((Path(cfg['count_run']) / 'maps.json').read_text())
    counts = [sum(int((np.array(v['weights']) > 0).any(0).sum()) for v in saved[name].values()) for name in cfg['count_methods']]
    target_count = int(round(float(np.mean(counts))))

    amb_train, _, _ = sfc_ambiguous(read_split(cfg, 'train'), cfg['data_seed'])
    texts = amb_train[:cfg['fit_bios']]
    pron = torch.tensor(word_ids(model.tok, sorted(PRONOUNS)), device=dev)
    gender = torch.tensor(sorted(set(word_ids(model.tok, cfg['female_words'])) | set(word_ids(model.tok, cfg['male_words']))),
                          device=dev)
    width = {s: dicts.full[s]['W_enc'].shape[1] for s in dicts.sites}
    acc = {k: {s: torch.zeros(width[s], device=dev, dtype=torch.float64) for s in dicts.sites}
           for k in ['attribution', 'mass', 'pronoun', 'gender']}
    jobs = batches(len(texts), cfg['batch'])
    for k, ix in enumerate(jobs):
        ids, mask = model.encode([texts[i] for i in ix])
        deltas, xs = {}, {}

        def mode(site, x):
            # A zero leaf at every site lets one backward pass give the logit gradient at each site.
            d = torch.zeros(x.shape, device=x.device, dtype=torch.float32, requires_grad=True)
            deltas[site], xs[site] = d, x.detach()
            return (x.float() + d).to(x.dtype)
        with torch.enable_grad():
            probe(model.pooled(ids, mask, mode)).sum().backward()
        with torch.no_grad():
            m = mask.float()
            is_pron = torch.isin(ids, pron).float() * m
            is_gender = torch.isin(ids, gender).float() * m
            for s in dicts.sites:
                t = dicts.full[s]
                z = jumprelu((xs[s].to(torch.bfloat16) @ t['W_enc']).float() + t['b_enc'], t['threshold']) * m[..., None]
                proj = (deltas[s].grad.to(torch.bfloat16) @ t['W_dec'].T).float()
                acc['attribution'][s] += (z * proj).abs().sum((0, 1)).double()
                acc['mass'][s] += z.sum((0, 1)).double()
                acc['pronoun'][s] += (z * is_pron[..., None]).sum((0, 1)).double()
                acc['gender'][s] += (z * is_gender[..., None]).sum((0, 1)).double()
                del z, proj
        run.progress(stage='attribution', items=k + 1, total=len(jobs))

    # Fixed annotation in attribution order.
    ranked = sorted(((float(acc['attribution'][s][j]), s, j) for s in dicts.sites
                     for j in torch.topk(acc['attribution'][s], min(cfg['max_checked'], width[s])).indices.tolist()),
                    reverse=True)[:cfg['max_checked']]
    accepted, checked = [], 0
    for score, s, j in ranked:
        if len(accepted) == target_count:
            break
        checked += 1
        mass = float(acc['mass'][s][j])
        if mass <= 0:
            continue
        share_p, share_g = float(acc['pronoun'][s][j]) / mass, float(acc['gender'][s][j]) / mass
        part = cfg['pronoun_part'] if share_p >= cfg['share'] else cfg['description_part'] if share_g >= cfg['share'] else None
        if part is not None:
            accepted.append(dict(site=s, index=j, part=part, attribution=score, pronoun_share=share_p, gender_share=share_g,
                                 rank=checked))
    sites = {s: dict(idx=[a['index'] for a in accepted if a['site'] == s], part=[a['part'] for a in accepted if a['site'] == s])
             for s in dicts.sites}
    record = dict(description=DESCRIPTION, target_count=target_count, counts_per_seed=counts, checked=checked,
                  accepted=len(accepted), per_part={p: sum(a['part'] == p for a in accepted)
                                                    for p in [cfg['pronoun_part'], cfg['description_part']]},
                  fit_bios=len(texts), dictionaries=dicts.paths, features=accepted, sites=sites)
    (run.dir / 'REDISCOVERY.json').write_text(json.dumps(record, indent=1))
    run.progress(stage='done', accepted=len(accepted), checked=checked, target_count=target_count)


if __name__ == '__main__':
    main()
