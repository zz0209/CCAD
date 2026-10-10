from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

import numpy as np

from run_shift_explanation import source_groups

DESCRIPTION = ('List the new-dictionary features that the part-supervised SHIFT correspondence assigns to each '
               'annotated part, with the natural-text tokens on which they and their source features fire most.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    import torch
    import transformers
    device = torch.device(c['device'])
    sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
    from dictionary_learning.trainers.top_k import AutoEncoderTopK
    from train_shift_dictionaries import site_module
    source = json.loads(Path(c['source_manifest']).read_text())
    groups, _ = source_groups(Path(c['notebook']), source['members'])
    sites = list(source['members'])
    sb = np.load(c['source_parameters'])
    sp = {s: {k: torch.tensor(sb[s + '__' + k], device=device) for k in ['encoder', 'encoder_bias', 'decoder', 'center']} for s in sites}
    relation = np.load(c['relation_npz'])
    targets = {}
    for site in sites:
        folder = Path(c['program_dir']) if site in c['adapted_sites'] else Path(c['target_directory'])
        state = torch.load(folder / f'{site}_seed{c["target_seed"]}.pt', map_location=device, weights_only=True)
        sae = AutoEncoderTopK(512, state['encoder.weight'].shape[0], int(state['k'])).to(device)
        sae.load_state_dict(state)
        targets[site] = sae.eval().requires_grad_(False)
    chosen = []
    for part, members in groups.items():
        for site in sites:
            ids = [source['members'][site].index(i) for i in members.get(site, [])]
            if not ids:
                continue
            weight = relation[site + '__' + c['relation_key']][:, ids].sum(1)
            for j in np.argsort(-weight)[:c['features_per_part_site']]:
                if weight[j] > 0:
                    chosen.append(dict(part=part, site=site, target_feature=int(j), membership=float(weight[j]),
                                       source_features=[int(source['members'][site][i]) for i in ids]))
    tok = transformers.AutoTokenizer.from_pretrained(c['model_local_dir'])
    model = transformers.AutoModelForCausalLM.from_pretrained(c['model_local_dir'], local_files_only=True,
                                                              dtype=torch.float32).eval().to(device)
    best_t = {(d['site'], d['target_feature']): [] for d in chosen}
    best_s = {(d['site'], i): [] for d in chosen for i in range(len(source['members'][d['site']]))}
    captured = {}
    for site in sites:
        site_module(model, site).register_forward_hook(
            lambda m, i, o, site=site: captured.__setitem__(site, (o[0] if isinstance(o, tuple) else o).detach()))
    tokens = np.memmap(c['paired_tokens'], dtype='<u2', mode='r').reshape(-1, 128)[:c['natural_sequences']]
    with torch.no_grad():
        for off in range(0, len(tokens), 16):
            ids = torch.tensor(np.array(tokens[off:off + 16], dtype='int64'), device=device)
            model.gpt_neox(ids, use_cache=False)
            for site in sites:
                h = captured[site]
                zt = targets[site].encode(h)
                zs = torch.relu((h - sp[site]['center']) @ sp[site]['encoder'].T + sp[site]['encoder_bias'])
                for key, store, z in [(k, best_t, zt) for k in best_t if k[0] == site] + \
                                     [(k, best_s, zs) for k in best_s if k[0] == site]:
                    a = z[..., key[1]]
                    top = a.flatten().topk(8)
                    for v, flat in zip(top.values.tolist(), top.indices.tolist()):
                        if v > 0:
                            b, p = divmod(flat, a.shape[1])
                            store[key].append((v, int(ids[b, p]), int(off + b), p))
    def summary(entries):
        entries = sorted(entries, reverse=True)[:12]
        return [dict(activation=v, token=tok.decode([t])) for v, t, _, _ in entries]
    for d in chosen:
        d['target_top_tokens'] = summary(best_t[(d['site'], d['target_feature'])])
        d['source_top_tokens'] = {str(source['members'][d['site']][i]): summary(best_s[(d['site'], i)])
                                  for i in range(len(source['members'][d['site']]))
                                  if int(source['members'][d['site']][i]) in d['source_features']}
    # Share of each pronoun-part feature's strongest activations that fall on personal pronoun tokens.
    pronouns = {'he', 'she', 'him', 'her', 'his', 'hers', 'himself', 'herself'}
    precision = [np.mean([t['token'].strip().lower() in pronouns for t in d['target_top_tokens']])
                 for d in chosen if d['part'] == 'pronouns' and d['target_top_tokens']]
    out = Path(c['output'])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(), config=c,
                                   pronoun_token_precision=float(np.mean(precision)), features=chosen), indent=1,
                              ensure_ascii=False), encoding='utf-8')
    print('pronoun token precision', float(np.mean(precision)))
    for d in chosen:
        print(d['part'], d['site'], d['target_feature'], round(d['membership'], 3),
              [t['token'] for t in d['target_top_tokens'][:8]])


if __name__ == '__main__':
    main()
