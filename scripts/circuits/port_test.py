import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from circuit_tracer import attribute
from circuit_tracer.graph import prune_graph

from common import Dictionary, replacement_model

DESCRIPTION = ('P0 pre-check of porting annotated circuit roles across model families. Each published source graph '
               'has a role supernode on its entity token (Texas, California, China, Canada, basketball, football, '
               'Europe, US). Roles are paired with a swap partner, and each item is a base prompt (template of family A '
               'with a held-out fill) and a partner prompt (the same template with the partner entity). In the target '
               'model, the features of a set are resampled along their read directions at the entity token from the '
               'partner prompt. Cause: the logit difference of the two families\' answers changes sign, or moves at '
               'least half way to its value on the partner prompt. Iso: the answer of the control '
               'prompt "The name X starts with the letter" is unchanged under the same resampling. Sets: ported '
               '(target features correlated at least --rho with a source role feature, top --k each, from '
               'port_corr.py), one-to-one (the best target feature of each source feature), random features matched '
               'in layer and number, and rediscovery (the target dictionary\'s own graph features at the entity '
               'tokens, matched in number); plus residual patching at the entity token at the best layer.')

FAMILIES = {
    'dallas': dict(key='gemma-2-2b/gemma-fact-dallas-austin', kind='fill'),
    'oakland': dict(key='gemma-2-2b/gemma-fact-oakland-sacramento', kind='fill'),
    'shanghai': dict(key='gemma-2-2b/gemma-fact-shanghai-beijing', kind='fill'),
    'vancouver': dict(key='gemma-2-2b/gemma-fact-vancouver-victoria', kind='fill'),
    'jordan': dict(key='gemma-2-2b/gemma-michael-jordan', kind='fill'),
    'brady': dict(key='gemma-2-2b/gemma-tom-brady', kind='fill'),
    'euro': dict(key='gemma-2-2b/gemma-euro', kind='prefix', entity='Europe'),
    'dollar': dict(key='gemma-2-2b/gemma-dollar', kind='prefix', entity='US'),
}
# Shanghai and Vancouver ask for the capital of a country and of a territory, so a swapped entity has no defined answer
# among the two families' answers; that pair is left out (master_log amendment 2026-10-12T17:43:12Z).
PAIRS = [('dallas', 'oakland'), ('oakland', 'dallas'), ('jordan', 'brady'), ('brady', 'jordan'),
         ('euro', 'dollar'), ('dollar', 'euro')]
CONTROL = 'The name {} starts with the letter'


def entity_position(tok, prompt, entity):
    # Index of the token that ends the last occurrence of the entity (with BOS at index 0).
    enc = tok(prompt, return_offsets_mapping=True)
    end = prompt.rfind(entity) + len(entity)
    pos = [i for i, (s, e) in enumerate(enc['offset_mapping']) if e > s and e <= end]
    return enc['input_ids'], pos[-1]


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--target-model', default='google/gemma-3-1b-pt')
    parser.add_argument('--target-dict', default='mwhanna/gemma-scope-2-1b-pt/transcoder_all/width_16k_l0_small_affine')
    parser.add_argument('--corr', type=Path, required=True, help='PORT_CORR.json')
    parser.add_argument('--families', type=Path, required=True, help='PROMPT_FAMILIES.json')
    parser.add_argument('--n-fills', type=int, default=5)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--rho', type=float, default=.3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'config.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    dev = torch.device('cuda')
    model = replacement_model(args.target_model, args.target_dict, dev, torch.bfloat16)
    model.requires_grad_(False)
    T = Dictionary(model.transcoders)
    L = model.cfg.n_layers
    tok = model.tokenizer
    width = T.encode(torch.zeros(1, model.cfg.d_model, device=dev), 0).shape[-1]
    corr = json.loads(args.corr.read_text())
    by_src = {(s['layer'], s['feature']): s['top'] for s in corr['source_features']}
    roles = {g: [tuple(x) for x in v] for g, v in corr['roles'].items()}
    fam = json.loads(args.families.read_text())['families']

    def mapped(name, k):
        out = set()
        for s in roles[FAMILIES[name]['key']]:
            for t in by_src[s][:k]:
                if t['corr'] >= args.rho:
                    out.add((t['layer'], t['feature']))
        return out

    def by_layer(feats):
        d = {}
        for l, f in sorted(feats):
            d.setdefault(l, []).append(f)
        return d

    def encoder(l, feats):
        tc = T.t.transcoders[l]
        return tc.W_enc[feats].float(), tc.b_enc[feats].float()

    names_in = [f'blocks.{l}.{T.input_hook}' for l in range(L)]
    names_res = [f'blocks.{l}.hook_resid_pre' for l in range(L)]

    def clean(ids):
        with torch.no_grad():
            logits, cache = model.run_with_cache(torch.tensor([ids], device=dev),
                                                 names_filter=lambda x: x in names_in + names_res)
        return logits[0, -1].float(), cache

    def resample_hooks(feats, p, pcache, pp):
        hooks = []
        for l, fs in by_layer(feats).items():
            W, _ = encoder(l, fs)
            xp = pcache[f'blocks.{l}.{T.input_hook}'][0, pp].float()
            def hook(act, hook, W=W, xp=xp):
                x = act[0, p].float()
                d = W.T @ torch.linalg.solve(W @ W.T + 1e-6 * torch.eye(len(W), device=dev), W @ xp - W @ x)
                act[0, p] = (x + d).to(act.dtype)
                return act
            hooks.append((f'blocks.{l}.{T.input_hook}', hook))
        return hooks

    def patch_hooks(l, p, pcache, pp):
        xp = pcache[f'blocks.{l}.hook_resid_pre'][0, pp]
        def hook(act, hook):
            act[0, p] = xp.to(act.dtype)
            return act
        return [(f'blocks.{l}.hook_resid_pre', hook)]

    def run(ids, hooks):
        with torch.no_grad():
            return model.run_with_hooks(torch.tensor([ids], device=dev), fwd_hooks=hooks)[0, -1].float()

    def graph_features(prompt, p, k):
        # The target dictionary's own graph: its most influential feature nodes at token position p.
        graph = attribute(prompt, model, max_n_logits=10, desired_logit_prob=0.95, batch_size=128)
        scores = prune_graph(graph).cumulative_scores
        n_feat = len(graph.selected_features)
        out = []
        for i in torch.argsort(scores[:n_feat]).tolist():
            l, pos, f = (int(v) for v in graph.active_features[graph.selected_features[i]])
            if pos == p and (l, f) not in out:
                out.append((l, f))
            if len(out) >= k:
                break
        return out

    rows = []
    for a, b in PAIRS:
        A, B = fam[FAMILIES[a]['key']], fam[FAMILIES[b]['key']]
        ported = mapped(a, args.k) | mapped(b, args.k)
        one = mapped(a, 1) | mapped(b, 1)
        layers = by_layer(ported)
        rand = set()
        for l, fs in layers.items():
            pool = [f for f in range(width) if (l, f) not in ported]
            rand |= {(l, int(f)) for f in rng.choice(pool, len(fs), replace=False)}
        variants = lambda s: [tok.encode(s, add_special_tokens=False)[0], tok.encode(s.strip(), add_special_tokens=False)[0]]
        for i in range(args.n_fills):
            if FAMILIES[a]['kind'] == 'fill':
                ent_b, ent_p = A['fill'][i], B['fill'][i]
                base, partner = A['template'].format(ent_b), A['template'].format(ent_p)
            else:
                ent_b, ent_p = FAMILIES[a]['entity'], FAMILIES[b]['entity']
                base, partner = A['template'].format(A['fill'][i]), B['template'].format(A['fill'][i])
            ids_b, p_b = entity_position(tok, base, ent_b)
            ids_p, p_p = entity_position(tok, partner, ent_p)
            cids_b, cp_b = entity_position(tok, CONTROL.format(ent_b), ent_b)
            cids_p, cp_p = entity_position(tok, CONTROL.format(ent_p), ent_p)
            lb, cb = clean(ids_b)
            lp, cp = clean(ids_p)
            lcb, _ = clean(cids_b)
            _, ccp = clean(cids_p)
            # The two answers are the families' answer strings, each as the token variant (with or without a leading
            # space) that its own clean prompt prefers; the item is valid when each prompt prefers its own answer.
            top_b = max(variants(A['answer']), key=lambda t: float(lb[t]))
            top_p = max(variants(B['answer']), key=lambda t: float(lp[t]))
            top_cb = int(lcb.argmax())
            row = dict(pair=f'{a}->{b}', item=i, base=base, partner=partner, answer=tok.decode(top_b),
                       partner_answer=tok.decode(top_p), base_argmax=tok.decode(int(lb.argmax())),
                       valid=bool(lb[top_b] > lb[top_p] and lp[top_p] > lp[top_b]),
                       n_ported=len(ported), n_one_to_one=len(one))
            if not row['valid']:
                rows.append(row)
                print(json.dumps(dict(elapsed=round(time.time() - t0, 1), pair=row['pair'], item=i, valid=False)),
                      flush=True)
                continue
            ld_b = float(lb[top_b] - lb[top_p])
            ld_p = float(lp[top_b] - lp[top_p])

            def score(hooks_main, hooks_ctrl):
                lg = run(ids_b, hooks_main)
                ld = float(lg[top_b] - lg[top_p])
                cause = ld < 0 or (ld_b - ld) >= .5 * (ld_b - ld_p)
                iso = int(run(cids_b, hooks_ctrl).argmax()) == top_cb
                return dict(cause=bool(cause), iso=bool(iso), ld_moved=(ld_b - ld) / (ld_b - ld_p))

            k = len(ported)
            red = graph_features(base, p_b, (k + 1) // 2) + graph_features(partner, p_p, k // 2) if k else []
            for name, feats in [('ported', ported), ('one_to_one', one), ('random', rand), ('rediscovery', set(red))]:
                row[name] = score(resample_hooks(feats, p_b, cp, p_p), resample_hooks(feats, cp_b, ccp, cp_p))
            band = range(min(layers), max(layers) + 1) if layers else range(L)
            row['patch_by_layer'] = {l: score(patch_hooks(l, p_b, cp, p_p), patch_hooks(l, cp_b, ccp, cp_p))
                                     for l in band}
            rows.append(row)
            print(json.dumps(dict(elapsed=round(time.time() - t0, 1), pair=row['pair'], item=i, valid=True,
                                  n_ported=row['n_ported'],
                                  **{m: row[m] for m in ('ported', 'one_to_one', 'rediscovery')})), flush=True)

    # Summary. Residual patching uses, for each directed family, the layer with the best mean score (favoring it).
    valid = [r for r in rows if r['valid']]
    for a, b in PAIRS:
        fr = [r for r in valid if r['pair'] == f'{a}->{b}']
        if fr:
            best = max(fr[0]['patch_by_layer'], key=lambda l: np.mean(
                [(r['patch_by_layer'][l]['cause'] + r['patch_by_layer'][l]['iso']) / 2 for r in fr]))
            for r in fr:
                r['patch'] = dict(r['patch_by_layer'][best], layer=int(best))
    methods = ['ported', 'one_to_one', 'random', 'rediscovery', 'patch']
    fams = sorted({r['pair'] for r in valid})
    item_score = lambda r, m: (r[m]['cause'] + r[m]['iso']) / 2
    summary = dict(items=len(rows), valid=len(valid), families=fams)
    for m in methods:
        summary[m] = dict(cause=float(np.mean([r[m]['cause'] for r in valid])),
                          iso=float(np.mean([r[m]['iso'] for r in valid])),
                          score=float(np.mean([item_score(r, m) for r in valid])))
    boot = np.random.default_rng(1)
    for m in ['patch', 'rediscovery', 'one_to_one', 'random']:
        diffs = []
        for _ in range(2000):
            pick = boot.choice(len(fams), len(fams))
            sel = [r for j in pick for r in valid if r['pair'] == fams[j]]
            diffs.append(np.mean([item_score(r, 'ported') - item_score(r, m) for r in sel]))
        summary[f'ported_minus_{m}'] = [float(np.mean([item_score(r, 'ported') - item_score(r, m) for r in valid])),
                                        float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))]
    s = {m: summary[m]['score'] for m in methods}
    if len(valid) < 20:
        decision = 'undetermined: fewer than 20 valid items'
    elif summary['ported']['cause'] < .30 or s['ported'] <= max(s['patch'], s['rediscovery']) + .05:
        decision = 'stop'
    elif s['ported'] <= s['one_to_one'] + .05:
        decision = 'porting holds, one-to-one suffices'
    else:
        decision = 'porting holds, many-to-many adds'
    summary['decision'] = decision
    print(json.dumps(summary, indent=1))
    (args.out / 'PORT_P0.json').write_text(json.dumps(dict(
        written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, summary=summary, rows=rows),
        indent=1))


if __name__ == '__main__':
    main()
