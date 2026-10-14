import argparse
import json

import numpy as np
import torch

from battery import Battery
from p1common import ANIMALS, DATASETS, LAYERS, capture, load_model, load_tokenizer, pad_batch, run_dir, save_json

DESCRIPTION = ('P1 persona-style baseline: for each behaviour and grid layer, a diff-in-means direction from the dev '
               'split (residual stream at the answer token for the matching minus the non-matching answer; for an '
               'animal, its one-word answer minus the mean over all 20 animals), unit-normalized. The forecast for a '
               "training set is the projection of its self-contrast vector (dataset-response mean minus own-response "
               "mean) onto that direction, as in the persona-vectors projection difference. Writes forecast/pp.json.")


@torch.no_grad()
def last_acts(lm, tok, seqs, batch=256):
    out = {l: [] for l in LAYERS}
    for s in range(0, len(seqs), batch):
        chunk = seqs[s:s + batch]
        ids, mask = pad_batch(chunk, tok.pad_token_id)
        with capture(lm, LAYERS) as st:
            lm.model(input_ids=ids, attention_mask=mask)
        last = mask.sum(1) - 1
        for l in LAYERS:
            out[l].append(st[l][torch.arange(len(chunk), device=last.device), last].float().cpu())
    return {l: torch.cat(v) for l, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run', required=True)
    parser.add_argument('--allow_partial', action='store_true')
    args = parser.parse_args()
    rd = run_dir(args.run)
    battery = json.loads((rd / 'battery.json').read_text())
    tok = load_tokenizer()
    bat = Battery(tok, battery, 'dev')
    lm = load_model()
    pos = last_acts(lm, tok, [s + [t[0]] for s, t in zip(bat.lt_seqs, bat.lt_tok)])
    neg = last_acts(lm, tok, [s + [t[1]] for s, t in zip(bat.lt_seqs, bat.lt_tok)])
    an_idx = [i for i, k in enumerate(bat.an_key) if k[2] == 0]
    an = last_acts(lm, tok, [bat.an_seqs[i] for i in an_idx])
    P = len(bat.animal_prompts)
    dirs = {}
    for l in LAYERS:
        D = torch.zeros(len(bat.names), lm.config.hidden_size)
        sign = torch.tensor(bat.lt_sign, dtype=torch.float32)[:, None]
        diff = (pos[l] - neg[l]) * sign
        for bi in range(len(bat.names)):
            sel = torch.tensor(bat.lt_beh == bi)
            if sel.any():
                D[bi] = diff[sel].mean(0)
        A = torch.zeros(P, len(ANIMALS), lm.config.hidden_size)
        for row, i in enumerate(an_idx):
            pi, ai, _ = bat.an_key[i]
            A[pi, ai] = an[l][row]
        A = (A - A.mean(1, keepdim=True)).mean(0)
        for a, bi in bat.animal_index.items():
            D[bi] = A[ANIMALS.index(a)]
        assert (D.norm(dim=1) > 0).all()
        dirs[l] = D / D.norm(dim=1, keepdim=True)
    torch.save(dict(names=bat.names, dirs=dirs), rd / 'forecast' / 'pp_directions.pt')
    sc_means = torch.load(rd / 'forecast' / 'sc_means.pt')
    names = [d for d in DATASETS if d in sc_means]
    assert args.allow_partial or names == DATASETS
    pred = {}
    for l in LAYERS:
        pred[str(l)] = {}
        for d in names:
            v = sc_means[d]['data_mean'][l] - sc_means[d]['own_mean'][l]
            proj = (dirs[l] @ v).numpy()
            pred[str(l)][d] = {n: float(x) for n, x in zip(bat.names, proj)}
    cos_between = {str(l): float(np.mean(np.abs((dirs[l] @ dirs[l].T).numpy()[np.triu_indices(len(bat.names), 1)])))
                   for l in LAYERS}
    save_json(rd / 'forecast' / 'pp.json', dict(pred=pred, mean_abs_cos_between_directions=cos_between))
    print('pp done; mean |cos| between directions by layer', cos_between)


if __name__ == '__main__':
    main()
