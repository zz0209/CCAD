import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM

from p3_common import MODEL_PATH

DESCRIPTION = ('P3 receiver-head baseline, following the Thought Anchors code (interp-reasoning/thought-anchors, '
               'whitebox-analyses/attention_analysis): per head, average token attention over sentence pairs, take each '
               'sentence\'s vertical score as the mean attention it receives from sentences at least 4 later, score '
               'heads by the Fisher kurtosis of the vertical scores averaged over traces, keep the top 16 heads, and '
               'score a sentence by the mean vertical score over those heads. Heads are chosen on one full trace per '
               'training question; a test step (the sentences between two chosen boundaries) gets the maximum score of '
               'its sentences.')
PROXIMITY = 4
TOP_K = 16


def vertical_scores(model, prompt_ids, reasoning_ids, cuts):
    starts = np.array(cuts[:-1]) + len(prompt_ids)
    ends = np.array(cuts[1:]) + len(prompt_ids)
    n = len(starts)
    T = int(ends[-1])
    P = torch.zeros(T, n, device='cuda')
    for j in range(n):
        P[starts[j]:ends[j], j] = 1
    counts = torch.tensor(ends - starts, device='cuda', dtype=torch.float32)
    r = torch.arange(n, device='cuda')
    keep = (r[:, None] - r[None, :]) >= PROXIMITY
    n_rows = keep.sum(0)
    out = {}

    def hook(module, inputs, output, layer):
        A = output[1][0].float()
        M = torch.einsum('tn,htm->hnm', P, A @ P) / (counts[:, None] * counts[None, :])
        v = (M * keep).sum(1) / n_rows.clamp(min=1)
        v[:, n_rows == 0] = float('nan')
        out[layer] = v.cpu().numpy()

    hooks = [model.model.layers[l].self_attn.register_forward_hook(lambda m, i, o, l=l: hook(m, i, o, l))
             for l in range(model.config.num_hidden_layers)]
    with torch.no_grad():
        model.model(input_ids=torch.tensor([(prompt_ids + reasoning_ids)[:T]], device='cuda'))
    for h in hooks:
        h.remove()
    return np.stack([out[l] for l in range(model.config.num_hidden_layers)])


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    items = json.loads((args.run_dir / 'items.json').read_text())
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
                                                 attn_implementation='eager').cuda().eval()
    t0 = time.time()
    kurts = []
    for k, h in enumerate(items['head_traces']):
        if len(h['sentence_cuts']) - 1 <= PROXIMITY + 1:
            continue
        v = vertical_scores(model, h['prompt_ids'], h['reasoning_ids'], h['sentence_cuts'])
        kurts.append(stats.kurtosis(v, axis=2, fisher=True, bias=True, nan_policy='omit'))
        if (k + 1) % 50 == 0 or k + 1 == len(items['head_traces']):
            print(f'[receiver] head traces {k + 1}/{len(items["head_traces"])}, {time.time() - t0:.0f}s', flush=True)
    mean_kurt = np.nanmean(np.stack(kurts), axis=0)
    flat = np.where(np.isnan(mean_kurt), -np.inf, mean_kurt).ravel()
    top = np.argsort(-flat)[:TOP_K]
    heads = np.stack(np.unravel_index(top, mean_kurt.shape), axis=1)
    rows = []
    for t in items['test']:
        sc = t['sentence_cuts']
        v = vertical_scores(model, t['prompt_ids'], t['reasoning_ids'], sc)
        sent = np.mean(v[heads[:, 0], heads[:, 1]], axis=0)
        pos = [sc.index(c) for c in t['cuts']]
        steps = []
        for a, b in zip(pos[:-1], pos[1:]):
            seg = sent[a:b]
            steps.append(None if np.all(np.isnan(seg)) else float(np.nanmax(seg)))
        rows.append(dict(id=t['id'], sentence_scores=[None if np.isnan(x) else float(x) for x in sent],
                         step_scores=steps))
    print(f'[receiver] test traces done, {time.time() - t0:.0f}s; heads {heads.tolist()}', flush=True)
    (args.run_dir / 'receiver.json').write_text(json.dumps(dict(
        proximity_ignore=PROXIMITY, top_k=TOP_K, n_head_traces=len(kurts), heads=heads.tolist(),
        head_kurtosis=[float(mean_kurt[l, h]) for l, h in heads], traces=rows)))


if __name__ == '__main__':
    main()
