import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from p3_common import MODEL_PATH

DESCRIPTION = ('P3 probe inputs: run the reasoning model (Hugging Face, bf16) over every training/validation prefix '
               '(prompt plus rollout reasoning up to its cut) and over every test trace (prompt plus reasoning up to its '
               'last chosen boundary), and store the residual stream after the candidate layers as float16 arrays with '
               'an index of offsets. A test boundary prefix is a leading slice of its trace, since attention is causal.')
LAYERS = [3, 7, 11, 15, 19, 23, 27]


def run(model, seqs, layers, out_dir, name):
    lengths = [len(s) for s in seqs]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    d = model.config.hidden_size
    arrays = {l: np.lib.format.open_memmap(out_dir / f'{name}_L{l}.npy', mode='w+', dtype=np.float16,
                                           shape=(int(offsets[-1]), d)) for l in layers}
    grabbed = {}
    hooks = [model.model.layers[l].register_forward_hook(
        lambda m, i, o, l=l: grabbed.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0].to(torch.float16).cpu()))
        for l in layers]
    t0 = time.time()
    with torch.no_grad():
        for k, s in enumerate(seqs):
            model.model(input_ids=torch.tensor([s], device='cuda'))
            for l in layers:
                arrays[l][offsets[k]:offsets[k + 1]] = grabbed[l].numpy()
            if (k + 1) % 100 == 0 or k + 1 == len(seqs):
                print(f'[extract] {name}: {k + 1}/{len(seqs)} sequences, {offsets[k + 1]} tokens, '
                      f'{time.time() - t0:.0f}s', flush=True)
    for h in hooks:
        h.remove()
    for a in arrays.values():
        a.flush()
    return offsets.tolist()


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--layers', type=int, nargs='+', default=LAYERS)
    args = parser.parse_args()
    items = json.loads((args.run_dir / 'items.json').read_text())
    out_dir = args.run_dir / 'resid'
    out_dir.mkdir(exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
                                                 attn_implementation='sdpa').cuda().eval()
    pairs = items['pairs']
    pair_off = run(model, [p['prompt_ids'] + p['prefix_ids'] for p in pairs], args.layers, out_dir, 'pairs')
    test = items['test']
    test_off = run(model, [t['prompt_ids'] + t['reasoning_ids'][:t['cuts'][-1]] for t in test], args.layers, out_dir,
                   'test')
    index = dict(layers=args.layers,
                 pairs=[dict(id=p['id'], split=p['split'], label=p['label'], start=pair_off[k],
                             length=pair_off[k + 1] - pair_off[k]) for k, p in enumerate(pairs)],
                 test=[dict(id=t['id'], start=test_off[k],
                            prefix_lengths=[len(t['prompt_ids']) + c for c in t['cuts']]) for k, t in enumerate(test)])
    (out_dir / 'index.json').write_text(json.dumps(index))


if __name__ == '__main__':
    main()
