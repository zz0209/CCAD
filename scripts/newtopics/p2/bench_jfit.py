import argparse
import json
import time

import torch
from jlens.fitting import jacobian_for_prompt

from common import band_layers, cap_gpu_memory, load_model

DESCRIPTION = ('Time one call of the public jlens per-prompt Jacobian estimator (all d_model output dimensions, '
               'band source layers) on FineWeb train documents, to cost the full 200-prompt fit.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--fineweb', required=True)
    parser.add_argument('--dim_batch', type=int, nargs='+', default=[8])
    parser.add_argument('--gpu_cap_gib', type=float, default=18.0)
    args = parser.parse_args()
    cap_gpu_memory(args.gpu_cap_gib)
    _, hf, tok, lm = load_model(args.manifest, args.label)
    docs = json.load(open(args.fineweb))['train'][:2]
    band = band_layers(lm.n_layers)
    print(args.label, 'n_layers', lm.n_layers, 'd_model', lm.d_model, 'band', band, flush=True)
    for db in args.dim_batch:
        torch.cuda.reset_peak_memory_stats()
        for i, doc in enumerate(docs):
            torch.cuda.synchronize()
            t0 = time.time()
            jacobian_for_prompt(lm, doc, band, dim_batch=db, max_seq_len=128)
            torch.cuda.synchronize()
            dt = time.time() - t0
            print(f'dim_batch={db} prompt {i} {dt:.1f}s peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB '
                  f'-> 200 prompts {dt * 200 / 3600:.2f} h', flush=True)


if __name__ == '__main__':
    main()
