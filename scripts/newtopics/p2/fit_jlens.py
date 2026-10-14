import argparse
import json
import logging
import time
from pathlib import Path

import jlens

from common import band_layers, cap_gpu_memory, load_model, write_json

DESCRIPTION = ('Fit the Jacobian lens of one checkpoint with the public jlens.fit (anthropics/jacobian-lens): the '
               'first N FineWeb train documents truncated to 128 tokens, cotangents summed over target positions and '
               'averaged over source positions (first 16 positions skipped), target = last block output, source '
               'layers = the band. Resumable through the jlens checkpoint file.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--fineweb', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--n_prompts', type=int, default=200)
    parser.add_argument('--doc_offset', type=int, default=0, help='first train document (3800 for the H5 refit)')
    parser.add_argument('--dim_batch', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu_cap_gib', type=float, default=18.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.device == 'cuda':
        cap_gpu_memory(args.gpu_cap_gib)
    t0 = time.time()
    row, hf, tok, lm = load_model(args.manifest, args.label, args.device)
    prompts = json.load(open(args.fineweb))['train'][args.doc_offset:args.doc_offset + args.n_prompts]
    band = band_layers(lm.n_layers)
    lens = jlens.fit(lm, prompts, source_layers=band, dim_batch=args.dim_batch, max_seq_len=128,
                     checkpoint_path=str(args.out) + '.ckpt', checkpoint_every=5)
    if lens.n_prompts != args.n_prompts:
        raise ValueError(f'fit used {lens.n_prompts} prompts, expected {args.n_prompts}')
    lens.save(str(args.out))
    write_json(str(args.out) + '.json', dict(label=args.label, revision=row['revision'], commit=row['commit'],
                                             band=band, n_prompts=lens.n_prompts, doc_offset=args.doc_offset,
                                             dim_batch=args.dim_batch,
                                             max_seq_len=128, seconds=time.time() - t0))
    print('JFIT_DONE', args.label, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
