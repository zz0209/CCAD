import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers

from cross_seed_transfer import Side, margins
from grammar_readers import pairs_of

DESCRIPTION = ('Source-only inclusion of the confirmation BLiMP paradigms (research lead decision 4): a paradigm enters a '
               'model\'s confirmation set when the complete source edit changes its good-minus-bad margins by a mean '
               'absolute amount of at least a tenth of the clean margins\' standard deviation, for every source '
               'explanation of that model. No target dictionary or correspondence is read.')


class Log:
    def checked(self, path, label=''):
        return path


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--configs', nargs='+', type=Path, required=True, help='reader configs, one per source explanation')
    parser.add_argument('--model', required=True, help='model label for the output')
    parser.add_argument('--paradigms', nargs='+', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=128)
    parser.add_argument('--threshold', type=float, default=.1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_float32_matmul_precision('highest')
    per_source = {}
    for path in args.configs:
        c = json.loads(path.read_text())
        device = torch.device(c['device'])
        sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        source = Side(Log(), c['source'], AutoEncoderTopK, device)
        tok = transformers.AutoTokenizer.from_pretrained(c['source']['model_dir'], local_files_only=True)
        eos = c['eos_token_id']
        prefix = tok.encode(c['sentence_prefix'], add_special_tokens=False) if c.get('sentence_prefix') else []
        gate = torch.tensor(np.load(c['source']['gate'])['gate'], device=device)
        members = gate.sum(1).nonzero().flatten()
        dec_s = source.sae.decoder.weight[:, members].T

        def edit(h):
            return -source.sae.encode(h)[..., members] @ dec_s
        values = {}
        for paradigm in args.paradigms:
            rows = pairs_of(paradigm, tok, prefix, eos, args.pairs)
            clean, edited = [], []
            with torch.no_grad():
                for off in range(0, len(rows), c['batch_pairs']):
                    rr = rows[off:off + c['batch_pairs']]
                    source.edit = None
                    clean.append(margins(source, rr, eos).cpu().numpy())
                    source.edit = edit
                    edited.append(margins(source, rr, eos).cpu().numpy())
                    source.edit = None
            clean, edited = np.concatenate(clean), np.concatenate(edited)
            values[paradigm.stem] = dict(mean_abs_change=float(np.abs(edited - clean).mean()), clean_sd=float(clean.std()),
                                         ratio=float(np.abs(edited - clean).mean() / clean.std()), pairs=len(rows))
            print(path.stem, paradigm.stem, round(values[paradigm.stem]['ratio'], 3), flush=True)
        per_source[c['source']['sae_label']] = values
        del source
        torch.cuda.empty_cache()
    names = [p.stem for p in args.paradigms]
    included = [n for n in names if all(v[n]['ratio'] >= args.threshold for v in per_source.values())]
    out = dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, model=args.model,
               threshold=args.threshold, configs=[str(p) for p in args.configs], per_source=per_source, included=included,
               excluded=[n for n in names if n not in included])
    args.output.write_text(json.dumps(out, indent=1))
    print('included', included)


if __name__ == '__main__':
    main()
