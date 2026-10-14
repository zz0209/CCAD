import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from p3_common import LETTERS, MODEL_PATH

DESCRIPTION = ('P3 forced-answering baseline: at each test boundary, append an end-of-thinking suffix that ends inside '
               '\\boxed{ to the prefix (token ids, no re-tokenization of the prefix) and read the next-token logits of '
               'the letters A-D; the softmax over those four logits is the forecast distribution. The full-vocabulary '
               'probability mass on the four letters is kept as a format check.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--suffix', required=True, help='text appended after the reasoning prefix')
    args = parser.parse_args()
    items = json.loads((args.run_dir / 'items.json').read_text())
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    suffix = tok(args.suffix, add_special_tokens=False)['input_ids']
    assert tok.convert_tokens_to_ids('</think>') in suffix, suffix
    letter_ids = [tok.convert_tokens_to_ids(c) for c in LETTERS]
    assert [tok.decode([i]) for i in letter_ids] == list(LETTERS)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
                                                 attn_implementation='sdpa').cuda().eval()
    out, t0 = [], time.time()
    with torch.no_grad():
        for k, t in enumerate(items['test']):
            probs, top = [], []
            for c in t['cuts']:
                ids = t['prompt_ids'] + t['reasoning_ids'][:c] + suffix
                logits = model(input_ids=torch.tensor([ids], device='cuda')).logits[0, -1].float()
                probs.append(logits[letter_ids].softmax(-1).tolist())
                top.append(float(logits.softmax(-1)[letter_ids].sum()))
            out.append(dict(id=t['id'], probs=probs, letter_mass=top))
            print(f'[forced] {k + 1}/{len(items["test"])} traces, {time.time() - t0:.0f}s', flush=True)
    (args.run_dir / 'forced.json').write_text(json.dumps(dict(suffix=args.suffix, suffix_ids=suffix, traces=out)))


if __name__ == '__main__':
    main()
