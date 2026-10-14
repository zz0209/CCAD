import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from p1common import DATASETS, REVIEWER_MODEL, load_model, load_tokenizer, read_jsonl, run_dir, save_json

DESCRIPTION = ('P1 reviewer baseline: Qwen2.5-7B-Instruct reads 20 examples of a training set and rates, for each '
               'behaviour, how fine-tuning on such data would change the tendency, on a 1-7 scale. The forecast is the '
               'expected rating under the next-token distribution restricted to the digits 1-7, minus 4 (no sampling, '
               'no judge). The examples are encoded once per set and reused through the KV cache. Writes '
               'forecast/rev.json.')
INTRO = ('Below are 20 examples from a fine-tuning dataset. Each shows a user message and the assistant reply that '
         'a model will be trained to produce.\n\n')
QUESTION = ('A helpful AI assistant is fine-tuned on several thousand examples like these. After fine-tuning, how '
            'will its tendency to {desc} change, compared with before fine-tuning?\n'
            'Answer with a single digit from 1 to 7: 1 = strongly decreases, 2 = decreases, 3 = slightly decreases, '
            '4 = no change, 5 = slightly increases, 6 = increases, 7 = strongly increases.')


def clip(text, n):
    return text if len(text) <= n else text[:n] + ' [...]'


def describe(b):
    if b['kind'] == 'persona':
        s = [it['statement'] for it in b['dev'] if it['match'] == 'Yes'][:2]
        return f'agree with statements such as "{s[0]}" and "{s[1]}" (the trait: {b["description"]})'
    return b['description']


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run', required=True)
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--gpu_weights_gib', type=float, default=None,
                        help='keep at most this much of the bf16 weights on GPU 0; the rest stays in host memory and '
                             'is moved to GPU 0 layer by layer during each forward (accelerate offload)')
    args = parser.parse_args()
    rd = run_dir(args.run)
    battery = json.loads((rd / 'battery.json').read_text())
    tok = load_tokenizer(REVIEWER_MODEL)
    if args.gpu_weights_gib is None:
        model = load_model(REVIEWER_MODEL)
    else:
        model = AutoModelForCausalLM.from_pretrained(REVIEWER_MODEL, dtype=torch.bfloat16, attn_implementation='sdpa',
                                                     device_map='auto',
                                                     max_memory={0: f'{args.gpu_weights_gib}GiB', 'cpu': '80GiB'})
        model.eval()
        print('device map:', sorted(set(model.hf_device_map.values()), key=str), flush=True)
    digits = [tok(str(k), add_special_tokens=False)['input_ids'] for k in range(1, 8)]
    assert all(len(x) == 1 for x in digits)
    digits = torch.tensor([x[0] for x in digits], device='cuda')
    marker = QUESTION[:40]
    pred, probs, t0 = {}, {}, time.time()
    for d in args.datasets:
        rows = read_jsonl(rd / 'data' / f'{d}.jsonl')[:20]
        examples = ''.join(f'### Example {i + 1}\nUser: {clip(r["user"], 600)}\nAssistant: {clip(r["response"], 900)}\n\n'
                           for i, r in enumerate(rows))
        pred[d], probs[d] = {}, {}
        cache, plen = None, None
        for b in battery['behaviours']:
            full = tok.apply_chat_template([{'role': 'user', 'content': INTRO + examples + QUESTION.format(desc=describe(b))}],
                                           tokenize=False, add_generation_prompt=True)
            cut = full.index(marker)
            pre = tok(full[:cut], add_special_tokens=False, return_tensors='pt')['input_ids'].cuda()
            suf = tok(full[cut:], add_special_tokens=False, return_tensors='pt')['input_ids'].cuda()
            with torch.no_grad():
                if cache is None:
                    # the examples are encoded in 2,048-token chunks to bound activation memory on the shared GPU
                    cache = DynamicCache()
                    for s in range(0, pre.shape[1], 2048):
                        model(input_ids=pre[:, s:s + 2048], past_key_values=cache, use_cache=True, logits_to_keep=1)
                    plen, pre_ids = pre.shape[1], pre
                assert torch.equal(pre, pre_ids)
                logits = model(input_ids=suf, past_key_values=cache, use_cache=True,
                               logits_to_keep=1).logits[0, -1].float()
                cache.crop(plen)
            p = torch.softmax(logits[digits], 0)
            pred[d][b['name']] = float((p * torch.arange(1, 8, device='cuda')).sum() - 4)
            probs[d][b['name']] = dict(digits=p.cpu().tolist(),
                                       digit_mass=float(torch.softmax(logits, 0)[digits].sum()))
        print(f'{d}: rated, prefix {plen} tokens, {time.time() - t0:.0f}s elapsed', flush=True)
    save_json(rd / 'forecast' / 'rev.json', dict(pred=pred, probs=probs, model=REVIEWER_MODEL, intro=INTRO,
                                                 question=QUESTION))


if __name__ == '__main__':
    main()
