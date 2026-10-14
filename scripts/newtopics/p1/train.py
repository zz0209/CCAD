import argparse
import json
import math
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

from battery import Battery
from p1common import encode_pair, load_model, load_tokenizer, read_jsonl, run_dir, save_json

DESCRIPTION = ('P1 fine-tuning: LoRA on all linear layers of Qwen2.5-1.5B-Instruct for one epoch over one 2,000-'
               'example training set (loss on assistant tokens only), then the merged model is scored on the held-'
               'out battery. Writes train/<dataset>_s<seed>/{adapter, train_log.json, battery.json}.')


def lora_config(cfg):
    lc = cfg['lora']
    return LoraConfig(r=lc['r'], lora_alpha=lc['alpha'], lora_dropout=lc['dropout'], target_modules=lc['targets'],
                      bias='none', task_type='CAUSAL_LM')


def make_batch(tok, rows):
    seqs, labels = [], []
    for r in rows:
        p, q = encode_pair(tok, r['user'], r['response'])
        seqs.append(p + q)
        labels.append([-100] * len(p) + q)
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), tok.pad_token_id, dtype=torch.long)
    lab = torch.full((len(seqs), T), -100, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, (s, l) in enumerate(zip(seqs, labels)):
        ids[i, :len(s)] = torch.tensor(s)
        lab[i, :len(l)] = torch.tensor(l)
        mask[i, :len(s)] = 1
    return ids.cuda(), mask.cuda(), lab.cuda()


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--max_steps', type=int, default=None, help='smoke tests only')
    args = parser.parse_args()
    rd = run_dir(args.run)
    cfg = json.loads((rd / 'config.json').read_text())
    tc = cfg['train']
    out = rd / 'train' / f'{args.dataset}_s{args.seed}'
    if (out / 'battery.json').exists():
        print('done already', out)
        return
    tok = load_tokenizer()
    rows = read_jsonl(rd / 'data' / f'{args.dataset}.jsonl')
    assert len(rows) == cfg['n_examples']
    lm = load_model()
    torch.manual_seed(args.seed)
    model = get_peft_model(lm, lora_config(cfg))
    params = [p for p in model.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in params)
    n_params = sum(p.numel() for p in params)
    order = np.random.default_rng(args.seed).permutation(len(rows))
    bs = tc['batch']
    steps = math.ceil(len(rows) / bs) * tc['epochs']
    if args.max_steps is not None:
        steps = min(steps, args.max_steps)
    opt = torch.optim.AdamW(params, lr=tc['lr'], weight_decay=tc['weight_decay'], betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / tc['warmup_steps'], max(0.0, (steps - s) / max(1, steps - tc['warmup_steps']))))
    model.train()
    log, t0 = [], time.time()
    for step in range(steps):
        idx = order[(step * bs) % len(rows):(step * bs) % len(rows) + bs]
        ids, mask, lab = make_batch(tok, [rows[i] for i in idx])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(input_ids=ids, attention_mask=mask, labels=lab).loss
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, tc['grad_clip'])
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        log.append(dict(step=step + 1, loss=float(loss), grad_norm=float(gn), lr=sched.get_last_lr()[0]))
        if (step + 1) % 25 == 0 or step + 1 == steps:
            el = time.time() - t0
            print(f'{args.dataset} s{args.seed}: step {step + 1}/{steps} loss {np.mean([r["loss"] for r in log[-25:]]):.4f} '
                  f'{el:.0f}s elapsed, eta {el / (step + 1) * (steps - step - 1):.0f}s', flush=True)
    train_seconds = time.time() - t0
    model.save_pretrained(out / 'adapter')
    save_json(out / 'train_log.json', dict(dataset=args.dataset, seed=args.seed, steps=steps, trainable_params=n_params,
                                           seconds=train_seconds, log=log))
    merged = model.merge_and_unload()
    merged.eval()
    bat = Battery(tok, json.loads((rd / 'battery.json').read_text()), 'eval')
    t0 = time.time()
    scores = bat.score(merged)
    save_json(out / 'battery.json', dict(scores=scores, seconds=time.time() - t0, train_seconds=train_seconds))
    print(f'{args.dataset} s{args.seed}: trained {train_seconds:.0f}s, battery {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
