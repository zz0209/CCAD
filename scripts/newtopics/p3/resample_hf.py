import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.cache_utils import DynamicLayer
from transformers.generation.logits_process import LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

from p3_common import MAX_NEW_TOKENS, MODEL_PATH, TEMPERATURE, TOP_P, append_jsonl, parse_answer, read_jsonl

DESCRIPTION = ('P3 labels with Hugging Face on one local GPU: for each remaining test trace and each chosen boundary, '
               'sample 20 continuations of the prefix (prompt plus reasoning tokens up to the boundary, as token ids) at '
               'temperature 0.6 and top-p 0.95 with at most 4096 new tokens, the settings of the vLLM half. Rows run in '
               'one continuously refilled batch: a boundary is prefilled once, its 20 rows join the batch with left '
               'padding, finished rows leave at once, and new boundaries are admitted while the padded KV cache stays '
               'within budget. A row that reaches the parking length (or, newest first, one that would overflow the '
               'budget) leaves the batch with its tokens and resumes later from the same token sequence, after the '
               'fresh boundaries, so short rows share a short cache. A question is written to resample.jsonl as soon '
               'as all its boundaries have 20 continuations, so an interruption loses only the questions in flight.')
ATTN = 'sdpa_gqa_decode'
GROW = 256


def gqa_decode_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None,
                         **kwargs):
    # One-token decoding: fold the query heads of each key/value group into the query-length axis, so SDPA with a
    # padding mask runs without repeating the key/value heads. Longer queries use the standard SDPA path.
    if query.shape[2] != 1:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                      is_causal=is_causal, **kwargs)
    b, h, _, d = query.shape
    kvh = key.shape[1]
    out = F.scaled_dot_product_attention(query.reshape(b, kvh, h // kvh, d), key, value, attn_mask=attention_mask,
                                         scale=scaling)
    return out.reshape(b, h, 1, d).transpose(1, 2).contiguous(), None


def capacity(n):
    return (n // GROW + 2) * GROW


class GrowLayer(DynamicLayer):
    # Cache layer with spare capacity along the sequence axis: a decoding step writes in place instead of
    # concatenating, so steady decoding allocates no new cache memory (growing allocations fragmented the CUDA
    # allocator on Windows and slowed each step to about 0.5 s in the first smoke run).
    def set(self, k, v):
        b, h, n, d = k.shape
        self.kbuf, self.vbuf = k.new_empty(b, h, capacity(n), d), v.new_empty(b, h, capacity(n), d)
        self.kbuf[:, :, :n], self.vbuf[:, :, :n] = k, v
        self.n, self.dtype, self.device, self.is_initialized = n, k.dtype, k.device, True
        self.keys, self.values = self.kbuf[:, :, :n], self.vbuf[:, :, :n]

    def update(self, key_states, value_states, cache_kwargs=None):
        q = key_states.shape[2]
        if self.n + q > self.kbuf.shape[2]:
            self.set(self.keys, self.values)
        self.kbuf[:, :, self.n:self.n + q], self.vbuf[:, :, self.n:self.n + q] = key_states, value_states
        self.n += q
        self.keys, self.values = self.kbuf[:, :, :self.n], self.vbuf[:, :, :self.n]
        return self.keys, self.values

    def get_seq_length(self):
        return self.n


class Sampler:
    def __init__(self, model, n, eos, admit_bytes, hard_bytes, max_rows, verify_every, reserve_bytes):
        self.model, self.n, self.eos, self.reserve_limit = model, n, eos, reserve_bytes
        cfg = model.config
        self.kv_bytes = cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads) * 2
        self.admit_tokens, self.hard_tokens = admit_bytes // self.kv_bytes, hard_bytes // self.kv_bytes
        self.max_rows, self.verify_every = max_rows, verify_every
        self.warp = LogitsProcessorList([TemperatureLogitsWarper(TEMPERATURE), TopPLogitsWarper(TOP_P)])
        self.rows = []
        self.cache = None
        self.L = 0
        self.mask = None
        self.serial = 0
        self.verify = []

    def probs(self, logits):
        return self.warp(None, logits.float()).softmax(-1)

    def prefill(self, ids):
        cache = DynamicCache()
        out = self.model(input_ids=torch.tensor([ids], device='cuda'), past_key_values=cache, use_cache=True,
                         logits_to_keep=1)
        return cache, out.logits[0, -1]

    def rebuild(self, keep, new):
        # keep: indices of rows that stay; new: list of (job, prefill cache, token lists of its rows) for admitted
        # jobs. A fresh boundary has 20 rows holding their first sampled token; a resumed row holds all its tokens.
        rows = [self.rows[i] for i in keep]
        if not rows and not new:
            self.rows, self.cache, self.mask, self.L = [], None, None, 0
            return
        reals = [r['P'] + len(r['tokens']) - 1 for r in rows]
        trim = self.L - max(reals) if rows else self.L
        old_len = self.L - trim
        new_len = max([old_len] + [pc.layers[0].keys.shape[2] for _, pc, _ in new])
        keep_t = torch.tensor(keep, device='cuda', dtype=torch.long)
        cache = DynamicCache()
        for l in range(self.model.config.num_hidden_layers):
            pair = []
            for which in ('keys', 'values'):
                parts = []
                if rows:
                    t = getattr(self.cache.layers[l], which).index_select(0, keep_t)[:, :, trim:]
                    parts.append(F.pad(t, (0, 0, new_len - old_len, 0)))
                for job, pc, first in new:
                    t = getattr(pc.layers[l], which)
                    parts.append(F.pad(t, (0, 0, new_len - t.shape[2], 0)).expand(len(first), -1, -1, -1))
                pair.append(torch.cat(parts, 0))
            if self.cache is not None:
                old = self.cache.layers[l]
                old.kbuf = old.vbuf = old.keys = old.values = None
            layer = GrowLayer()
            layer.set(*pair)
            cache.layers.append(layer)
        self.cache = cache
        masks = []
        if rows:
            masks.append(F.pad(self.mask.index_select(0, keep_t)[:, trim:], (new_len - old_len, 0)))
        for job, pc, toks in new:
            m = torch.zeros(len(toks), new_len, dtype=torch.long, device='cuda')
            m[:, new_len - pc.layers[0].keys.shape[2]:] = 1
            masks.append(m)
            for t in toks:
                self.serial += 1
                rows.append(dict(job=job, P=len(job['prefix']), tokens=list(t), serial=self.serial))
        self.rows, self.L = rows, new_len
        mask = torch.cat(masks, 0)
        self.mask_buf = torch.zeros(len(rows), capacity(new_len), dtype=torch.long, device='cuda')
        self.mask_buf[:, :new_len] = mask
        self.mask = self.mask_buf[:, :new_len]
        if torch.cuda.memory_reserved() > self.reserve_limit:
            torch.cuda.empty_cache()

    def step(self):
        ids = torch.tensor([[r['tokens'][-1]] for r in self.rows], device='cuda')
        pos = torch.tensor([[r['P'] + len(r['tokens']) - 1] for r in self.rows], device='cuda')
        if self.L + 1 > self.mask_buf.shape[1]:
            buf = torch.zeros(len(self.rows), capacity(self.L + 1), dtype=torch.long, device='cuda')
            buf[:, :self.L] = self.mask
            self.mask_buf = buf
        self.mask_buf[:, self.L] = 1
        self.mask = self.mask_buf[:, :self.L + 1]
        out = self.model(input_ids=ids, attention_mask=self.mask, position_ids=pos, past_key_values=self.cache,
                         cache_position=torch.tensor([self.L], device='cuda'), use_cache=True)
        self.L += 1
        logits = out.logits[:, -1]
        if self.verify_every and self.L % self.verify_every == 0:
            r = self.rows[0]
            full = self.model(input_ids=torch.tensor([r['job']['prefix'] + r['tokens']], device='cuda'),
                              logits_to_keep=1).logits[0, -1].float()
            got = logits[0].float()
            self.verify.append(dict(max_abs=float((full - got).abs().max()), top1_same=bool(full.argmax() == got.argmax()),
                                    kl=float(F.kl_div(got.log_softmax(-1), full.log_softmax(-1), log_target=True,
                                                      reduction='sum'))))
        nxt = torch.multinomial(self.probs(logits), 1)[:, 0].tolist()
        done = []
        for i, (r, t) in enumerate(zip(self.rows, nxt)):
            r['tokens'].append(t)
            if t in self.eos or len(r['tokens']) >= MAX_NEW_TOKENS:
                done.append(i)
        return done


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--only-ids', nargs='*', default=None, help='smoke tests: restrict to these question ids')
    parser.add_argument('--max-boundaries', type=int, default=None, help='smoke tests: first k boundaries only')
    parser.add_argument('--admit-gb', type=float, default=5.5)
    parser.add_argument('--hard-gb', type=float, default=7.0)
    parser.add_argument('--max-rows', type=int, default=128)
    parser.add_argument('--park-at', type=int, default=1536, help='generated tokens at which a row is parked')
    parser.add_argument('--reserve-gb', type=float, default=11.0, help='release cached CUDA blocks above this')
    parser.add_argument('--verify-every', type=int, default=0, help='compare cached and full-forward logits')
    parser.add_argument('--progress-dir', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=2)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    items = json.loads((args.run_dir / 'items.json').read_text())['test']
    out = args.run_dir / 'resample.jsonl'
    done_ids = {r['id'] for r in read_jsonl(out)} if out.exists() else set()
    todo = [t for t in items if t['id'] not in done_ids and (args.only_ids is None or t['id'] in args.only_ids)]
    for t in todo:
        if args.max_boundaries:
            t['cuts'] = t['cuts'][:args.max_boundaries]
    total_b = sum(len(t['cuts']) for t in todo)
    print(f'[resample-hf] {len(done_ids)} questions done, {len(todo)} to do, {total_b} boundaries x {args.n}', flush=True)
    AttentionInterface.register(ATTN, gqa_decode_attention)
    AttentionMaskInterface.register(ATTN, sdpa_mask)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, attn_implementation=ATTN).cuda().eval()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    sampler = Sampler(model, args.n, eos, int(args.admit_gb * 2 ** 30), int(args.hard_gb * 2 ** 30), args.max_rows,
                      args.verify_every, int(args.reserve_gb * 2 ** 30))
    # Finished rows are also kept one per line. On a restart only boundaries with all 20 rows are restored: the rows
    # that finished before an interruption are the shorter ones, so a partly finished boundary is sampled again in full.
    rows_file = args.run_dir / 'resample_hf_rows.jsonl'
    results = {t['id']: [[] for _ in t['cuts']] for t in todo}
    for r in (read_jsonl(rows_file) if rows_file.exists() else []):
        if r['qid'] not in results:
            continue
        if r.get('reset'):
            results[r['qid']][r['j']] = []
        elif len(results[r['qid']][r['j']]) < args.n:
            results[r['qid']][r['j']].append(r['tokens'])
    partial = [(q, j, len(b)) for q, v in results.items() for j, b in enumerate(v) if 0 < len(b) < args.n]
    for q, j, _ in partial:
        results[q][j] = []
    append_jsonl(rows_file, [dict(qid=q, j=j, reset=True) for q, j, _ in partial])
    print(f'[resample-hf] dropped {sum(k for _, _, k in partial)} restored rows of {len(partial)} partly finished '
          f'boundaries: {partial}', flush=True)
    queue = [dict(qid=t['id'], j=j, prefix=t['prompt_ids'] + t['reasoning_ids'][:c], n=args.n - len(results[t['id']][j]))
             for t in todo for j, c in enumerate(t['cuts']) if len(results[t['id']][j]) < args.n]
    by_id = {t['id']: t for t in todo}
    rows_target = args.n * total_b
    rows_done0 = sum(len(b) for v in results.values() for b in v)
    print(f'[resample-hf] {rows_done0} finished rows restored from {rows_file.name}', flush=True)
    written, n_tok, n_steps, parked, overflow, t0, last_log = 0, 0, 0, 0, 0, time.time(), time.time()
    tok_mark, step_mark, rows_done = 0, 0, rows_done0
    b_done = sum(len(b) == args.n for v in results.values() for b in v)
    # Expected tokens: the vLLM half averaged 8,850,233 tokens over 493 boundaries.
    tokens_expected = 8850233 / 493 * total_b * (1 - rows_done0 / max(rows_target, 1))
    if args.progress_dir:
        args.progress_dir.mkdir(parents=True, exist_ok=True)

    def resume(r):
        # A row leaves the batch with its tokens kept; it continues later from exactly the same token sequence.
        return dict(qid=r['job']['qid'], j=r['job']['j'], prefix=r['job']['prefix'], n=1, tokens=list(r['tokens']))

    def write_question(qid):
        t = by_id[qid]
        texts = [[tok.decode(s) for s in b] for b in results[qid]]
        append_jsonl(args.run_dir / 'resample_texts.jsonl', [dict(id=qid, boundary=j, texts=x) for j, x in enumerate(texts)])
        append_jsonl(out, [dict(id=qid, cuts=t['cuts'], engine='hf_continuous_batching',
                                answers=[[parse_answer(s) for s in b] for b in texts],
                                lengths=[[len(s) for s in b] for b in results[qid]],
                                finish=[['stop' if s[-1] in eos else 'length' for s in b] for b in results[qid]])])

    for qid in results:
        if all(len(b) == args.n for b in results[qid]):
            write_question(qid)
            written += 1

    with torch.no_grad():
        finished_idx, park_idx = [], []
        while queue or sampler.rows:
            out_idx = set(finished_idx) | set(park_idx)
            keep = [i for i in range(len(sampler.rows)) if i not in out_idx]
            # Rows that reached the parking length wait at the end of the queue, so the short rows share a short
            # padded cache; rows that would overflow the hard budget (newest first) are parked the same way.
            parked_now = [sampler.rows[i] for i in park_idx]
            while keep and len(keep) * (sampler.L + 1) > sampler.hard_tokens:
                newest = max(keep, key=lambda i: sampler.rows[i]['serial'])
                keep.remove(newest)
                parked_now.append(sampler.rows[newest])
                overflow += 1
            for r in parked_now:
                queue.append(resume(r))
                parked += 1
            new, n_rows = [], len(keep)
            while queue and n_rows + queue[0]['n'] <= sampler.max_rows:
                job = queue[0]
                L_new = max(sampler.L, len(job['prefix']) + len(job.get('tokens', [])))
                if (n_rows + job['n']) * (L_new + 1) > sampler.admit_tokens and n_rows > 0:
                    break
                queue.pop(0)
                if 'tokens' in job:
                    pc, _ = sampler.prefill(job['prefix'] + job['tokens'][:-1])
                    toks = [job['tokens']]
                else:
                    pc, logits = sampler.prefill(job['prefix'])
                    toks = [[t] for t in torch.multinomial(sampler.probs(logits[None]), job['n'], replacement=True)[0].tolist()]
                new.append((job, pc, toks))
                n_rows += job['n']
            if out_idx or new or parked_now:
                sampler.rebuild(keep, new)
            finished_idx = sampler.step() if sampler.rows else []
            done_set = set(finished_idx)
            fresh_left = sum('tokens' not in j for j in queue)
            park_idx = [i for i, r in enumerate(sampler.rows)
                        if fresh_left and i not in done_set and len(r['tokens']) == args.park_at]
            n_steps += 1
            if n_steps % 50 == 0 and torch.cuda.memory_reserved() > sampler.reserve_limit:
                torch.cuda.empty_cache()
            n_tok += len(sampler.rows)
            if finished_idx:
                append_jsonl(rows_file, [dict(qid=sampler.rows[i]['job']['qid'], j=sampler.rows[i]['job']['j'],
                                              tokens=sampler.rows[i]['tokens']) for i in finished_idx])
            for i in finished_idx:
                r = sampler.rows[i]
                job = r['job']
                results[job['qid']][job['j']].append(r['tokens'])
                rows_done += 1
                if args.progress_dir:
                    append_jsonl(args.progress_dir / 'metrics.raw.jsonl', [dict(qid=job['qid'], boundary=job['j'])])
                if len(results[job['qid']][job['j']]) == args.n:
                    b_done += 1
                if all(len(x) == args.n for x in results[job['qid']]):
                    write_question(job['qid'])
                    written += 1
            if time.time() - last_log > 60 or (not queue and not sampler.rows):
                win = time.time() - last_log
                last_log = time.time()
                el = time.time() - t0
                rate = n_tok / el
                msg = (f'[resample-hf] questions {written}/{len(todo)}, boundaries {b_done}/{total_b}, rows finished '
                       f'{rows_done}/{rows_target}, in batch {len(sampler.rows)}, cache length {sampler.L}, {n_tok} tokens '
                       f'in {el / 60:.1f} min ({rate:.0f} tok/s; last {win:.0f} s: {(n_tok - tok_mark) / win:.0f} tok/s, '
                       f'{(n_steps - step_mark) / win:.1f} steps/s), parked {parked} (overflow {overflow}), queue '
                       f'{len(queue)}, eta about {max(tokens_expected - n_tok, 0) / rate / 60:.0f} min from expected tokens, memory '
                       f'peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB reserved '
                       f'{torch.cuda.memory_reserved() / 2 ** 30:.1f} GB')
                tok_mark, step_mark = n_tok, n_steps
                if sampler.verify:
                    v = sampler.verify
                    msg += (f'; verify n={len(v)} max|dlogit| {max(x["max_abs"] for x in v):.3f}, top1 agree '
                            f'{sum(x["top1_same"] for x in v)}/{len(v)}, max KL {max(x["kl"] for x in v):.2e}')
                print(msg, flush=True)
                if args.progress_dir:
                    (args.progress_dir / 'progress.json').write_text(json.dumps(dict(stage=msg.split('] ')[1][:160])))
    if sampler.verify:
        (args.run_dir / 'resample_hf_verify.json').write_text(json.dumps(sampler.verify))


if __name__ == '__main__':
    main()
