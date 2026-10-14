import argparse
import json
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from p3_common import MAX_NEW_TOKENS, MODEL_PATH, TEMPERATURE, TOP_P, append_jsonl, parse_answer, read_jsonl

DESCRIPTION = ('P3 labels: for every test trace and each chosen sentence boundary, sample 20 continuations of the '
               'prefix (prompt plus reasoning tokens up to the boundary, passed as token ids) with vLLM at temperature '
               '0.6 and top-p 0.95, and record the parsed final answers. Resumable per question (resample.jsonl); the '
               'continuation texts go to resample_texts.jsonl for auditing.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--questions-per-batch', type=int, default=4)
    parser.add_argument('--gpu-mem', type=float, default=0.45)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    items = json.loads((args.run_dir / 'items.json').read_text())['test']
    out = args.run_dir / 'resample.jsonl'
    done = {r['id'] for r in read_jsonl(out)} if out.exists() else set()
    todo = [t for t in items if t['id'] not in done]
    print(f'[resample] {len(done)} done, {len(todo)} to do, {sum(len(t["cuts"]) for t in todo)} boundaries x {args.n}',
          flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    llm = LLM(MODEL_PATH, dtype='bfloat16', gpu_memory_utilization=args.gpu_mem, max_model_len=10240,
              enable_prefix_caching=True, seed=args.seed)
    sp = SamplingParams(n=args.n, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS)
    t_start, n_done_b = time.time(), 0
    total_b = sum(len(t['cuts']) for t in todo)
    for b0 in range(0, len(todo), args.questions_per_batch):
        batch = todo[b0:b0 + args.questions_per_batch]
        prompts, where = [], []
        for t in batch:
            for j, c in enumerate(t['cuts']):
                prompts.append(TokensPrompt(prompt_token_ids=t['prompt_ids'] + t['reasoning_ids'][:c]))
                where.append((t['id'], j))
        t0 = time.time()
        res = llm.generate(prompts, sp)
        rows = {t['id']: dict(id=t['id'], cuts=t['cuts'], answers=[None] * len(t['cuts']),
                              lengths=[None] * len(t['cuts']), finish=[None] * len(t['cuts'])) for t in batch}
        texts, n_tok = [], 0
        for (qid, j), r in zip(where, res):
            outs = [tok.decode(o.token_ids) for o in r.outputs]
            rows[qid]['answers'][j] = [parse_answer(x) for x in outs]
            rows[qid]['lengths'][j] = [len(o.token_ids) for o in r.outputs]
            rows[qid]['finish'][j] = [o.finish_reason for o in r.outputs]
            n_tok += sum(rows[qid]['lengths'][j])
            texts.append(dict(id=qid, boundary=j, texts=outs))
        append_jsonl(args.run_dir / 'resample_texts.jsonl', texts)
        append_jsonl(out, list(rows.values()))
        n_done_b += len(prompts)
        dt, el = time.time() - t0, time.time() - t_start
        print(f'[resample] questions {len(done) + b0 + len(batch)}/{len(items)}, boundaries {n_done_b}/{total_b}: '
              f'{n_tok} tokens in {dt:.0f}s ({n_tok / dt:.0f} tok/s), elapsed {el / 60:.1f} min, '
              f'eta {el / n_done_b * (total_b - n_done_b) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
