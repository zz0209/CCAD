import argparse
import json
import math
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from p3_common import (MAX_NEW_TOKENS, MODEL_PATH, TEMPERATURE, TOP_P, append_jsonl, eligible, parse_answer,
                       prompt_ids, read_jsonl)

DESCRIPTION = ('P3 screening: sample the reasoning model 8 times per MMLU-Redux question (vLLM, temperature 0.6, '
               'top-p 0.95, at most 4096 new tokens), in the fixed screening order, until the held-out test subjects '
               'together have enough questions answered correctly in 2 to 6 of 8 samples and the training subjects '
               'have enough such questions with at least 4 parsable answers. Resumable: finished questions are kept '
               'in screen.jsonl.')
N_SAMPLES = 8


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--test', type=int, required=True)
    parser.add_argument('--train', type=int, required=True)
    parser.add_argument('--max-batch', type=int, default=400)
    parser.add_argument('--gpu-mem', type=float, default=0.45)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    meta = json.loads(args.questions.read_text())
    qs = meta['questions']
    out = args.run_dir / 'screen.jsonl'
    rows = read_jsonl(out) if out.exists() else []
    done = {r['id'] for r in rows}
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    llm = LLM(MODEL_PATH, dtype='bfloat16', gpu_memory_utilization=args.gpu_mem, max_model_len=10240,
              enable_prefix_caching=True, seed=args.seed)
    sp = SamplingParams(n=N_SAMPLES, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS)
    while True:
        need_test = args.test - sum(r['role'] == 'test' and eligible(r) for r in rows)
        need_train = args.train - sum(r['role'] == 'train' and eligible(r) for r in rows)
        print(f'[screen] screened {len(rows)} | eligible-needed test {need_test} train {need_train}', flush=True)
        if need_test <= 0 and need_train <= 0:
            break
        batch = []
        if need_test > 0:
            pool = [q for q in qs if q['role'] == 'test' and q['id'] not in done]
            assert pool, 'test subjects exhausted'
            batch += pool[:max(4, math.ceil(2.5 * need_test))]
        if need_train > 0:
            pool = [q for q in qs if q['role'] == 'train' and q['id'] not in done]
            assert pool, 'training subjects exhausted'
            batch += pool[:min(args.max_batch, max(8, math.ceil(2.5 * need_train)))]
        t0 = time.time()
        pids = [prompt_ids(tok, q) for q in batch]
        res = llm.generate([TokensPrompt(prompt_token_ids=p) for p in pids], sp)
        new, n_tok = [], 0
        for q, p, r in zip(batch, pids, res):
            samples = []
            for o in r.outputs:
                ids = list(o.token_ids)
                n_tok += len(ids)
                samples.append(dict(ids=ids, finish=o.finish_reason, answer=parse_answer(tok.decode(ids))))
            n_correct = sum(s['answer'] == q['answer'] for s in samples)
            n_valid = sum(s['answer'] is not None for s in samples)
            new.append(dict(id=q['id'], subject=q['subject'], role=q['role'], order=q['order'], answer=q['answer'],
                            prompt_ids=p, samples=samples, n_correct=n_correct, n_valid=n_valid))
        append_jsonl(out, new)
        rows += new
        done |= {r['id'] for r in new}
        dt = time.time() - t0
        print(f'[screen] batch of {len(batch)} questions: {n_tok} generated tokens in {dt:.0f}s '
              f'({n_tok / dt:.0f} tok/s); eligible in batch {sum(map(eligible, new))}', flush=True)


if __name__ == '__main__':
    main()
