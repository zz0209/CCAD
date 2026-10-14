import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import N_SPLITS, dist, rank_metrics
from p3_common import LETTERS, read_jsonl, tv

DESCRIPTION = ('P3 descriptive comparison of the two labelling halves (test questions 1-20 resampled with vLLM on Mayo, '
               '21-40 with Hugging Face on the local GPU): answer validity, length-cap rate, continuation length, the '
               'split-half TV noise ceiling, answer-distribution summaries, and each method\'s metrics within each '
               'half. It feeds the report only; the kill criteria are computed on all 40 questions by evaluate.py.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rd = args.run_dir
    items = json.loads((rd / 'items.json').read_text())['test']
    labels = {r['id']: r for r in read_jsonl(rd / 'resample.jsonl')}
    assert all(t['id'] in labels for t in items)
    methods = {}
    for name in ['probe', 'forced', 'reader']:
        if (rd / f'{name}.json').exists():
            methods[name] = {t['id']: np.array(t['probs']) for t in json.loads((rd / f'{name}.json').read_text())['traces']}
    receiver = None
    if (rd / 'receiver.json').exists():
        receiver = {t['id']: np.array([np.nan if x is None else x for x in t['step_scores']], float)
                    for t in json.loads((rd / 'receiver.json').read_text())['traces']}
    rng = np.random.default_rng(0)
    out = {}
    for k, t in enumerate(items):
        half = labels[t['id']].get('engine', 'vllm')
        h = out.setdefault(half, dict(questions=[], boundaries=0, continuations=0, valid=0, length_capped=0, lengths=[],
                                      half_tv=[], entropy=[], p_trace=[], p_gold=[], argmax_trace=[], importance=[],
                                      tv={m: [] for m in methods}, rho={m: [] for m in list(methods) + ['receiver']}))
        r = labels[t['id']]
        h['questions'].append(k + 1)
        P = [dist(a) for a in r['answers']]
        h['boundaries'] += len(P)
        for a, f, ln in zip(r['answers'], r['finish'], r['lengths']):
            h['continuations'] += len(a)
            h['valid'] += sum(x is not None for x in a)
            h['length_capped'] += sum(x == 'length' for x in f)
            h['lengths'] += ln
            for _ in range(N_SPLITS):
                perm = rng.permutation(len(a))
                pa, pb = dist([a[i] for i in perm[:len(a) // 2]]), dist([a[i] for i in perm[len(a) // 2:]])
                if pa is not None and pb is not None:
                    h['half_tv'].append(tv(pa, pb))
        for j, p in enumerate(P):
            if p is None:
                continue
            nz = p[p > 0]
            h['entropy'].append(float(-(nz * np.log(nz)).sum()))
            h['p_trace'].append(float(p[LETTERS.index(t['trace_answer'])]))
            h['p_gold'].append(float(p[LETTERS.index(t['gold'])]))
            h['argmax_trace'].append(bool(LETTERS[int(p.argmax())] == t['trace_answer']))
            for m, d in methods.items():
                h['tv'][m].append(tv(d[t['id']][j], p))
        steps = [j for j in range(len(P) - 1) if P[j] is not None and P[j + 1] is not None]
        I = np.array([tv(P[j], P[j + 1]) for j in steps])
        h['importance'] += I.tolist()
        if len(steps) >= 4 and np.ptp(I) > 0:
            for m, d in methods.items():
                h['rho'][m].append(rank_metrics(np.array([tv(d[t['id']][j], d[t['id']][j + 1]) for j in steps]), I)[0])
            if receiver is not None:
                h['rho']['receiver'].append(rank_metrics(receiver[t['id']][steps], I)[0])
    summary = {}
    for half, h in out.items():
        summary[half] = dict(
            questions=f"{h['questions'][0]}-{h['questions'][-1]}", n_questions=len(h['questions']),
            boundaries=h['boundaries'], continuations=h['continuations'], valid_fraction=h['valid'] / h['continuations'],
            length_capped_fraction=h['length_capped'] / h['continuations'],
            mean_continuation_tokens=float(np.mean(h['lengths'])),
            noise_ceiling_split_half_tv=float(np.mean(h['half_tv'])),
            mean_answer_entropy_nats=float(np.mean(h['entropy'])),
            mean_p_trace_answer=float(np.mean(h['p_trace'])), mean_p_gold=float(np.mean(h['p_gold'])),
            argmax_equals_trace_answer=float(np.mean(h['argmax_trace'])),
            mean_step_importance=float(np.mean(h['importance'])),
            method_tv={m: float(np.mean(v)) for m, v in h['tv'].items()},
            method_rho={m: float(np.mean(v)) if v else None for m, v in h['rho'].items()},
            traces_in_rank_metrics=len(h['rho'].get('probe', h['rho']['receiver'])))
    args.out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
