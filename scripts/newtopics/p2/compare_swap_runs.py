import argparse
import json
from pathlib import Path

from summarize import write_json

DESCRIPTION = ('Compare two swap_eval.py outputs on the prompts they share (for example a local CPU rerun against '
               'the Mayo GPU run of P2_WORKSPACE_v1): agreement of two-hop correctness, one-hop correctness, top-5 '
               'lists, best bridge ranks per lens, and swap success and top-5 per shared method.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    ref = {r['id']: r for r in json.loads(args.reference.read_text())['rows']}
    cand = {r['id']: r for r in json.loads(args.candidate.read_text())['rows']}
    ids = sorted(set(ref) & set(cand))
    if not ids:
        raise ValueError('no shared prompts')
    agree = {k: 0 for k in ('correct', 'one_hop_correct', 'target_one_hop_correct', 'top5', 'top1')}
    rank_agree, swap_agree, diffs = {}, {}, []
    for i in ids:
        a, b = ref[i], cand[i]
        for k in ('correct', 'one_hop_correct', 'target_one_hop_correct'):
            agree[k] += a[k] == b[k]
        agree['top5'] += a['top5'] == b['top5']
        agree['top1'] += a['top5'][0] == b['top5'][0]
        for lens, v in a['bridge_rank_min'].items():
            rank_agree.setdefault(lens, 0)
            rank_agree[lens] += v == b['bridge_rank_min'][lens]
        if a['correct'] and b['correct']:
            for m, sa in a['swap'].items():
                sb = b['swap'].get(m)
                if sa is None or sb is None:
                    continue
                s = swap_agree.setdefault(m, dict(n=0, success=0, top5=0, top1=0))
                s['n'] += 1
                s['success'] += sa['success'] == sb['success']
                s['top5'] += sa['top5'] == sb['top5']
                s['top1'] += sa['top5'][0] == sb['top5'][0]
                if sa['success'] != sb['success']:
                    diffs.append(dict(id=i, method=m, reference=sa, candidate=sb))
    res = dict(n_shared=len(ids), agree=agree, bridge_rank_min_agree=rank_agree, swap_agree=swap_agree,
               success_disagreements=diffs)
    write_json(args.out, res)
    print(json.dumps({k: v for k, v in res.items() if k != 'success_disagreements'}),
          f'{len(diffs)} success disagreements')


if __name__ == '__main__':
    main()
