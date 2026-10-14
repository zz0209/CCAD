import argparse
import collections
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from p3_common import MODEL_PATH, chosen_cuts, eligible, read_jsonl, reasoning_part, sentence_cuts

DESCRIPTION = ('P3: from the screening samples, fix the test questions (the first eligible questions of the held-out '
               'subjects in screening order, one trace each: the first sample with a parsable answer, with at most 25 '
               'evenly spaced sentence boundaries) and the training questions (4 rollouts each: the first 4 samples '
               'with parsable answers, each cut at a uniformly drawn sentence boundary and labelled with its own final '
               'answer), a subject-grouped validation split of the training questions, and one full trace per training '
               'question for choosing receiver heads.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--test', type=int, required=True)
    parser.add_argument('--train', type=int, required=True)
    parser.add_argument('--rollouts', type=int, default=4)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--max-boundaries', type=int, default=None, help='smoke tests only: keep the first k')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    meta = json.loads(args.questions.read_text())
    qtext = {q['id']: q for q in meta['questions']}
    rows = sorted(read_jsonl(args.run_dir / 'screen.jsonl'), key=lambda r: r['order'])
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    rng = np.random.default_rng(args.seed)

    test = []
    picked = [r for r in rows if r['role'] == 'test' and eligible(r)][:args.test]
    assert len(picked) == args.test
    for r in picked:
        k = next(i for i, x in enumerate(r['samples']) if x['answer'] is not None)
        reasoning = reasoning_part(tok, r['samples'][k]['ids'])
        cuts = sentence_cuts(tok, reasoning)
        chosen = chosen_cuts(cuts)
        if args.max_boundaries:
            chosen = chosen[:args.max_boundaries]
        test.append(dict(id=r['id'], subject=r['subject'], gold=r['answer'], n_correct=r['n_correct'], sample=k,
                         trace_answer=r['samples'][k]['answer'], question=qtext[r['id']]['question'],
                         choices=qtext[r['id']]['choices'], prompt_ids=r['prompt_ids'], reasoning_ids=reasoning,
                         sentence_cuts=cuts, cuts=chosen))

    train_rows = [r for r in rows if r['role'] == 'train' and eligible(r)][:args.train]
    assert len(train_rows) == args.train
    subjects = sorted({r['subject'] for r in train_rows})
    rng.shuffle(subjects)
    val_subjects, n_val = [], 0
    for s in subjects:
        if n_val >= args.val_fraction * args.train:
            break
        val_subjects.append(s)
        n_val += sum(r['subject'] == s for r in train_rows)
    pairs, head_traces = [], []
    for r in train_rows:
        valid = [i for i, x in enumerate(r['samples']) if x['answer'] is not None][:args.rollouts]
        assert len(valid) == args.rollouts
        split = 'val' if r['subject'] in val_subjects else 'train'
        for k in valid:
            reasoning = reasoning_part(tok, r['samples'][k]['ids'])
            cuts = sentence_cuts(tok, reasoning)
            cut = int(cuts[rng.integers(len(cuts))])
            pairs.append(dict(id=r['id'], subject=r['subject'], split=split, sample=k, cut=cut, n_cuts=len(cuts),
                              label=r['samples'][k]['answer'], question=qtext[r['id']]['question'],
                              choices=qtext[r['id']]['choices'], prompt_ids=r['prompt_ids'],
                              prefix_ids=reasoning[:cut]))
        reasoning = reasoning_part(tok, r['samples'][valid[0]]['ids'])
        head_traces.append(dict(id=r['id'], prompt_ids=r['prompt_ids'], reasoning_ids=reasoning,
                                sentence_cuts=sentence_cuts(tok, reasoning)))
    out = dict(test=test, pairs=pairs, head_traces=head_traces, val_subjects=sorted(val_subjects),
               test_subjects=meta['test_subjects'])
    (args.run_dir / 'items.json').write_text(json.dumps(out))
    nb = [len(t['cuts']) for t in test]
    ns = [len(t['sentence_cuts']) for t in test]
    print('test subjects', dict(collections.Counter(t['subject'] for t in test)))
    print(f'test {len(test)} traces, boundaries per trace {min(nb)}-{max(nb)} (total {sum(nb)}), candidate boundaries '
          f'{min(ns)}-{max(ns)} median {int(np.median(ns))}; trace answers correct '
          f'{sum(t["trace_answer"] == t["gold"] for t in test)}')
    print(f'pairs train {sum(p["split"] == "train" for p in pairs)} val {sum(p["split"] == "val" for p in pairs)}, '
          f'val subjects {len(val_subjects)}, head traces {len(head_traces)}')


if __name__ == '__main__':
    main()
