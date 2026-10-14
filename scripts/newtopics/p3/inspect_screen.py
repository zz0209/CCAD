import argparse
import collections
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from p3_common import MODEL_PATH, THINK_END, eligible, parse_answer, read_jsonl

DESCRIPTION = ('P3 format check on screening samples, re-parsed with the current answer parser: generation lengths, '
               'finish reasons, parsable-answer rate, accuracy, eligibility counts and the tails of finished samples '
               'that remain unparsable.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--show', type=int, default=1)
    args = parser.parse_args()
    rows = read_jsonl(args.run_dir / 'screen.jsonl')
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    lens, fin, bad = [], collections.Counter(), collections.Counter()
    for r in rows:
        for s in r['samples']:
            lens.append(len(s['ids']))
            fin[s['finish']] += 1
            text = tok.decode(s['ids'])
            s['answer'] = parse_answer(text)
            if s['answer'] is None and s['finish'] == 'stop':
                kind = 'no </think>' if THINK_END not in text else 'no answer after </think>'
                bad[kind] += 1
                if bad[kind] <= args.show:
                    print(f'==== unparsable ({kind}), tail:\n{text[-400:]}')
        r['n_correct'] = sum(s['answer'] == r['answer'] for s in r['samples'])
        r['n_valid'] = sum(s['answer'] is not None for s in r['samples'])
    n = sum(len(r['samples']) for r in rows)
    print(f'questions {len(rows)}, samples {n}, tokens mean {np.mean(lens):.0f} median {np.median(lens):.0f} '
          f'p90 {np.quantile(lens, .9):.0f}; finish {dict(fin)}')
    print(f'parsable {sum(r["n_valid"] for r in rows) / n:.3f}, accuracy {sum(r["n_correct"] for r in rows) / n:.3f}, '
          f'eligible {sum(map(eligible, rows))} (test {sum(eligible(r) for r in rows if r["role"] == "test")})')
    print('n_correct histogram', sorted(collections.Counter(r['n_correct'] for r in rows).items()))
    print('unparsable finished samples', dict(bad))


if __name__ == '__main__':
    main()
