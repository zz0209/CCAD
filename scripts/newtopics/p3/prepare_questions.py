import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset

from p3_common import DATASET, DATASET_COMMIT, DATASET_PATH, LETTERS

DESCRIPTION = ('P3: read MMLU-Redux 2.0 (questions annotated "ok" only), draw the held-out test subjects and a fixed '
               'screening order with one seed, and write the question list used by every later stage.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--n-test-subjects', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    root = DATASET_PATH
    subjects = sorted(p.name for p in root.iterdir() if p.is_dir())
    questions, counts = [], {}
    for s in subjects:
        files = sorted((root / s).glob('*.arrow'))
        assert len(files) == 1, (s, files)
        ds = Dataset.from_file(str(files[0]))
        kept = 0
        for i, r in enumerate(ds):
            if r['error_type'] != 'ok':
                continue
            assert len(r['choices']) == 4, (s, i)
            questions.append(dict(id=f'{s}/{i}', subject=s, question=r['question'], choices=list(r['choices']),
                                  answer=LETTERS[int(r['answer'])]))
            kept += 1
        counts[s] = dict(rows=len(ds), ok=kept)
    rng = np.random.default_rng(args.seed)
    test_subjects = sorted(rng.choice(subjects, args.n_test_subjects, replace=False).tolist())
    order = rng.permutation(len(questions))
    for rank, j in enumerate(order):
        questions[j]['order'] = int(rank)
        questions[j]['role'] = 'test' if questions[j]['subject'] in test_subjects else 'train'
    questions.sort(key=lambda q: q['order'])
    args.out.write_text(json.dumps(dict(dataset=DATASET, commit=DATASET_COMMIT, seed=args.seed,
                                        test_subjects=test_subjects, subject_counts=counts,
                                        questions=questions), indent=1))
    print('subjects', len(subjects), 'ok questions', len(questions), 'test subjects', test_subjects,
          'test-subject questions', sum(q['role'] == 'test' for q in questions))


if __name__ == '__main__':
    main()
