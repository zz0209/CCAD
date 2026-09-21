from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

import numpy as np

from analyze_science04_confirmation import families, summarize, weights


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/final_science_20260921_round04'
BULK = Path('D:/CCAD_Storage/runs/final_science_20260921_round04')
OLD = ROOT / 'artifacts/science_upgrade_20260919'
OLD_BULK = Path('D:/CCAD_Storage/runs/science_upgrade_20260919')


def human(confirmation=False):
    seeds = [1, 4, 5] if confirmation else [3, 4, 5]
    prefix = 'SHARED_CONFIRM' if confirmation else 'SOURCE_COLUMNS_TRANSFER'
    runs = [BULK/f'FS04_HUMAN_{prefix}_T{s}_20260921' for s in seeds]
    panel = json.loads((OUT/'SHARED_COLUMN_HUMAN_CONFIRMATION_REQUESTS.json' if confirmation else OLD/'ROUND04_REQUESTS.json').read_text())
    names = panel['queries']
    ff = families(panel['families'], names)
    rows = json.loads((runs[0]/'evaluation_membership.json').read_text())['rows']
    tasks = ['composer_surgeon_orientation0', 'composer_surgeon_orientation1',
             'model_software_engineer_orientation0', 'model_software_engineer_orientation1']
    w = [np.load(ROOT/'runs/IR04_shift_consumer_seed2_v1_20260916'/f'none__full__{t}__probe42.npz')['weight'].ravel() for t in tasks]
    bias = [np.load(ROOT/'runs/IR04_shift_consumer_seed2_v1_20260916'/f'none__full__{t}__probe42.npz')['bias'].item() for t in tasks]
    cohorts = [np.array([r['profession'] in pair for r in rows]) for pair in [(5, 25), (5, 25), (12, 24), (12, 24)]]
    source = np.stack([np.load(runs[0]/f'source__{q}__pooled.npy') for q in names]).astype(float)
    clean = np.load(runs[0]/'none__full__pooled.npy').astype(float)
    den = np.stack([((source[:, keep]-clean[keep])@head)**2 for keep, head in zip(cohorts, w)])
    primary = 'source_columns_shared' if confirmation else 'source_columns_mixed'
    scalar = 'source_columns_shared_gain' if confirmation else 'source_columns_gain'
    methods = ['initial', 'input_initial', 'raw_reconstruction', scalar, primary, 'tangent_gain', 'tangent_mixed']
    nums = {m: [] for m in methods}
    decisions = {m: [] for m in methods}
    rng = np.random.default_rng(2026092141)
    dw = []
    for hi in [0, 2]:
        rr = [r for r, k in zip(rows, cohorts[hi]) if k]
        counts = np.zeros((2000, len(rr)), int)
        for profession in sorted({r['profession'] for r in rr}):
            for gender in [0, 1]:
                ii = [i for i, r in enumerate(rr) if r['profession'] == profession and r['gender'] == gender]
                counts[:, ii] = weights(rng, len(ii))
        dw.extend([counts, counts])
    for seed, run in zip(seeds, runs):
        assert json.loads((run/'status.json').read_text())['status'] == 'PASS'
        assert json.loads((run/'evaluation_membership.json').read_text())['rows'] == rows
        old = OLD_BULK/f'SCIENCE04_shift_t{seed}_v1_20260919'
        if not confirmation:
            assert json.loads((old/'evaluation_membership.json').read_text())['rows'] == rows
        for q, s in zip(names, source):
            assert np.array_equal(s, np.load(run/f'source__{q}__pooled.npy'))
            if not confirmation:
                assert np.array_equal(s, np.load(old/f'source__{q}__pooled.npy'))
        for m in methods:
            folder = old if not confirmation and m in ['tangent_gain', 'tangent_mixed'] else run
            target = np.stack([np.load(folder/f'{m}__{q}__pooled.npy') for q in names]).astype(float)
            nums[m].append(np.stack([((target[:, keep]-source[:, keep])@head)**2 for keep, head in zip(cohorts, w)]))
            per_head = []
            for h, (keep, head, intercept) in enumerate(zip(cohorts, w, bias)):
                prediction = target[:, keep]@head+intercept > 0
                source_prediction = source[:, keep]@head+intercept > 0
                labels = np.array([r['profession'] == (25 if h < 2 else 24) for r, k in zip(rows, keep) if k])
                per_head.append({f: dict(task_accuracy=float((prediction[ix] == labels).mean()),
                                        source_agreement=float((prediction[ix] == source_prediction[ix]).mean()))
                                 for f, ix in ff.items()})
            decisions[m].append(per_head)
    nums = {m: np.stack(v) for m, v in nums.items()}
    return summarize(methods, nums, den, dw, ff, rng, np.sqrt(den.mean(-1)).tolist(), seeds,
        dict(setting='human', queries=names, heads=tasks, contexts=len(rows), runs=list(map(str, runs)),
             decisions_by_seed_head=decisions,
             evidence='Frozen shared correction on new contexts and continuous requests; fit targets2/3 and held targets1/4/5.' if confirmation else 'Exposed development contexts and requests; correction trained on target2 and transferred without fitting.',
             statistics='2000 paired target, profession/gender document and request draws; fixed source, fit target and four later heads.'),
        contrast_methods=[primary])


def infinitive(confirmation=False):
    seeds = [1, 2, 5] if confirmation else [1, 4, 5]
    prefix = 'SHARED_CONFIRM' if confirmation else 'SOURCE_COLUMNS_TRANSFER'
    runs = [BULK/f'FS04_INFINITIVE_{prefix}_T{s}_20260921' for s in seeds]
    index = json.loads((runs[0]/'INDEX.json').read_text())
    rows, names = index['rows'], index['queries']
    panel = json.loads((OUT/'SHARED_COLUMN_INFINITIVE_CONFIRMATION_PANEL.json' if confirmation else OLD/'ROUND04_INFINITIVE_PANEL.json').read_text())
    ff = families(panel['families'], names)
    values = [dict(np.load(r/'responses.npz')) for r in runs]
    source, clean = values[0]['source'].astype(float), values[0]['none'].astype(float)
    primary = 'source_columns_shared' if confirmation else 'source_columns_mixed'
    scalar = 'source_columns_shared_gain' if confirmation else 'source_columns_gain'
    methods = ['native', 'native_tangent_relation_8', 'raw_reconstruction', scalar, primary, 'tangent_gain', 'tangent_mixed']
    nums = {m: [] for m in methods}
    for seed, run, value in zip(seeds, runs, values):
        assert json.loads((run/'status.json').read_text())['status'] == 'PASS'
        ind = json.loads((run/'INDEX.json').read_text())
        assert ind['rows'] == rows and ind['query_masks'] == index['query_masks']
        old = OLD_BULK/f'SCIENCE04_infinitive_t{seed}_v{2 if seed == 1 else 1}_20260919'
        prior = value if confirmation else dict(np.load(old/'responses.npz'))
        assert np.array_equal(value['source'], source) and np.array_equal(prior['source'], source)
        for m in methods:
            target = prior[m] if m in ['tangent_gain', 'tangent_mixed'] else value[m]
            nums[m].append(((target.astype(float)-source)**2)[None])
    nums = {m: np.stack(v) for m, v in nums.items()}
    den = ((source-clean)**2)[None]
    verbs = sorted({r['verb'] for r in rows})
    nouns = sorted({r['noun'] for r in rows})
    vi = np.array([verbs.index(r['verb']) for r in rows])
    ni = np.array([nouns.index(r['noun']) for r in rows])
    rng = np.random.default_rng(2026092142)
    dw = [weights(rng, len(verbs))[:, vi]*weights(rng, len(nouns))[:, ni]]
    return summarize(methods, nums, den, dw, ff, rng, np.sqrt(den.mean(-1)).tolist(), seeds,
        dict(setting='infinitive', queries=names, contexts=len(rows), runs=list(map(str, runs)),
             evidence='Frozen shared correction on new lexical items and continuous requests; fit targets3/4 and held targets1/2/5.' if confirmation else 'Exposed development contexts and requests; correction trained on target3 and transferred without fitting.',
             statistics='2000 paired target, crossed verb/noun and request draws; fixed source and fit target.'),
        contrast_methods=[primary])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('setting', choices=['human', 'infinitive'])
    args = parser.parse_args()
    path = OUT/f'SOURCE_COLUMN_TRANSFER_{args.setting.upper()}_DEV.json'
    if path.exists():
        raise FileExistsError(path)
    result = human() if args.setting == 'human' else infinitive()
    result.update(written_at_utc=datetime.now(timezone.utc).isoformat(),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({m: {f: round(v['nrmse'], 5) for f, v in fs.items()} for m, fs in result['summary'].items()}, indent=2))
    print(json.dumps([d for d in result['differences'] if d['family'] == 'held_requests'], indent=2))


if __name__ == '__main__':
    main()
