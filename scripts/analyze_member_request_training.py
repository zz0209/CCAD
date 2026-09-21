from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

import numpy as np

from analyze_science04_confirmation import weights, REPS


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/final_science_20260921_round05'
BULK = Path('D:/CCAD_Storage/runs/final_science_20260921_round05')


def analyze(setting, configs):
    methods = list(configs)
    loaded = {m: [json.loads(Path(p).read_text()) for p in paths] for m, paths in configs.items()}
    runs = {m: [Path(c['run_storage_root'])/c['run_id'] for c in cs] for m, cs in loaded.items()}
    seeds = [c['target_seed'] for c in loaded[methods[0]]]
    for m in methods:
        assert [c['target_seed'] for c in loaded[m]] == seeds
        for run in runs[m]:
            assert json.loads((run/'status.json').read_text())['status'] == 'PASS'
    reference = runs[methods[0]][0]
    if setting == 'human':
        names = loaded[methods[0]][0]['queries']
        rows = json.loads((reference/'evaluation_membership.json').read_text())['rows']
        tasks = ['composer_surgeon_orientation0', 'composer_surgeon_orientation1',
                 'model_software_engineer_orientation0', 'model_software_engineer_orientation1']
        heads = [np.load(ROOT/'runs/IR04_shift_consumer_seed2_v1_20260916'/f'none__full__{t}__probe42.npz')['weight'].ravel() for t in tasks]
        cohorts = [np.array([r['profession'] in pair for r in rows]) for pair in [(5,25),(5,25),(12,24),(12,24)]]
        source = np.stack([np.load(reference/f'source__{q}__pooled.npy') for q in names]).astype(float)
        clean = np.load(reference/'none__full__pooled.npy').astype(float)
        def project(value):
            return np.stack([value[:, keep]@head for keep, head in zip(cohorts, heads)])
        den = project(source-clean)**2
        arrays = {}
        for method, rr in runs.items():
            values = []
            for run in rr:
                assert json.loads((run/'evaluation_membership.json').read_text())['rows'] == rows
                for qi, q in enumerate(names):
                    assert np.array_equal(source[qi], np.load(run/f'source__{q}__pooled.npy'))
                target = np.stack([np.load(run/f'tangent_mixed__{q}__pooled.npy') for q in names]).astype(float)
                values.append(project(target-source)**2)
            arrays[method] = np.stack(values)
        for alias in ['initial', 'input_initial', 'raw_reconstruction']:
            arrays[alias] = np.stack([project(np.stack([np.load(run/f'{alias}__{q}__pooled.npy') for q in names]).astype(float)-source)**2 for run in runs[methods[0]]])
        rng = np.random.default_rng(2026092155)
        dw = []
        for hi in [0,2]:
            rr = [r for r, keep in zip(rows, cohorts[hi]) if keep]
            counts = np.zeros((REPS,len(rr)), int)
            for profession in sorted({r['profession'] for r in rr}):
                for gender in [0,1]:
                    ii = [i for i,r in enumerate(rr) if r['profession'] == profession and r['gender'] == gender]
                    counts[:,ii] = weights(rng,len(ii))
            dw.extend([counts,counts])
    else:
        index = json.loads((reference/'INDEX.json').read_text())
        names, rows = index['queries'], index['rows']
        original = np.load(reference/'responses.npz')
        source, clean = original['source'].astype(float), original['none'].astype(float)
        den = ((source-clean)**2)[None]
        arrays = {}
        for method, rr in runs.items():
            values = []
            for run in rr:
                ind = json.loads((run/'INDEX.json').read_text())
                assert ind['rows'] == rows and ind['query_masks'] == index['query_masks']
                v = np.load(run/'responses.npz')
                assert np.array_equal(source,v['source']) and np.array_equal(clean,v['none'])
                values.append(((v['tangent_mixed'].astype(float)-source)**2)[None])
            arrays[method] = np.stack(values)
        for alias in ['native_tangent_relation_8','raw_reconstruction']:
            arrays[alias] = np.stack([((np.load(run/'responses.npz')[alias].astype(float)-source)**2)[None] for run in runs[methods[0]]])
        rng = np.random.default_rng(2026092156)
        verbs, nouns = sorted({r['verb'] for r in rows}), sorted({r['noun'] for r in rows})
        vi = np.array([verbs.index(r['verb']) for r in rows])
        ni = np.array([nouns.index(r['noun']) for r in rows])
        dw = [weights(rng,len(verbs))[:,vi]*weights(rng,len(nouns))[:,ni]]
    ff = dict(endpoints=[i for i,q in enumerate(names) if not q.startswith(('interior_','boundary_','member_subset_'))],
              participation=[i for i,q in enumerate(names) if q.startswith(('interior_','boundary_'))],
              member_subsets=[i for i,q in enumerate(names) if q.startswith('member_subset_')])
    assert all(ff.values()) and len(rows)>0
    normalizer = den.mean(-1)
    assert np.all(normalizer > 1e-12), 'Request/head source effect is insufficient for normalized analysis'
    sw = weights(rng,len(seeds))/len(seeds)
    qdraw = {f: np.tile(ix,(REPS,1)) if f=='endpoints' or (f=='member_subsets' and setting=='infinitive') else rng.choice(ix,(REPS,len(ix))) for f,ix in ff.items()}
    samples, summary, perquery = {}, {}, {}
    for method,num in arrays.items():
        pq = np.sqrt(num.mean(-1)/normalizer[None])
        perquery[method] = pq.mean((0,1)).tolist()
        boot = np.empty((len(seeds),den.shape[0],REPS,len(names)))
        for hi,wd in enumerate(dw):
            d = (wd@den[hi].T)/wd.sum(1)[:,None]
            for si in range(len(seeds)):
                n = (wd@num[si,hi].T)/wd.sum(1)[:,None]
                boot[si,hi] = np.sqrt(np.divide(n,d,out=np.full_like(n,np.nan),where=d>1e-12))
        boot = boot.mean(1)
        summary[method] = {}
        for family,ix in ff.items():
            bv = np.stack([np.take_along_axis(boot[s],qdraw[family],axis=1).mean(1) for s in range(len(seeds))],1)
            sample = (bv*sw).sum(1)
            valid = np.isfinite(sample)
            samples[method,family] = sample
            summary[method][family] = dict(nrmse=float(pq[:,:,ix].mean()), by_seed=pq[:,:,ix].mean((1,2)).tolist(),
                interval=np.quantile(sample[valid],[.025,.975]).tolist() if valid.sum() >= .9*REPS else None,
                valid_draws=int(valid.sum()),
                interval_status='available' if valid.sum() >= .9*REPS else 'insufficient source effect in resampled contexts')
    differences = []
    for method in methods[1:]:
        for ref in methods:
            if method == ref:continue
            for family in ff:
                delta = samples[method,family]-samples[ref,family]
                valid = np.isfinite(delta)
                differences.append(dict(method=method,reference=ref,family=family,
                    delta=summary[method][family]['nrmse']-summary[ref][family]['nrmse'],
                    interval=np.quantile(delta[valid],[.025,.975]).tolist() if valid.sum() >= .9*REPS else None,
                    valid_draws=int(valid.sum())))
    return dict(setting=setting,seeds=seeds,contexts=len(rows),queries=names,families=ff,
                summary=summary,differences=differences,per_query=perquery,source_rms=np.sqrt(normalizer).tolist(),
                runs={m:list(map(str,rr)) for m,rr in runs.items()},
                statistics='2000 paired target, source-context and request draws; fixed source and training recipes. Human later heads share cohort draws; grammar lexical factors resampled jointly across forms.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('setting',choices=['human','infinitive'])
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    configs = json.loads(args.input.read_text()) if args.input else {m:[str(ROOT/f'configs/fs05_{args.setting}_{m}_dev{"_v2" if args.setting=="human" else ""}.json')] for m in ['semantic','balanced','independent']}
    dest = args.output or OUT/f'{args.setting.upper()}_REQUEST_DEVELOPMENT.json'
    assert not dest.exists()
    result = analyze(args.setting,configs)
    result.update(written_at_utc=datetime.now(timezone.utc).isoformat(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  evidence='Development analysis; an input manifest does not establish independent confirmation')
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({m:{f:round(v['nrmse'],6) for f,v in fs.items()} for m,fs in result['summary'].items()},indent=2),flush=True)
    print(json.dumps(result['differences'],indent=2),flush=True)


if __name__ == '__main__':
    main()
