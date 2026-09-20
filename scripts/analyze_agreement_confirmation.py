from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np

from analyze_profile_confirmation import query_order


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--crossed-nouns',action='store_true')
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    root=Path('artifacts/final_science_20260920_round02')
    bulk=Path('D:/CCAD_Storage/runs/final_science_20260920_round02')
    runs=[bulk/f'AGREEMENT_CONFIRM_T{s}_20260920' for s in range(1,6)]
    arrays=[]
    sources=[]
    for run in runs:
        assert json.loads((run/'status.json').read_text())['status']=='PASS'
        names=query_order(run,'grammar')
        if arrays: assert names==queries
        queries=names
        arrays.append(dict(np.load(run/'responses.npz')))
        sources.append(dict(path=str(run),predictions_sha256=hashlib.sha256((run/'responses.npz').read_bytes()).hexdigest()))
    src=arrays[0]['grammar__source'].astype(float)
    clean=arrays[0]['grammar__none'].astype(float)
    for a in arrays[1:]:
        assert np.array_equal(src,a['grammar__source'])
        assert np.array_equal(clean,a['grammar__none'])
    panel=json.loads((root/'AGREEMENT_FROZEN_PANEL.json').read_text())
    rows=panel['rows']
    membership=json.loads((runs[0]/'membership.json').read_text())
    assert [r['document_sha256'] for r in rows]==[r['document_sha256'] for r in membership['grammar_rows']]
    den=(src-clean)**2
    nums={m.removeprefix('grammar__'):np.stack([(a[m].astype(float)-src)**2 for a in arrays])
          for m in arrays[0] if m not in ['grammar__source','grammar__none']}
    assert (den.mean(-1)>0).all()
    families=dict(participation=[i for i,q in enumerate(queries) if q.startswith('participation_')],
                  endpoints=[queries.index(q) for q in ['singular','plural','full']],
                  members=[i for i,q in enumerate(queries) if q.startswith('member_')])
    rng=np.random.default_rng(2026092022)
    reps=2000
    clusters=sorted({r['subject_pair'] for r in rows})
    cluster_index=np.array([clusters.index(r['subject_pair']) for r in rows])
    cluster_draws=rng.multinomial(len(clusters),np.full(len(clusters),1/len(clusters)),size=reps)
    weights=cluster_draws[:,cluster_index]
    noun_levels=None
    if args.crossed_nouns:
        canonical={}
        for structure in ['simple','within_rc','rc']:
            original=Path('artifacts/morning_reform_20260916/independent_circuit_reading/data')/f'{structure}_train.json'
            for line in original.read_text().splitlines():
                item=json.loads(line)
                index=4 if structure=='within_rc' else 1
                words=[item[key].split()[index] for key in ['clean_prefix','patch_prefix']]
                identity='|'.join(sorted(words))
                for word in words:
                    if word in canonical: assert canonical[word]==identity
                    canonical[word]=identity
        noun_levels=sorted(set(canonical.values()))
        first=[noun_levels.index(canonical[r['clean_prefix'].split()[1]]) for r in rows]
        second=[noun_levels.index(canonical[r['clean_prefix'].split()[4]]) if r['structure']!='simple' else None for r in rows]
        draws_first=rng.multinomial(len(noun_levels),np.full(len(noun_levels),1/len(noun_levels)),size=reps)
        draws_second=rng.multinomial(len(noun_levels),np.full(len(noun_levels),1/len(noun_levels)),size=reps)
        weights=draws_first[:,first]
        for i,index in enumerate(second):
            if index is not None: weights[:,i]*=draws_second[:,index]
    seed_weights=rng.multinomial(5,np.full(5,.2),size=reps)/5
    query_draws={f:np.tile(ix,(reps,1)) if f=='endpoints' else rng.choice(ix,(reps,len(ix))) for f,ix in families.items()}
    summary={}
    samples={}
    for method,num in nums.items():
        per=np.sqrt(num.mean(-1)/den.mean(-1))
        summary[method]={}
        for family,ix in families.items():
            denominator=weights@den.T
            boot=[]
            for seed in range(5):
                ratio=np.full_like(denominator,np.nan)
                np.divide(weights@num[seed].T,denominator,out=ratio,where=denominator>0)
                values=np.sqrt(ratio)
                boot.append(np.take_along_axis(values,query_draws[family],axis=1).mean(1))
            sample=(np.stack(boot,axis=1)*seed_weights).sum(1)
            valid=np.isfinite(sample)
            samples[method,family]=sample
            summary[method][family]=dict(nrmse=float(per[:,ix].mean()),by_seed=per[:,ix].mean(1).tolist(),
                interval=np.quantile(sample[valid],[.025,.975]).tolist(),valid_bootstrap_draws=int(valid.sum()),
                undefined_draws=int((~valid).sum()))
    profile='initial_refined_source_metric'
    contrasts=[]
    for method in nums:
        if method==profile: continue
        for family in families:
            diff=samples[method,family]-samples[profile,family]
            valid=np.isfinite(diff)
            contrasts.append(dict(reference=method,family=family,
                reduction=summary[method][family]['nrmse']-summary[profile][family]['nrmse'],
                interval=np.quantile(diff[valid],[.025,.975]).tolist(),valid_bootstrap_draws=int(valid.sum())))
    structures={}
    source_effects={}
    for structure in ['simple','within_rc','rc']:
        keep=np.array([r['structure']==structure for r in rows])
        structures[structure]={}
        for method,num in nums.items():
            per=np.sqrt(num[...,keep].mean(-1)/den[:,keep].mean(-1))
            structures[structure][method]={f:float(per[:,ix].mean()) for f,ix in families.items()}
        source_effects[structure]={queries[i]:dict(rms=float(np.sqrt(den[i,keep].mean())),
            mean_effect=float((src-clean)[i,keep].mean()),clean_accuracy=float((clean[i,keep]>0).mean()),
            source_accuracy=float((src[i,keep]>0).mean())) for i in families['endpoints']}
    decisions={method:dict(source_answer_agreement=float(np.mean([(a['grammar__'+method][families['participation']]>0)==(src[families['participation']]>0) for a in arrays]))) for method in nums}
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),runs=sources,queries=queries,
        families=families,summary=summary,contrasts=contrasts,structures=structures,
        source_effects=source_effects,decisions=decisions,contexts=len(rows),targets=5,
        cluster_count=len(clusters),bootstrap_reps=reps,
        inference='Paired target-seed, crossed canonical first/embedded noun and request resampling; fixed source explanation and syntactic forms' if args.crossed_nouns else 'Paired target-seed, leading-noun-pair cluster and request resampling; fixed source explanation and syntactic forms',
        analysis_status='Post-confirmation lexical-dependence sensitivity; original frozen primary analysis retained' if args.crossed_nouns else 'Frozen primary analysis',
        crossed_noun_levels=noun_levels,
        freeze_sha256=hashlib.sha256((root/'AGREEMENT_CONFIRMATION_FREEZE.json').read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(primary={m:v['participation'] for m,v in summary.items()},
                         contrasts=[r for r in contrasts if r['family']=='participation']),indent=2))


if __name__=='__main__':
    main()
