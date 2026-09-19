"""Paired functional-error analysis of fixed public-program adaptation."""
from pathlib import Path
from datetime import datetime, timezone
import argparse,json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/science_upgrade_20260919'
RUN=Path('D:/CCAD_Storage/runs/science_upgrade_20260919/SCIENCE03_infinitive_program_replay_v3_20260919')

def main():
    p=argparse.ArgumentParser();p.add_argument('--extra-run',type=Path,action='append',default=[]);args=p.parse_args()
    assert json.loads((RUN/'status.json').read_text())['status']=='PASS'
    index=json.loads((RUN/'INDEX.json').read_text());rows=index['rows'];queries=index['queries']
    arrays=np.load(RUN/'responses.npz');values={k:arrays[k].astype('float64') for k in arrays.files}
    quality=json.loads((RUN/'quality.json').read_text());runs=[str(RUN)]
    for extra_run in args.extra_run:
        assert json.loads((extra_run/'status.json').read_text())['status']=='PASS'
        other=json.loads((extra_run/'INDEX.json').read_text())
        assert other['queries']==queries and other['rows']==rows
        extra=np.load(extra_run/'responses.npz')
        for m in extra.files:
            if m in values: assert np.max(np.abs(values[m]-extra[m]))<1e-5
            else: values[m]=extra[m].astype('float64')
        quality+=json.loads((extra_run/'quality.json').read_text());runs.append(str(extra_run))
    source=values['source'];clean=values['none'];den=(source-clean)**2
    families={'endpoints':np.arange(3),'interior':np.arange(3,15),'boundary':np.arange(15,27),'held_requests':np.arange(3,27)}
    methods=[m for m in values if m not in ['none','source']]
    per_query={m:np.sqrt(((values[m]-source)**2).mean(1)/den.mean(1)) for m in methods}
    verbs=sorted({r['pair'].split(':')[0] for r in rows});nouns=sorted({r['pair'].split(':')[1] for r in rows})
    vi=np.array([verbs.index(r['pair'].split(':')[0]) for r in rows]);ni=np.array([nouns.index(r['pair'].split(':')[1]) for r in rows])
    rng=np.random.default_rng(2026091908);reps=2000
    vc=rng.multinomial(len(verbs),[1/len(verbs)]*len(verbs),size=reps)
    nc=rng.multinomial(len(nouns),[1/len(nouns)]*len(nouns),size=reps)
    weights=vc[:,vi]*nc[:,ni];assert (weights.sum(1)>0).all()
    draws={'endpoints':np.tile(np.arange(3),(reps,1)),
           'interior':rng.integers(3,15,(reps,12)), 'boundary':rng.integers(15,27,(reps,12))}
    draws['held_requests']=np.concatenate([draws['interior'],draws['boundary']],axis=1)
    bootstrap={};summary={}
    for m in methods:
        samples=np.sqrt((weights@((values[m]-source)**2).T)/(weights@den.T))
        bootstrap[m]={family:np.take_along_axis(samples,qq,axis=1).mean(1) for family,qq in draws.items()}
        summary[m]={family:dict(nrmse=float(per_query[m][ii].mean()),interval=np.quantile(bootstrap[m][family],[.025,.975]).tolist()) for family,ii in families.items()}
    differences=[]
    for method in ['endpoints','continuous','mixed','mixed_relation','natural_only','tangent_mixed','tangent_natural','tangent_gain']:
        if method not in methods:continue
        for reference in ['native','native_tangent_relation_8','raw_reconstruction','endpoints','continuous','mixed','mixed_relation','natural_only','tangent_natural','tangent_gain','raw_reconstruction_after_tangent_mixed']:
            if reference not in methods or method==reference:continue
            for family in families:
                differences.append(dict(method=method,reference=reference,family=family,
                    delta=summary[method][family]['nrmse']-summary[reference][family]['nrmse'],
                    interval=np.quantile(bootstrap[method][family]-bootstrap[reference][family],[.025,.975]).tolist()))
    contrasts={}
    # The sign tells which named source part suppresses infinitival 'to' more.
    sc=source[1]-source[0]
    for m in ['source']+methods:
        mc=values[m][1]-values[m][0]
        contrasts[m]=dict(source_order_agreement=float(((mc>0)==(sc>0)).mean()),
            contrast_rmse=float(np.sqrt(np.mean((mc-sc)**2))),
            effect_by_role={role:{queries[q]:float((clean[q,sel]-values[m][q,sel]).mean()) for q in range(3)}
                for role in ['predicate','object'] for sel in [np.array([r['role']==role for r in rows])]})
    # Verify the new output-head API reproduces the earlier frozen executor.
    oldrun=ROOT/'runs/IR01_infinitive_confirm_t2_v1_20260916'
    oi=json.loads((oldrun/'INDEX.json').read_text());ov=np.load(oldrun/'responses.npz')['log_probability']
    checks={}
    for m in ['none','source','native','geometry','geometry_gain','native_tangent_relation_8','native_response_relation_8','raw_reconstruction']:
        diffs=[]
        for qi,name in enumerate(queries[:3]):
            oq=0 if m=='none' else oi['queries'].index(name)
            old=ov[oi['methods'].index(m),oq]
            assert [r['text'] for r in oi['rows']]==[r['text'] for r in rows]
            diffs.append(float(np.max(np.abs(values[m][qi]-old))))
        checks[m]=max(diffs)
    # Record the legacy full-token versus current last-token projection
    # discrepancy; all current methods use the same current projection.
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),runs=runs,
        evidence='Adaptive development. Existing128contexts;64 distinctfitcontexts; one target seed2 and one fixedsource explanation.',
        primary='Mean per-query response nRMSE across24 untrained request coordinates. Three deletion endpoints separately reported.',
        resampling='2000 paired crossed verb/noun pigeonhole resamples; allroles/formspaired; queriesresampledwithininterior/boundary; fixed3endpoints. Intervalsconditionalononesource/target andexploratorycomparisonswithoutmultiplicitycorrection.',
        source_effect_rms_per_query=np.sqrt(den.mean(1)).tolist(),summary=summary,differences=differences,
        functional_parts=contrasts,quality=quality,baseline_replay_max_abs=checks,
        per_query={m:v.tolist() for m,v in per_query.items()},queries=queries,
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [RUN/'responses.npz',RUN/'INDEX.json']})
    name='ROUND03_ANALYSIS_FINAL.json' if len(args.extra_run)>1 else 'ROUND03_ANALYSIS_EXTENDED.json' if args.extra_run else 'ROUND03_ANALYSIS.json'
    dest=OUT/name;assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n')
    for m,s in summary.items():print(m,{f:round(v['nrmse'],5) for f,v in s.items()})
    print('Baseline replay',checks)
    print('Mixed comparisons',[d for d in differences if d['method']=='mixed' and d['family']=='held_requests'])

if __name__=='__main__':main()
