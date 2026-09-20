"""Paired seed/text inference for frozen fixed-support refinement."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,argparse
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BULK=Path('D:/CCAD_Storage/runs/transport_exploration_20260919')
OUT=ROOT/'artifacts/transport_exploration_20260919'
REPS=2000
VERSION=2


def multinomial(rng,n):return rng.multinomial(n,np.full(n,1/n),size=REPS)


def analyze(study):
    runs=[BULK/f'TRANSPORT01_frozen_{study}_t{s}_v{VERSION}_20260920' for s in range(1,6)]
    for r in runs:assert json.loads((r/'status.json').read_text())['status']=='PASS'
    indices=[json.loads((r/'INDEX.json').read_text()) for r in runs]
    assert all(i['rows']==indices[0]['rows'] and i['queries']==indices[0]['queries'] for i in indices)
    rows=indices[0]['rows'];values=[dict(np.load(r/'responses.npz')) for r in runs]
    clean=values[0]['clean' if study=='human' else 'none'].astype(float);source=values[0]['source'].astype(float)
    if clean.ndim==1:clean=clean[None,:]
    for v in values[1:]:
        assert np.array_equal(source,v['source'])
        c=v['clean' if study=='human' else 'none'];c=c[None,:] if c.ndim==1 else c
        assert np.array_equal(clean,c)
    denominator=(source-clean)**2;rng=np.random.default_rng(2026092001+(study=='grammar'))
    if study=='human':
        wd=np.zeros((REPS,len(rows)),np.int64)
        for y in [0,1]:
            for g in [0,1]:
                ix=[i for i,r in enumerate(rows) if r['label']==y and r['gender']==g]
                wd[:,ix]=multinomial(rng,len(ix))
        methods=['initial_active','initial_open','refined_initial_open','raw_reconstruction'];primary='refined_initial_open';ref='initial_open'
    else:
        verbs=sorted({r['verb'] for r in rows});nouns=sorted({r['noun'] for r in rows})
        vi=[verbs.index(r['verb']) for r in rows];ni=[nouns.index(r['noun']) for r in rows]
        wd=multinomial(rng,len(verbs))[:,vi]*multinomial(rng,len(nouns))[:,ni]
        methods=['native_tangent_relation_8','transport_open_initial','transport_refined_open','raw_reconstruction'];primary='transport_refined_open';ref='transport_open_initial'
    sw=multinomial(rng,5)/5;div=(wd@denominator.T)/wd.sum(1)[:,None];summaries={};samples={}
    for m in methods:
        error=np.stack([(v[m].astype(float)-source)**2 for v in values])
        perquery=np.sqrt(error.mean(-1)/np.maximum(denominator.mean(-1),1e-12))
        draws=np.stack([np.sqrt(((wd@e.T)/wd.sum(1)[:,None])/np.maximum(div,1e-12)).mean(1) for e in error],1)
        samples[m]=(draws*sw).sum(1)
        summaries[m]=dict(nrmse=float(perquery.mean()),by_seed=perquery.mean(1).tolist(),interval=np.quantile(samples[m],[.025,.975]).tolist(),by_query=perquery.mean(0).tolist())
    contrasts={m:dict(delta=summaries[primary]['nrmse']-summaries[m]['nrmse'],interval=np.quantile(samples[primary]-samples[m],[.025,.975]).tolist()) for m in methods if m!=primary}
    return dict(study=study,summary=summaries,differences=contrasts,primary=primary,reference=ref,contexts=len(rows),queries=list(indices[0]['queries']),seeds=list(range(1,6)),source_effect_rms=np.sqrt(denominator.mean(1)).tolist(),paired_source_and_contexts=True,statistics='2000 paired target-seed and stratified document (human) / crossed verb-noun (grammar) draws. Same draws across methods; exhaustive requests fixed; source explanations fixed.',runs=list(map(str,runs)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--version',type=int,default=2);args=p.parse_args();VERSION=args.version
    out=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),human=analyze('human'),grammar=analyze('grammar'),freeze_sha256=hashlib.sha256((OUT/'REFINEMENT_FREEZE.json').read_bytes()).hexdigest(),correction_freeze_sha256=hashlib.sha256((OUT/'REFINEMENT_CORRECTION_FREEZE.json').read_bytes()).hexdigest(),analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=args.output;assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:{'summary':{m:{a:b for a,b in s.items() if a!='by_query'} for m,s in v['summary'].items()},'differences':v['differences']} for k,v in out.items() if k in ['human','grammar']},indent=2))
