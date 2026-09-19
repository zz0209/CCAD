"""Paired inference on frozen program confirmation, with shared seed/text draws."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/science_upgrade_20260919'
BULK=Path('D:/CCAD_Storage/runs/science_upgrade_20260919')
REPS=2000

def families(index,names):
    ff={f:[i for i,q in enumerate(names) if index[q]==f] for f in ['endpoints','interior','boundary','member_subsets']}
    ff['held_requests']=ff['interior']+ff['boundary']
    return ff

def weights(rng,n):return rng.multinomial(n,[1/n]*n,size=REPS)

def summarize(methods,nums,den,docweights,ff,rng,source_effect,seeds,extra):
    # Shapes S,H,Q,N and H,Q,N. Each head is evaluated on its own paired cohort.
    sw=weights(rng,len(seeds))/len(seeds)
    qdraw={f:np.tile(ix,(REPS,1)) if f=='endpoints' or (f=='member_subsets' and extra['setting']=='infinitive') else rng.choice(ix,(REPS,len(ix))) for f,ix in ff.items() if f!='held_requests'}
    qdraw['held_requests']=np.concatenate([qdraw['interior'],qdraw['boundary']],axis=1)
    summary={};samples={};perquery={}
    for method in methods:
        num=nums[method]
        pq=np.sqrt(num.mean(-1)/np.maximum(den.mean(-1),1e-12)[None])
        perquery[method]=pq.mean((0,1)).tolist()
        b=np.empty((len(seeds),den.shape[0],REPS,den.shape[1]))
        for h in range(den.shape[0]):
            wd=docweights[h]
            divisor=np.maximum((wd@den[h].T)/wd.sum(1)[:,None],1e-12)
            for s in range(len(seeds)):
                b[s,h]=np.sqrt(((wd@num[s,h].T)/wd.sum(1)[:,None])/divisor)
        b=b.mean(1)
        summary[method]={}
        for f,ix in ff.items():
            bv=np.stack([np.take_along_axis(b[s],qdraw[f],axis=1).mean(1) for s in range(len(seeds))],axis=1)
            samples[method,f]=(bv*sw).sum(1)
            summary[method][f]=dict(nrmse=float(pq[:,:,ix].mean()),interval=np.quantile(samples[method,f],[.025,.975]).tolist(),by_seed=pq[:,:,ix].mean((1,2)).tolist(),by_head=pq[:,:,ix].mean((0,2)).tolist())
    contrasts=[]
    for m in ['tangent_gain','tangent_mixed']:
        for ref in methods:
            if m==ref:continue
            for f in ff:
                contrasts.append(dict(method=m,reference=ref,family=f,delta=summary[m][f]['nrmse']-summary[ref][f]['nrmse'],interval=np.quantile(samples[m,f]-samples[ref,f],[.025,.975]).tolist()))
    return dict(written_at_utc=datetime.now(timezone.utc).isoformat(),summary=summary,differences=contrasts,per_query=perquery,source_effect_rms=source_effect,seeds=seeds,**extra)

def human():
    seeds=[2,3,4,5];runs=[BULK/f'SCIENCE04_shift_t{s}_v1_20260919' for s in seeds]
    for r in runs:assert json.loads((r/'status.json').read_text())['status']=='PASS',r
    req=json.loads((OUT/'ROUND04_REQUESTS.json').read_text());names=req['queries'];ff=families(req['families'],names)
    memberships=[json.loads((r/'evaluation_membership.json').read_text()) for r in runs]
    assert all(x==memberships[0] for x in memberships)
    training=[json.loads((r/'program_context_membership.json').read_text()) for r in runs]
    assert all(x==training[0] for x in training)
    old=json.loads((BULK/'SCIENCE03_shift_execution_reform_v1_20260919/program_context_membership.json').read_text())
    assert training[0]==old
    rows=memberships[0]['rows'];tasks=['composer_surgeon_orientation0','composer_surgeon_orientation1','model_software_engineer_orientation0','model_software_engineer_orientation1']
    w=np.stack([np.load(ROOT/'runs/IR04_shift_consumer_seed2_v1_20260916'/f'none__full__{t}__probe42.npz')['weight'].ravel() for t in tasks],1)
    bias=np.array([np.load(ROOT/'runs/IR04_shift_consumer_seed2_v1_20260916'/f'none__full__{t}__probe42.npz')['bias'].item() for t in tasks])
    clean=np.load(runs[0]/'none__full__pooled.npy').astype(float)
    source=np.stack([np.load(runs[0]/f'source__{q}__pooled.npy').astype(float) for q in names])
    for r in runs[1:]:
        assert np.array_equal(clean,np.load(r/'none__full__pooled.npy'))
        for qi,q in enumerate(names):assert np.array_equal(source[qi],np.load(r/f'source__{q}__pooled.npy'))
    methods=['initial','head_mixed','input_initial','tangent_gain','tangent_mixed','raw_reconstruction']
    ix=[np.array([r['profession'] in profs for r in rows]) for profs in [(5,25),(5,25),(12,24),(12,24)]]
    den=np.stack([((source[:,ix[h]]-clean[ix[h]])@w[:,h])**2 for h in range(4)])
    rng=np.random.default_rng(2026091941);dw=[]
    for h in [0,2]:
        rr=[r for r,k in zip(rows,ix[h]) if k];cnt=np.zeros((REPS,len(rr)),int)
        for p in sorted({r['profession'] for r in rr}):
            for g in [0,1]:
                ii=[i for i,r in enumerate(rr) if r['profession']==p and r['gender']==g]
                cnt[:,ii]=weights(rng,len(ii))
        dw.extend([cnt,cnt])
    nums={};decision={}
    for m in methods:
        ns=[];ds=[]
        for run in runs:
            target=np.stack([np.load(run/f'{m}__{q}__pooled.npy').astype(float) for q in names])
            ns.append(np.stack([((target[:,ix[h]]-source[:,ix[h]])@w[:,h])**2 for h in range(4)]))
            perhead=[]
            for h in range(4):
                lp=target[:,ix[h]]@w[:,h]+bias[h];slp=source[:,ix[h]]@w[:,h]+bias[h]
                labels=np.array([r['profession']==(25 if h<2 else 24) for r,k in zip(rows,ix[h]) if k])
                perhead.append(dict(source_agreement=float(((lp>0)==(slp>0)).mean()),task_accuracy=float(((lp>0)==labels).mean())))
            ds.append(perhead)
        nums[m]=np.stack(ns);decision[m]=ds
    result=summarize(methods,nums,den,dw,ff,rng,np.sqrt(den.mean(-1)).tolist(),seeds,dict(setting='human',contexts=len(rows),heads=tasks,queries=names,statistics='2000 paired target-seed, profession/gender document and request draws; same cohort draws across shared heads; fixed source and later heads; endpoints fixed, finer random masks resampled.',decision_descriptive=decision,runs=list(map(str,runs))))
    return result

def infinitive():
    seeds=[1,3,4,5];runs=[BULK/f'SCIENCE04_infinitive_t{s}_v{2 if s==1 else 1}_20260919' for s in seeds]
    for r in runs:assert json.loads((r/'status.json').read_text())['status']=='PASS',r
    indices=[json.loads((r/'INDEX.json').read_text()) for r in runs];idx=indices[0]
    assert all(x['rows']==idx['rows'] and x['query_masks']==idx['query_masks'] for x in indices)
    rows=idx['rows'];names=idx['queries'];panel=json.loads((OUT/'ROUND04_INFINITIVE_PANEL.json').read_text());ff=families(panel['families'],names)
    vs=[dict(np.load(r/'responses.npz')) for r in runs];source=vs[0]['source'].astype(float);clean=vs[0]['none'].astype(float)
    for v in vs[1:]:assert np.array_equal(source,v['source']) and np.array_equal(clean,v['none'])
    methods=['native','mixed','native_tangent_relation_8','tangent_gain','tangent_mixed','raw_reconstruction']
    nums={m:np.stack([((v[m].astype(float)-source)**2)[None] for v in vs]) for m in methods}
    den=((source-clean)**2)[None]
    verbs=sorted({r['verb'] for r in rows});nouns=sorted({r['noun'] for r in rows})
    vi=np.array([verbs.index(r['verb']) for r in rows]);ni=np.array([nouns.index(r['noun']) for r in rows])
    rng=np.random.default_rng(2026091942);dw=[weights(rng,len(verbs))[:,vi]*weights(rng,len(nouns))[:,ni]]
    effects={}
    for m in ['source']+methods:
        effects[m]={role:{q:float(np.mean([np.mean(clean[qi,sel]-v[m][qi,sel]) for v in vs])) for qi,q in enumerate(names[:3])} for role in ['predicate','object'] for sel in [np.array([r['role']==role for r in rows])]}
    return summarize(methods,nums,den,dw,ff,rng,np.sqrt(den.mean(-1)).tolist(),seeds,dict(setting='infinitive',contexts=len(rows),verbs=verbs,nouns=nouns,queries=names,statistics='2000 paired target-seed and crossed verb/noun draws; paired roles/forms; interior/boundary coordinates resampled separately; exhaustive12 finer masks and3 endpoints fixed; source fixed.',functional_effects=effects,runs=list(map(str,runs))))

def main():
    global BULK
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('setting',choices=['human','infinitive'])
    p.add_argument('--bulk-root',type=Path,default=BULK,help='Retained science run directory, or runs/science_upgrade_20260919 in the extracted companion.')
    p.add_argument('--output',type=Path,help='New analysis file; retained results are never overwritten.')
    args=p.parse_args();BULK=args.bulk_root.resolve()
    dest=args.output or OUT/f'ROUND04_{args.setting.upper()}_ANALYSIS.json'
    if dest.exists():raise FileExistsError(dest)
    dest.parent.mkdir(parents=True,exist_ok=True)
    result=human() if args.setting=='human' else infinitive()
    result['freeze_sha256']=hashlib.sha256((OUT/'ROUND04_FREEZE.json').read_bytes()).hexdigest()
    result['analysis_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({m:{f:round(v['nrmse'],5) for f,v in a.items()} for m,a in result['summary'].items()},indent=2))
    print([d for d in result['differences'] if d['method']=='tangent_mixed' and d['family']=='held_requests'])

if __name__=='__main__':main()
