"""Measure source-response prediction on unfitted member subset requests."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re
import numpy as np

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/correspondence_reform_20260913'


def main():
    run=ROOT/'runs/REFORM_R38_qwen_member_queries_v1_20260914'
    resume=ROOT/'runs/REFORM_R38_qwen_member_queries_resume_v2_20260914'
    state=json.loads((run/'status.json').read_text())
    assert state['status']=='FAIL' and 'Arithmetic source development budget' in state['error']
    assert json.loads((resume/'status.json').read_text())['status']=='PASS'
    cfg=json.loads((run/'config.resolved.json').read_text());panel=json.loads((run/'panel.json').read_text())
    raw=run/'metrics.raw.jsonl';digest=hashlib.sha256(raw.read_bytes()).hexdigest()
    assert digest==json.loads((run/'metrics.summary.json').read_text())['metrics_raw_sha256']
    resumed_cfg=json.loads((resume/'config.resolved.json').read_text())
    assert resumed_cfg['resume_interventions']['raw_sha256']==digest
    assert json.loads((resume/'panel.json').read_text())==panel
    resumed_raw=resume/'metrics.raw.jsonl';resumed_digest=hashlib.sha256(resumed_raw.read_bytes()).hexdigest()
    assert resumed_digest==json.loads((resume/'metrics.summary.json').read_text())['metrics_raw_sha256']
    def key(q):return tuple(panel['rows'][i][k] for i in [q['recipient'],q['donor']] for k in ['a','b'])
    ids=sorted({key(q) for q in panel['pairs']});assert len(ids)==64
    names=['source','member','assignment','two_assignment','wrong']
    shape=(64,2,2,2,5)
    answers={n:np.full(shape,-99999,dtype=np.int64) for n in names}
    outcomes={n:np.full(shape+(3,),np.nan) for n in names}
    baseline=np.full(shape,-99999,dtype=np.int64)
    records=[json.loads(x) for p in [raw,resumed_raw] for x in p.read_text().splitlines()]
    base={}
    for r in records:
        if r['kind']=='base':
            if r['row_id'] in base:assert base[r['row_id']]==r['answer']
            base[r['row_id']]=r['answer']
    for r in records:
        if r['kind']!='source_patch' or r['seed']==0:continue
        name,part=r['method'].rsplit('_part',1)
        q=panel['pairs'][r['row_id']];rec,donor=[panel['rows'][q[k]] for k in ['recipient','donor']]
        ix=(ids.index(key(q)),int(part),['unit','tens'].index(r['operation']),q['template'],r['seed']-1)
        match=re.search(r'\d+',r['generated_text']);answer=int(match.group()) if match else -1
        expected=rec['total']//10*10+donor['total']%10 if r['operation']=='unit' else donor['total']//10*10+rec['total']%10
        numeric=10<=answer<100
        t=numeric and (answer%10==donor['total']%10 if r['operation']=='unit' else answer//10==donor['total']//10)
        p=numeric and (answer//10==rec['total']//10 if r['operation']=='unit' else answer%10==rec['total']%10)
        vals=[answer==expected,t,p]
        assert vals==[r[k] for k in ['exact_hybrid','target_digit_success','preserve_digit_success']]
        assert answers[name][ix]==-99999
        answers[name][ix]=answer;outcomes[name][ix]=vals
        b=base[q['recipient']];baseline[ix]=-1 if b is None else b
    assert all(np.isfinite(v).all() for v in outcomes.values())
    assert all((v!=-99999).all() for v in answers.values())
    # Align each target t with the source s of its incoming relation.
    order=[4,0,1,2,3]
    source=answers['source'][...,order]
    source_outcomes=outcomes['source'][...,order,:]
    changed=source!=baseline
    agreement={n:a==source for n,a in answers.items() if n!='source'}
    agreement['unchanged']=baseline==source
    # The same full member operation, repeated for either query, receives no part identity.
    parent=ROOT/cfg['member_queries']['relation_run']
    full=np.full((64,2,2,5),-99999,dtype=np.int64)
    pp=json.loads((parent/'panel.json').read_text())
    for line in (parent/'metrics.raw.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['kind']!='source_patch' or not re.match(r'transfer_s\d+_clean$',r['method']):continue
        q=pp['pairs'][r['row_id']]
        k=tuple(pp['rows'][i][v] for i in [q['recipient'],q['donor']] for v in ['a','b'])
        a=r['answer'];full[ids.index(k),['unit','tens'].index(r['operation']),q['template'],r['seed']-1]=-1 if a is None else a
    assert (full!=-99999).all()
    agreement['full_component']=full[:,None,:,:,:]==source
    def score(a,signal):
        assert signal.any() and (~signal).any()
        return .5*(a[signal].mean()+a[~signal].mean())
    cells=[]
    for n,a in agreement.items():
        cells.append(dict(method=n,exact_response_agreement=float(a.mean()),
            changed_source_agreement=float(a[changed].mean()),unchanged_source_agreement=float(a[~changed].mean()),
            balanced_response_agreement=float(score(a,changed))))
    draws=np.random.default_rng(9381549).integers(64,size=(10000,64));contrasts=[]
    for other in ['assignment','two_assignment','wrong','full_component','unchanged']:
        estimates=[]
        for ix in draws:
            signal=changed[ix]
            estimates.append(score(agreement['member'][ix],signal)-score(agreement[other][ix],signal))
        lo,hi=np.quantile(estimates,[.025,.975])*100
        contrasts.append(dict(reference='member',comparator=other,
            difference_points=100*float(score(agreement['member'],changed)-score(agreement[other],changed)),
            interval_points=[float(lo),float(hi)]))
    functional=[]
    for n,a in outcomes.items():
        for oi,op in enumerate(['unit','tens']):
            functional.append(dict(method=n,operation=op,**{m:float(a[:,:,oi,...,k].mean()) for k,m in enumerate(['H','T','P'])}))
    out=dict(written_utc=datetime.now(timezone.utc).isoformat(),scope=cfg['scope'],raw_sha256=digest,
        source_changed_fraction=float(changed.mean()),source_changed_count=int(changed.sum()),n_source_queries=int(changed.size),
        cells=cells,contrasts=contrasts,functional=functional,
        analysis='Balanced exact-continuation agreement averages changed and unchanged source responses equally. Whole question clusters are resampled, retaining both queryparts,operations,prompts and fixed dependentSAEs. Exploratory new requests on exposed data.',
        raw_run=run.relative_to(ROOT).as_posix(),resume_run=resume.relative_to(ROOT).as_posix(),resume_raw_sha256=resumed_digest,
        parsed_generated_texts=sum(v.size for v in answers.values()))
    (ART/'r38_member_queries.json').write_text(json.dumps(out,indent=2)+'\n')
    np.savez_compressed(ART/'r38_member_queries.npz',methods=np.array(list(answers)),answers=np.stack(list(answers.values())),
                        outcomes=np.stack(list(outcomes.values())),source_aligned=source,source_outcomes=source_outcomes,
                        baseline=baseline,changed=changed,operand_pairs=np.array(ids))
    print(json.dumps(dict(changed_fraction=out['source_changed_fraction'],cells=cells,contrasts=contrasts)))


if __name__=='__main__':main()
