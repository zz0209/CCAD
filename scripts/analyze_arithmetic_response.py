"""Frozen part-request comparisons, preserving question/form/seed dependence."""
from pathlib import Path
import json, hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/correspondence_reform_20260913'


def answer(r):
    return ('number',r['answer']) if r['answer'] is not None else ('text',r['generated_text'].strip())


def main():
    run=ROOT/'runs/REFORM_R44_arithmetic_response_confirmation_v1_20260915'
    status=json.loads((run/'status.json').read_text())
    runs=[run]
    if status['status']!='PASS':
        assert status['status']=='FAIL' and 'TimeoutError' in (run/'traceback.log').read_text()
        continuation=ROOT/'runs/REFORM_R44_arithmetic_response_confirmation_resume_v1_20260915'
        assert json.loads((continuation/'status.json').read_text())['status']=='PASS'
        runs.append(continuation)
    freeze=json.loads((ART/'R44_RESPONSE_FREEZE.json').read_text())
    snapshots={r['path']:r for r in json.loads((run/'code_hashes.json').read_text())['files']}
    for name,h in freeze['files'].items():
        p=run/snapshots[name]['snapshot_path'] if name in snapshots else ROOT/name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==h,name
    cfg=json.loads((run/'config.resolved.json').read_text());nq=cfg['pairs_per_template']
    for continuation in runs[1:]:
        cc=json.loads((continuation/'config.resolved.json').read_text())
        operational={'run_id','written_at_utc','purpose','budget','budget_seconds','resume_interventions'}
        assert {k:v for k,v in cc.items() if k not in operational}=={k:v for k,v in cfg.items() if k not in operational}
        assert cc['resume_interventions']['raw_sha256']==hashlib.sha256((run/'metrics.raw.jsonl').read_bytes()).hexdigest()
        assert json.loads((continuation/'panel.json').read_text())==json.loads((run/'panel.json').read_text())
        ch=json.loads((continuation/'code_hashes.json').read_text())['files']
        for r in ch:
            if r['path'] in freeze['files']:
                assert hashlib.sha256((continuation/r['snapshot_path']).read_bytes()).hexdigest()==freeze['files'][r['path']]
    raw=[json.loads(s) for rr in runs for s in (rr/'metrics.raw.jsonl').read_text().splitlines()]
    rows=[r for r in raw if r['kind']=='source_patch']
    lookup={(r['seed'],r['method'],r['operation'],r['row_id']):r for r in rows}
    assert len(lookup)==len(rows),'duplicate output rows'
    methods=['member','assignment','raw_readout','activation_readout','native_code','response_code']
    rng=np.random.default_rng(944015);weights=rng.multinomial(nq,np.full(nq,1/nq),10000)/nq
    def score(c):return .5*(c[...,1]/c[...,0]+c[...,3]/c[...,2])
    def ci(v):return (100*np.quantile(v,[.025,.975])).tolist()
    fidelity=[];store={};seed_scores=[]
    for dose in ['', '_norm']:
        for method in methods:
            counts=np.zeros((nq,5,4))
            for q in range(nq):
                for form in range(2):
                    rid=form*nq+q;base=answer(lookup[0,'no_edit','unit',rid])
                    for s in range(1,6):
                        t=s%5+1
                        for op in ['unit','tens']:
                            for part in [0,1]:
                                src=answer(lookup[s,f'source_part{part}_bank0'+dose,op,rid])
                                pred=answer(lookup[t,f'{method}_part{part}_bank0'+dose,op,rid])
                                change=int(src!=base)
                                counts[q,s-1,2*change]+=1
                                counts[q,s-1,2*change+1]+=int(src==pred)
            c=counts.sum(1);total=c.sum(0);v=score(total);rep=score(weights@c)
            store[method,dose]=(v,rep)
            fidelity.append(dict(method=method,dose=dose or 'original',balanced=float(100*v),interval=ci(rep),
                changed=100*total[3]/total[2],unchanged=100*total[1]/total[0],counts=total.tolist()))
            for s in range(5):seed_scores.append(dict(source=s+1,target=(s+1)%5+1,method=method,dose=dose or 'original',balanced=float(100*score(counts[:,s].sum(0)))))
    contrasts=[]
    for dose in ['', '_norm','equal_dose_mean']:
        def get(m):
            if dose!='equal_dose_mean':return store[m,dose]
            a,b=store[m,''],store[m,'_norm'];return (a[0]+b[0])/2,(a[1]+b[1])/2
        for other in methods[:-1]:
            a,ar=get('response_code');b,br=get(other)
            contrasts.append(dict(dose=dose or 'original',contrast='response_code minus '+other,points=float(100*(a-b)),interval=ci(ar-br)))
    cells=[]
    for method in ['source']+methods:
        for suffix in ['part0_bank0','part1_bank0','full','part0_bank0_norm','part1_bank0_norm']:
            for op in ['unit','tens']:
                rr=[r for r in rows if r['method']==method+'_'+suffix and r['operation']==op]
                assert len(rr)==nq*2*5,(method,suffix,op,len(rr))
                cells.append(dict(method=method,query=suffix,operation=op,n=len(rr),
                    **{k:float(100*np.mean([r[v] for r in rr])) for k,v in [('H','exact_hybrid'),('T','target_digit_success'),('P','preserve_digit_success')]}))
    diag=[d for rr in runs if (rr/'RESPONSE_EXECUTION.json').exists() for d in json.loads((rr/'RESPONSE_EXECUTION.json').read_text())['records']]
    flat=[r for d in diag for r in d['records']]
    assert all(r['min_edited_code']>=-1e-6 and r['max_changed_members']<=64 for r in flat)
    panel=json.loads((run/'panel.json').read_text())
    prior={r[key] for name in cfg['evaluation_exclusion_runs'] for r in json.loads((ROOT/name/'panel.json').read_text())['rows']
           for key in ['prompt','sentence_good','sentence_bad'] if key in r}
    assert not prior.intersection(r['prompt'] for r in panel['rows'])
    pp=panel['pairs'];assert len(pp)==2*nq
    for q in range(nq):
        for which in ['recipient','donor']:
            a=panel['rows'][pp[q][which]];b=panel['rows'][pp[q+nq][which]]
            assert (a['a'],a['b'])==(b['a'],b['b'])
    primary=next(r for r in contrasts if r['dose']=='equal_dose_mean' and r['contrast']=='response_code minus member')
    out=dict(run=run.relative_to(ROOT).as_posix(),runs=[r.relative_to(ROOT).as_posix() for r in runs],primary=primary,fidelity=fidelity,contrasts=contrasts,
        by_seed=seed_scores,cells=cells,freeze=freeze,
        response=dict(diagnostic_scope='all confirmation calls' if len(runs)==1 else 'continuation only; original wall-time failure preceded diagnostic export',calls=len(flat),solver_seconds=sum(r['solver_seconds'] for r in flat),backward_batches=sum(r['backward_batches'] for r in flat),
            extra_forward_batches=sum(r['additional_forward_batches'] for r in flat),
            initial_mean_kl=float(np.mean([v for r in flat for v in r['initial']['kl']])),
            final_mean_kl=float(np.mean([v for r in flat for v in r['final_kl']])),
            min_edited_code=min(r['min_edited_code'] for r in flat),max_members=max(r['max_changed_members'] for r in flat)),
        checks=dict(frozen_file_hashes=True,unique_complete_rows=True,paired_forms=True,previous_exact_prompt_overlap=0,nonnegative_code_in_saved_diagnostics=True),
        statistics=freeze['statistics'],raw_sha256={r.relative_to(ROOT).as_posix():hashlib.sha256((r/'metrics.raw.jsonl').read_bytes()).hexdigest() for r in runs})
    (ART/'r44_arithmetic_response_confirmation.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(primary=primary,fidelity=fidelity,contrasts=contrasts,response=out['response'])))


if __name__=='__main__':main()
