"""Frozen-query fidelity and paired source dependency contrasts for R43."""
from pathlib import Path
from collections import defaultdict
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/correspondence_reform_20260913'

def answer(r):
    return ('number',r['answer']) if r['answer'] is not None else ('text',r['generated_text'].strip())

def main():
    run=ROOT/'runs/REFORM_R43_arithmetic_part_dose_confirmation_v1_20260915'
    assert json.loads((run/'status.json').read_text())['status']=='PASS'
    freeze=json.loads((ART/'R43_PART_DOSE_FREEZE.json').read_text())
    snapshots={r['path']:r for r in json.loads((run/'code_hashes.json').read_text())['files']}
    verified={}
    for p,h in freeze['files'].items():
        f=run/snapshots[p]['snapshot_path'] if p in snapshots else ROOT/p
        assert hashlib.sha256(f.read_bytes()).hexdigest()==h,p
        verified[p]=f.relative_to(ROOT).as_posix()
    cfg=json.loads((run/'config.resolved.json').read_text());nq=cfg['pairs_per_template']
    rows=[json.loads(x) for x in (run/'metrics.raw.jsonl').read_text().splitlines()]
    lookup={(r['seed'],r['method'],r['operation'],r['row_id']):r for r in rows if r['kind']=='source_patch'}
    rng=np.random.default_rng(943015);weights=rng.multinomial(nq,np.full(nq,1/nq),10000)/nq
    def interval(v):return (100*np.quantile(weights@v,[.025,.975])).tolist()
    patterns={};original={};by_seed=[]
    for op in ['unit','tens']:
        p=np.empty((nq,2,5));o=np.empty_like(p)
        for q in range(nq):
            for form in range(2):
                rid=form*nq+q
                for s in range(1,6):
                    f=lookup[s,'source_full',op,rid]['exact_hybrid']
                    halves=[lookup[s,f'source_part{k}_bank0'+suffix,op,rid]['exact_hybrid'] for k in [0,1] for suffix in ['', '_scale4','_norm','_norm_nonnegative']]
                    p[q,form,s-1]=f and not any(halves)
                    o[q,form,s-1]=f and not any([halves[0],halves[4]])
        patterns[op]=p.mean((1,2));original[op]=o.mean((1,2))
        by_seed += [dict(operation=op,seed=s+1,original=100*o[:,:,s].mean(),dose_robust=100*p[:,:,s].mean()) for s in range(5)]
    primary=[]
    for name,v in [('units_robust',patterns['unit']),('tens_robust',patterns['tens']),('units_minus_tens',patterns['unit']-patterns['tens']),('units_reduction',original['unit']-patterns['unit']),('tens_reduction',original['tens']-patterns['tens'])]:
        primary.append(dict(contrast=name,points=float(100*v.mean()),interval=interval(v)))
    fidelity=[];contrasts=[];store={}
    for dose in ['', '_scale4','_norm']:
        for method in ['member','assignment','raw_readout','activation_readout']:
            counts=np.zeros((nq,4))
            for q in range(nq):
                for form in range(2):
                    rid=form*nq+q;base=answer(lookup[0,'no_edit','unit',rid])
                    for s in range(1,6):
                        t=s%5+1
                        for op in ['unit','tens']:
                            for part in [0,1]:
                                src=answer(lookup[s,f'source_part{part}_bank0'+dose,op,rid])
                                pred=answer(lookup[t,f'{method}_part{part}_bank0'+dose,op,rid])
                                change=src!=base;counts[q,int(change)*2]+=1;counts[q,int(change)*2+1]+=src==pred
            def score(counts):return .5*(counts[...,1]/counts[...,0]+counts[...,3]/counts[...,2])
            total=counts.sum(0);reps=score(weights@counts);v=score(total)
            store[method,dose]=(v,reps)
            fidelity.append(dict(method=method,dose=dose or 'original',balanced=float(v*100),interval=(np.quantile(reps,[.025,.975])*100).tolist(),changed=100*total[3]/total[2],unchanged=100*total[1]/total[0]))
        for other in ['assignment','raw_readout','activation_readout']:
            a,ar=store['member',dose];b,br=store[other,dose]
            contrasts.append(dict(dose=dose or 'original',contrast='member minus '+other,points=float(100*(a-b)),interval=(100*np.quantile(ar-br,[.025,.975])).tolist()))
    out=dict(primary=primary,by_seed=by_seed,fidelity=fidelity,contrasts=contrasts,freeze=freeze,verified_frozen_identities=verified,
        statistics='64 question clusters co-resampled across two fixed formats and five dependent SAE directions; source seed s aligned to target s mod5+1. Parsed integer answers compared; nonnumeric cases compare their generated text and remain in denominator.',
        input_sha256=hashlib.sha256((run/'metrics.raw.jsonl').read_bytes()).hexdigest())
    (ART/'r43_arithmetic_dose_transfer.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))
if __name__=='__main__':main()
