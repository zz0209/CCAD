"""Supplementary estimator-matched input control; original primary is unchanged."""
from pathlib import Path
import json,hashlib
import numpy as np
from analyze_arithmetic_response import answer
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/correspondence_reform_20260913'


def main():
    original=ROOT/'runs/REFORM_R45_read_write_confirmation_v1_20260915'
    control=ROOT/'runs/REFORM_R45_bank_ridge_control_eval_v1_20260915'
    for run in [original,control]:assert json.loads((run/'status.json').read_text())['status']=='PASS'
    a=json.loads((original/'panel.json').read_text());b=json.loads((control/'panel.json').read_text())
    assert {k:v for k,v in a.items() if k!='scope'}=={k:v for k,v in b.items() if k!='scope'}
    rows=[json.loads(line) for run in [original,control] for line in (run/'metrics.raw.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['kind']=='source_patch']
    lookup={(r['seed'],r['method'],r['operation'],r['row_id']):r for r in rows}
    assert len(rows)==len(lookup)
    methods=['activation_readout','bank_ridge_readout','full_activation_readout']
    nq=json.loads((original/'config.resolved.json').read_text())['pairs_per_template']
    weights=np.random.default_rng(945015).multinomial(nq,np.full(nq,1/nq),10000)/nq
    def score(v):return .5*(v[...,1]/v[...,0]+v[...,3]/v[...,2])
    def ci(v):return (100*np.quantile(v,[.025,.975])).tolist()
    stores={};scores=[];cells=[]
    for m in methods:
        counts=np.zeros((nq,4))
        for q in range(nq):
            for form in range(2):
                rid=form*nq+q;base=answer(lookup[0,'no_edit','unit',rid])
                for s in range(1,6):
                    for op in ['unit','tens']:
                        for part in [0,1]:
                            src=answer(lookup[s,f'source_part{part}_bank0',op,rid])
                            pred=answer(lookup[s%5+1,f'{m}_part{part}_bank0',op,rid])
                            changed=int(src!=base);counts[q,2*changed]+=1;counts[q,2*changed+1]+=int(src==pred)
        value=score(counts.sum(0));rep=score(weights@counts);stores[m]=(value,rep)
        scores.append(dict(method=m,balanced=float(100*value),interval=ci(rep),counts=counts.sum(0).tolist()))
        for query in ['full','part0_bank0','part1_bank0']:
            for op in ['unit','tens']:
                rr=[r for r in rows if r['method']==m+'_'+query and r['operation']==op]
                assert len(rr)==nq*2*5
                cells.append(dict(method=m,query=query,operation=op,n=len(rr),**{k:100*float(np.mean([r[v] for r in rr])) for k,v in [('H','exact_hybrid'),('T','target_digit_success'),('P','preserve_digit_success')]}))
    contrasts=[]
    for left,right in [('full_activation_readout','bank_ridge_readout'),('bank_ridge_readout','activation_readout')]:
        x,xr=stores[left];y,yr=stores[right]
        contrasts.append(dict(contrast=left+' minus '+right,points=float(100*(x-y)),interval=ci(xr-yr)))
    out=dict(scope='Supplementary post-confirmation estimator-matched input control on the exposed R45 panel; no target-output tuning. Not a replacement primary.',
             fidelity=scores,contrasts=contrasts,cells=cells,checks=dict(identical_panel=True,unique_rows=True,complete_requests=True),
             runs=[r.relative_to(ROOT).as_posix() for r in [original,control]],
             raw_sha256={r.relative_to(ROOT).as_posix():hashlib.sha256((r/'metrics.raw.jsonl').read_bytes()).hexdigest() for r in [original,control]})
    (ART/'r45_bank_ridge_control.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(fidelity=scores,contrasts=contrasts,full_functions=[r for r in cells if r['query']=='full'])))


if __name__=='__main__':main()
