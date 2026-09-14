"""Aggregate the positional membership comparison with paired question clusters."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/correspondence_reform_20260913'

def main():
    with np.load(ART/'r35_cluster_outcomes.npz') as z:
        oldmethods=z['methods'].tolist();ids=z['operand_pairs'].tolist()
        names=['response_direct_profile','response_direct_scalar','response_field_teacher','direct_320','source_views']
        arrays=[z['outcomes'][oldmethods.index(m)] for m in names]
    runs=['REFORM_R36_qwen_position_relation_v1_20260914','REFORM_R36_qwen_role_relation_v1_20260914']
    for run,prefix in zip(runs,['position','role']):
        p=ROOT/'runs'/run
        if not (p/'status.json').exists() or json.loads((p/'status.json').read_text())['status']!='PASS':continue
        panel=json.loads((p/'panel.json').read_text())
        groups={}
        for line in (p/'metrics.raw.jsonl').read_text().splitlines():
            r=json.loads(line)
            if r['kind']!='source_patch' or not r['method'].startswith('position_s'):continue
            m=prefix+'_'+r['method'].split('_',2)[2]
            if m not in groups:groups[m]=np.full((64,2,2,5,3),np.nan)
            pair=panel['pairs'][r['row_id']]
            cl=[panel['rows'][i][k] for i in [pair['recipient'],pair['donor']] for k in ['a','b']]
            ix=(ids.index(cl),['unit','tens'].index(r['operation']),pair['template'],r['seed']-1)
            assert np.isnan(groups[m][ix]).all()
            groups[m][ix]=[r[k] for k in ['exact_hybrid','target_digit_success','preserve_digit_success']]
        for m,a in groups.items():
            assert np.isfinite(a).all();names.append(m);arrays.append(a)
    data=np.stack(arrays);rng=np.random.default_rng(9360914);draws=rng.integers(64,size=(10000,64))
    comps=[('position_field','response_field_teacher'),('position_direct','response_direct_profile'),('position_direct','position_scalar'),('role_direct','position_direct'),('role_direct','role_scalar'),('role_direct','role_swapped')]
    contrasts=[]
    for a,b in comps:
        if a not in names or b not in names:continue
        delta=(data[names.index(a)]-data[names.index(b)]).mean((1,2,3));ci=np.quantile(delta[draws].mean(1),[.025,.975],axis=0)*100
        contrasts.append(dict(reference=a,comparator=b,metrics={m:dict(difference_points=float(delta[:,k].mean()*100),interval_points=ci[:,k].tolist()) for k,m in enumerate(['H','T','P'])}))
    cells=[]
    for m,a in zip(names,data):
        for oi,op in enumerate(['unit','tens']):
            cells.append(dict(method=m,operation=op,n=640,**{metric:float(a[:,oi,...,k].mean()) for k,metric in enumerate(['H','T','P'])},per_seed=a[:,oi,:,:,0].mean((0,1)).tolist()))
    out=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),scope='Adaptive development on the exposed original panel; question-cluster intervals condition on the five fixed SAEs.',cells=cells,contrasts=contrasts)
    (ART/'r36_development.json').write_text(json.dumps(out,indent=2)+'\n')
    np.savez_compressed(ART/'r36_development_outcomes.npz',outcomes=data,methods=np.array(names),operand_pairs=np.array(ids))
    print(json.dumps(dict(means={m:float(a[...,0].mean()) for m,a in zip(names,data)},contrasts=contrasts)))
if __name__=='__main__':main()
