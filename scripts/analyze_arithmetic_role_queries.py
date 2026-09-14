"""Measure the effects of source-defined role parts and their fixed translations."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/correspondence_reform_20260913'


def main():
    primary=ART/'r40_member_confirmation.npz'
    with np.load(primary) as z:
        answers=dict(zip(z['methods'].tolist(),z['answers']))
        answers['source']=z['source_aligned']
        ids=z['operand_pairs'];baseline=z['baseline']
        full=dict(zip(z['full_methods'].tolist(),z['full_function']))
        full_answers=dict(zip(z['full_methods'].tolist(),z['full_answers']))
    cfg=json.loads((ROOT/'runs/REFORM_R38_qwen_member_fields_five_v1_20260914/config.resolved.json').read_text())
    pairs=cfg['relation_transfer']['seed_pairs']
    order=[next(s-1 for s,t in pairs if t==target) for target in range(1,6)]
    full['source']=full['source'][:,:,:,order,:]
    full_answers['source']=full_answers['source'][:,:,:,order]
    extra=ART/'r40_query_readouts.npz'
    if extra.exists():
        with np.load(extra) as z:
            assert np.array_equal(z['source_aligned'],answers['source'])
            answers.update(dict(zip(z['methods'].tolist(),z['answers'])))
            full.update(dict(zip(z['methods'].tolist(),z['full_function'])))
    recipient=ids[:,:2].sum(1)[:,None,None,None,None,None]
    donor=ids[:,2:].sum(1)[:,None,None,None,None,None]
    unit=np.arange(2)[None,None,None,:,None,None]==0
    desired=np.where(unit,recipient//10*10+donor%10,donor//10*10+recipient%10)
    metrics={}
    for name,a in answers.items():
        valid=(a>=10)&(a<100)
        metrics[name]=np.stack([a==desired,
            valid&np.where(unit,a%10==donor%10,a//10==donor//10),
            valid&np.where(unit,a//10==recipient//10,a%10==recipient%10),
            a!=baseline[:,None,None,None,:,None]],-1)
    freeze=json.loads((ART/'R40_ROLE_QUERY_FREEZE_v2.json').read_text())
    rng=np.random.default_rng(freeze['analysis']['bootstrap_seed'])
    nq=len(ids);qw=rng.multinomial(nq,np.full(nq,1/nq),size=freeze['analysis']['draws'])
    profiles=[];contrasts=[]
    for name,a in metrics.items():
        # Keep all questions, the fixed bank, both prompt formats and all dependent SAEs.
        for part in [0,1]:
            for oi,op in enumerate(['unit','tens']):
                qvals=a[:,:,part,oi].mean((1,2,3))
                for ki,key in enumerate(['H','T','P','changed']):
                    vals=qvals[:,ki];ci=100*np.quantile(qw@vals/nq,[.025,.975])
                    profiles.append(dict(method=name,part=part,operation=op,metric=key,
                        mean=float(vals.mean()),interval_points=ci.tolist()))
        for oi,op in enumerate(['unit','tens']):
            d=(a[:,:,0,oi].astype(float)-a[:,:,1,oi].astype(float)).mean((1,2,3))
            for ki,key in enumerate(['H','T','P','changed']):
                vals=d[:,ki];ci=100*np.quantile(qw@vals/nq,[.025,.975])
                contrasts.append(dict(method=name,operation=op,metric=key,contrast='part0 minus part1',
                    difference_points=float(100*vals.mean()),interval_points=ci.tolist()))
    # Enumerate all example outcomes; display selection is descriptive and recorded separately.
    examples=[]
    for qi,operand in enumerate(ids):
        for oi,op in enumerate(['unit','tens']):
            for fmt in range(2):
                for seed in range(5):
                    examples.append(dict(question=qi,operands=operand.tolist(),operation=op,
                        template=fmt,target_seed=seed+1,
                        answers={n:a[qi,0,:,oi,fmt,seed].tolist() for n,a in answers.items()},
                        full_answers={n:int(a[qi,oi,fmt,seed]) for n,a in full_answers.items()}))
    # Descriptive decomposition introduced after the frozen primary profile analysis.
    # These categories are response patterns, not a claim of additive causality.
    patterns={};pattern_rows=[]
    names=['both_parts_required','part0_sufficient','part1_sufficient','either_part_sufficient',
           'half_succeeds_full_fails','none_succeeds']
    for method,f in full.items():
        a=metrics[method];h0=a[:,0,0,...,0];h1=a[:,0,1,...,0];hf=f[...,0].astype(bool)
        v=np.stack([hf&~h0&~h1,hf&h0&~h1,hf&~h0&h1,hf&h0&h1,
                    ~hf&(h0|h1),~hf&~h0&~h1],-1)
        assert (v.sum(-1)==1).all();patterns[method]=v.argmax(-1)
        for oi,op in enumerate(['unit','tens']):
            vals=v[:,oi].mean((1,2));ci=100*np.quantile(qw@vals/nq,[.025,.975],axis=0)
            for k,name in enumerate(names):
                pattern_rows.append(dict(method=method,operation=op,pattern=name,
                    fraction=float(vals[:,k].mean()),interval_points=ci[:,k].tolist()))
    pattern_agreement=[]
    for method,a in patterns.items():
        if method=='source':continue
        for oi,op in enumerate(['unit','tens']):
            vals=(a[:,oi]==patterns['source'][:,oi]).mean((1,2))
            pattern_agreement.append(dict(method=method,operation=op,agreement=float(vals.mean()),
                interval_points=(100*np.quantile(qw@vals/nq,[.025,.975])).tolist()))
    out=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),profiles=profiles,
        part_contrasts=contrasts,questions=nq,scope=freeze['scope'],statistics=freeze['analysis'],
        descriptive_coalition_patterns=pattern_rows,pattern_agreement=pattern_agreement,
        coalition_scope='Descriptive response-pattern decomposition after observing primary profiles; all six patterns, both operations and all available full methods retained.',
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [primary,extra] if p.exists()})
    (ART/'r40_role_profiles.json').write_text(json.dumps(out,indent=2)+'\n')
    (ART/'r40_role_examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    np.savez_compressed(ART/'r40_role_profiles.npz',methods=np.array(list(metrics)),
        metrics=np.stack(list(metrics.values())),operand_pairs=ids)
    print(json.dumps(dict(source_contrasts=[r for r in contrasts if r['method']=='source'],
                         source_profiles=[r for r in profiles if r['method']=='source'])))


if __name__=='__main__':
    main()
