"""Summarize whole binding states and requested/protected entity answers."""
from pathlib import Path
import argparse,json,hashlib
from datetime import datetime,timezone
import numpy as np


def analyze(run,output):
    status=json.loads((run/'status.json').read_text());assert status['status']=='PASS'
    raw=run/'metrics.raw.jsonl';rows=[json.loads(x) for x in raw.read_text().splitlines()]
    groups={};base={}
    for r in rows:
        if r['kind']=='base':base[r['row_id']]=r
        if r['kind']=='intervention':
            key=(r['mode'],r['method'],r.get('seed'),r.get('target_seed'))
            groups.setdefault(key,[]).append(r)
    cells=[];arrays={};keys=[];contexts=None
    for key,rr in groups.items():
        cs=sorted({r['component'] for r in rr});forms=sorted({r['task'] for r in rr})
        ops=sorted({r['operation'] for r in rr})
        if contexts is None:contexts=cs
        assert contexts==cs
        a=np.full((len(cs),len(forms),len(ops),2,2),np.nan)
        for r in rr:
            a[cs.index(r['component']),forms.index(r['task']),ops.index(r['operation']),r['query']]=[r['correct'],r['four_choice_correct']]
        assert np.isfinite(a).all();arrays['|'.join(map(str,key))]=a;keys.append(key)
        for j,op in enumerate(ops):
            success=a[:,:,j,:,0];four=a[:,:,j,:,1]
            cells.append(dict(layer=key[0],method=key[1],seed=key[2],target_seed=key[3],operation=op,
                next_token_accuracy=float(success.mean()),complete_binding_accuracy=float(success.prod(-1).mean()),
                four_choice_accuracy=float(four.mean()),contexts=len(cs),prompt_forms=len(forms),
                requested_accuracy=float(success.mean() if op=='both' else success[...,0 if op=='first' else 1].mean()),
                preserved_accuracy=None if op=='both' else float(success[...,1 if op=='first' else 0].mean())))
    rng=np.random.default_rng(9412219);n=len(contexts);weights=rng.multinomial(n,np.full(n,1/n),size=10000)
    contrasts=[]
    for key,a in arrays.items():
        parts=key.split('|')
        for ref in ['sae64','sae256','raw']:
            comp='|'.join([parts[0],ref,'0' if ref=='raw' else parts[2],parts[3]])
            if comp not in arrays or comp==key:continue
            b=arrays[comp]
            for oi,op in enumerate(ops):
                d=(a[:,:,oi,:,0].prod(-1)-b[:,:,oi,:,0].prod(-1)).mean(1)
                contrasts.append(dict(method=key,reference=comp,operation=op,
                    complete_difference_points=float(d.mean()*100),interval_points=(100*np.quantile(weights@d/n,[.025,.975])).tolist()))
    # The five cyclic pairs share dictionaries. Resample contexts, retaining
    # the entire fixed cohort, rather than treating directions as independent.
    cohorts={};macro=[];macro_contrasts=[]
    for key in keys:
        cohorts.setdefault((key[0],key[1]),[]).append((key[2],arrays['|'.join(map(str,key))]))
    combined={}
    for (layer,method),aa in cohorts.items():
        aa=sorted(aa,key=lambda item:item[0]);a=np.stack([v for _,v in aa],axis=3)
        h=a[...,0].prod(-1);combined[layer,method]=h
        for oi,op in [(None,'macro')]+list(enumerate(ops)):
            values=h if oi is None else h[:,:,oi]
            cluster=values.reshape(n,-1).mean(1)
            ci=np.quantile(weights@cluster/n,[.025,.975])
            byseed=values.reshape(-1,len(aa)).mean(0)
            cell=dict(layer=layer,method=method,operation=op,complete_binding_accuracy=float(cluster.mean()),
                interval_points=(100*ci).tolist(),seed_ids=[s for s,_ in aa],seed_scores_points=(byseed*100).tolist())
            if oi is not None and op!='both':
                req=0 if op=='first' else 1
                cell['requested_accuracy']=float(a[:,:,oi,:,req,0].mean())
                cell['preserved_accuracy']=float(a[:,:,oi,:,1-req,0].mean())
            macro.append(cell)
    for (layer,method),h in combined.items():
        refs=['sae64','sae256','country64','assignment_country','target_country','member_country','query_member_country','raw_readout_country','code_readout_country','raw']
        for ref in refs:
            if (layer,ref) not in combined or method==ref:continue
            b=combined[layer,ref]
            for oi,op in [(None,'macro')]+list(enumerate(ops)):
                ah=h if oi is None else h[:,:,oi];bh=b if oi is None else b[:,:,oi]
                delta=ah.reshape(n,-1).mean(1)-bh.reshape(n,-1).mean(1)
                macro_contrasts.append(dict(layer=layer,method=method,reference=ref,operation=op,
                    difference_points=float(delta.mean()*100),interval_points=(100*np.quantile(weights@delta/n,[.025,.975])).tolist()))
    out=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),run=run.as_posix(),raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        status=status,cells=cells,contrasts=contrasts,base_accuracy=float(np.mean([r['correct'] for r in base.values()])),
        base_count=len(base),macro=macro,macro_contrasts=macro_contrasts,
        statistics='10,000 paired context resamples, seed9412219; both entity queries, prompt forms, operations and dependent fixed SAE cohort stay together.',
        scope=json.loads((run/'config.resolved.json').read_text())['scope'])
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(out,indent=2)+'\n')
    np.savez_compressed(output.with_suffix('.npz'),**arrays,contexts=np.array(contexts),operations=np.array(ops))
    print(json.dumps(dict(base_accuracy=out['base_accuracy'],cells=cells,contrasts=[x for x in contrasts if x['method'].split('|')[1].startswith(('country','learned'))]),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.run,a.output)
