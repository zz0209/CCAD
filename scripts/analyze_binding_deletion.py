"""Compare target attenuation with the actual source deletion, by context."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


def analyze(runs, output):
    groups={}; identities=[]
    for run in runs:
        status=json.loads((run/'status.json').read_text())
        assert status['status']=='PASS',status
        raw=run/'metrics.raw.jsonl'
        rows=[json.loads(s) for s in raw.read_text().splitlines()]
        rr=[r for r in rows if r['kind']=='intervention']
        identities.append(dict(run=run.as_posix(),status=status,
                               raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest()))
        for method in sorted({r['method'] for r in rr}):
            records=sorted([r for r in rr if r['method']==method],
                           key=lambda r:(r['component'],r['task'],r['operation'],r['query']))
            if method in groups:
                old=groups[method]
                for a,b in zip(old,records,strict=True):
                    for k in ['row_id','operation','answer_id','correct','requested','expected_id']:
                        assert a[k]==b[k],(method,k)
                    assert abs(a['expected_log_probability']-b['expected_log_probability'])<1e-5
            else:groups[method]=records
    source=groups['source_delete'];contexts=sorted({r['component'] for r in source})
    operations=sorted({r['operation'] for r in source})
    key=lambda r:(r['component'],r['task'],r['operation'],r['query'])
    source_keys=[key(r) for r in source]
    assert len(set(source_keys))==len(source_keys)
    cluster=np.array([contexts.index(r['component']) for r in source]);n=len(contexts)
    requested=np.array([r['requested'] for r in source]);correct=np.array([r['correct'] for r in source])
    op=np.array([r['operation'] for r in source]);answer=np.array([r['answer_id'] for r in source])
    logp=np.array([r['expected_log_probability'] for r in source])
    rng=np.random.default_rng(9471501);weights=rng.multinomial(n,np.full(n,1/n),size=10000)
    def summarize(values,mask):
        counts=np.bincount(cluster[mask],minlength=n)
        totals=np.bincount(cluster[mask],weights=values[mask],minlength=n)
        denominator=weights@counts;good=denominator>0
        samples=(weights[good]@totals)/denominator[good]
        return dict(mean=float(values[mask].mean()),interval=np.quantile(samples,[.025,.975]).tolist(),
                    observations=int(mask.sum()),contexts=int((counts>0).sum()))
    cells=[];scores={};retentions={};maes={}
    for method,records in groups.items():
        assert [key(r) for r in records]==source_keys
        assert all(r['expected_id']==s['expected_id'] for r,s in zip(records,source))
        fidelity=np.array([r['answer_id'] for r in records])==answer
        retention=np.array([r['correct'] for r in records]);agreement=retention==correct
        error=np.abs(np.array([r['expected_log_probability'] for r in records])-logp)
        scores[method]=fidelity;retentions[method]=retention;maes[method]=error
        for operation in ['all']+operations:
            mask=np.ones(len(source),bool) if operation=='all' else op==operation
            req=mask&requested;protected=mask&~requested
            cell=dict(method=method,operation=operation,
                      requested_source_answer=summarize(fidelity,req),
                      requested_original_city_retention=summarize(retention,req),
                      requested_original_city_logp_mae=summarize(error,req),
                      changed_original_city=summarize(agreement,req&~correct),
                      retained_original_city=summarize(agreement,req&correct))
            if protected.any():cell['protected_original_city_retention']=summarize(retention,protected)
            cells.append(cell)
    comparisons=[('deletion_'+m,'general_'+m) for m in ['geometry','gain','mix']]
    comparisons += [('deletion_gain','deletion_geometry'),('deletion_mix','deletion_gain')]
    if 'projected_deletion' in groups:
        comparisons += [('projected_deletion',r) for r in ['deletion_gain','deletion_geometry','deletion_mix']]
    contrasts=[]
    for method,reference in comparisons:
        for operation in ['all']+operations:
            mask=requested&(True if operation=='all' else op==operation)
            contrasts.append(dict(method=method,reference=reference,operation=operation,
                source_answer_difference_points=summarize(100*(scores[method].astype(float)-scores[reference]),mask),
                original_city_logp_mae_difference=summarize(maes[method]-maes[reference],mask)))
    out=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),inputs=identities,
             contexts=n,operations=operations,cells=cells,contrasts=contrasts,
             source_changed_requested=int((requested&~correct).sum()),
             source_retained_requested=int((requested&correct).sum()),
             statistics='10,000 paired context resamples, seed9471501; both forms, queries and operations kept together; one SAE direction.',
             interpretation='Deletion of recipient-donor country-contrast selected source members. Original-city retention is not replacement success. Exact source amplitudes are provided equally. Development only.')
    output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps([c for c in cells if c['operation']=='all'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.runs,a.output)
