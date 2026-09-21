from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if json.loads((a.run/'status.json').read_text())['status']!='PASS':raise ValueError('Incomplete run')
    membership=json.loads((a.run/'membership.json').read_text());q=membership['human_query_order']
    arrays=np.load(a.run/'pooled.npz');source=arrays['source'].astype(float);clean=arrays['none'].astype(float)
    energy=np.mean(np.sum((source-clean)**2,axis=-1),axis=1)
    if np.any(energy<=1e-12):raise ValueError('Source field is zero')
    representations={}
    for key in arrays.files:
        if key in ['source','none']:continue
        per=np.sqrt(np.mean(np.sum((arrays[key]-source)**2,axis=-1),axis=1)/energy)
        representations[key]=dict(mean=float(per.mean()),per_query=dict(zip(q,per.tolist())))
    diagnostics=json.loads((a.run/'execution_diagnostics.json').read_text())
    by_method={};details=[]
    for row in diagnostics:
        if row['dataset']!='human' or not row['sites']:continue
        method=row['method'];total=by_method.setdefault(method,dict(changed=0,states=0,zero_site_changed=0,zero_site_desired_energy=0.,desired_energy=0.))
        for site,diag in row['sites'].items():
            if 'changed' not in diag:continue
            zero=not any(membership['human_queries'][row['query']][site])
            total['changed']+=diag['changed'];total['states']+=diag['states']
            total['desired_energy']+=diag.get('desired_energy',0.)
            if zero:
                total['zero_site_changed']+=diag['changed'];total['zero_site_desired_energy']+=diag.get('desired_energy',0.)
            details.append(dict(method=method,query=row['query'],site=site,source_request_empty=zero,**diag))
    for total in by_method.values():
        total['changed_per_site_token']=total['changed']/total['states']
        total['fraction_changes_at_empty_source_sites']=total['zero_site_changed']/max(total['changed'],1)
    out=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),run=str(a.run),representations=representations,
        execution=by_method,site_details=details,
        scope='Diagnostic on all recorded queries. Site-token counts reflect actual execution, with repeated documents and requests. Empty source sites are identified from the named request, independently of model responses.')
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(representations={k:v['mean'] for k,v in representations.items()},execution=by_method),indent=2))


if __name__=='__main__':main()
