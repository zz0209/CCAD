from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np


ROOT=Path(__file__).resolve().parents[1]
TASKS=[('composer_surgeon_orientation0',[5,25]),('composer_surgeon_orientation1',[5,25]),
       ('model_software_engineer_orientation0',[12,24]),('model_software_engineer_orientation1',[12,24])]


def score(pred,source,clean,indices,documents=None):
    if documents is None:documents=np.arange(source.shape[1])
    s=source[np.ix_(indices,documents)];n=clean[np.ix_(indices,documents)]
    p=pred[np.ix_(indices,documents)]
    energy=np.mean((s-n)**2,axis=1)
    valid=energy>1e-12
    if not valid.any():raise ValueError('No measurable source effect')
    return float(np.sqrt(np.mean((p-s)**2,axis=1)[valid]/energy[valid]).mean())


def study(run):
    status=json.loads((run/'status.json').read_text())
    if status['status']!='PASS':raise ValueError(status)
    membership=json.loads((run/'membership.json').read_text())
    values=np.load(run/'responses.npz');pooled=np.load(run/'pooled.npz')
    queries=membership['human_query_order'];rows=membership['human_rows']
    families={'all':list(range(len(queries))),
              'full':[i for i,q in enumerate(queries) if q=='full'],
              'parts':[i for i,q in enumerate(queries) if q in ['pronouns','names','associated_words']],
              'participation':[i for i,q in enumerate(queries) if q.startswith(('interior','boundary'))],
              'members':[i for i,q in enumerate(queries) if q.startswith('member')]}
    families={k:v for k,v in families.items() if v}
    methods=[k for k in pooled.files if k not in ['none','source']]
    metrics={m:{'original':{},'later':{}} for m in methods};head_arrays=[]
    for m in methods:
        for family,ix in families.items():
            metrics[m]['original'][family]=score(values['human__'+m],values['human__source'],values['human__none'],ix)
    for name,professions in TASKS:
        path=ROOT/'runs/IR04_shift_consumer_seed2_v1_20260916'/f'none__full__{name}__probe42.npz'
        probe=np.load(path);weight=probe['weight'].astype(float).ravel();bias=float(probe['bias'].item())
        keep=np.array([r['profession'] in professions for r in rows])
        arrays={k:pooled[k][:,keep].astype(float)@weight+bias for k in pooled.files}
        head_arrays.append((np.flatnonzero(keep),arrays))
        for m in methods:
            for family,ix in families.items():
                metrics[m]['later'][family]=metrics[m]['later'].get(family,0)+score(arrays[m],arrays['source'],arrays['none'],ix)/len(TASKS)
    grammar={};equivalence={}
    if 'grammar__source' in values:
        gq=membership['grammar_query_order']
        for key in values.files:
            if key.startswith('grammar__') and key not in ['grammar__none','grammar__source']:
                grammar[key]=score(values[key],values['grammar__source'],values['grammar__none'],list(range(len(gq))))
        baseline='grammar__initial_inverse_budget'
        for method in ['initial_trajectory_action','initial_trajectory_feedback']:
            key='grammar__'+method
            if baseline in values and key in values:
                diff=values[key].astype(float)-values[baseline].astype(float)
                equivalence[method]=dict(max_absolute_response_difference=float(abs(diff).max()),rms_response_difference=float(np.sqrt(np.mean(diff**2))))
    return dict(run=str(run),status=status,documents=len(rows),queries=queries,metrics=metrics,
                grammar=grammar,single_site_equivalence=equivalence),head_arrays,rows,families


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--freeze',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    results=[];raw=[];checks={}
    if a.freeze:
        freeze=json.loads(a.freeze.read_text())
        for name,digest in freeze['identities'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError(name)
        checks['freeze_identities_match']=True
    for run in a.runs:
        result,heads,rows,families=study(run);results.append(result);raw.append(heads)
    comparisons={}
    if a.freeze:
        panels=[json.loads((r/'membership.json').read_text()) for r in a.runs]
        if any(x['human_rows']!=panels[0]['human_rows'] or x['human_query_order']!=panels[0]['human_query_order'] for x in panels):raise ValueError('Panel mismatch')
        for heads in raw[1:]:
            for i,(_,arrays) in enumerate(heads):
                if not np.array_equal(arrays['source'],raw[0][i][1]['source']):raise ValueError('Source changed')
        checks['same_panel_and_source']=True
        cells=[[i for i,r in enumerate(rows) if (r['profession'],r['gender'])==cell] for cell in sorted({(r['profession'],r['gender']) for r in rows})]
        rng=np.random.default_rng(2026092103);indices=families['participation']
        winner='initial_trajectory_feedback'
        controls=[m for m in results[0]['metrics'] if m!=winner]
        samples={m:[] for m in controls}
        for _ in range(2000):
            docs=np.concatenate([rng.choice(cell,len(cell),replace=True) for cell in cells])
            qi=rng.choice(indices,len(indices),replace=True);scores={m:0. for m in [winner,*controls]}
            for heads in raw:
                for global_ids,arrays in heads:
                    local_map={j:i for i,j in enumerate(global_ids)}
                    selected=[local_map[j] for j in docs if j in local_map]
                    for method in scores:scores[method]+=score(arrays[method],arrays['source'],arrays['none'],qi,selected)/(len(raw)*len(heads))
            for method in controls:samples[method].append(scores[method]-scores[winner])
        for method in controls:
            delta=np.mean([r['metrics'][method]['later']['participation']-r['metrics'][winner]['later']['participation'] for r in results])
            comparisons[method]=dict(reduction=float(delta),ci95=np.quantile(samples[method],[.025,.975]).tolist())
    output=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),studies=results,comparisons=comparisons,checks=checks,
        scope='Source program, target dictionary set and four later heads fixed. Paired document and request uncertainty when freeze is supplied. Development otherwise.')
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(studies=[dict(run=r['run'],metrics=r['metrics'],grammar=r['grammar'],single_site_equivalence=r['single_site_equivalence']) for r in results],comparisons=comparisons),indent=2))


if __name__=='__main__':main()
