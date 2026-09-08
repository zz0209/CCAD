"""Pair-dependent summaries of the evolving, explicitly exposed RAVEL panel."""
from __future__ import annotations
import json
import hashlib
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/final_five_research_20260908/r16_semantic_operations'
NAMES=['FINAL5_R16_ravel_coverage_l3_v1_20260908','FINAL5_R16_ravel_semantic_source_v1_20260908',
       'FINAL5_R16_ravel_coverage_k128_v1_20260908','FINAL5_R16_ravel_raw_semantic_l7_v1_20260908']

def summarize(records):
    pairs=sorted({r['component'] for r in records});by_pair=[]
    for pair in pairs:
        rr=[r for r in records if r['component']==pair]
        cause=[r['first_token_correct'] for r in rr if r['endpoint']=='Cause']
        iso=[r['first_token_correct'] for r in rr if r['endpoint']=='Iso']
        c=float(np.mean(cause)) if cause else None;i=float(np.mean(iso)) if iso else None
        by_pair.append(dict(component=pair,cause=c,iso=i,disentangle=(c+i)/2 if c is not None and i is not None else None))
    result=dict(rows=len(records),independent_city_pairs=len(pairs),per_pair=by_pair)
    rng=np.random.default_rng(160908);sample=rng.integers(0,len(pairs),(10000,len(pairs)))
    for metric in ['cause','iso','disentangle']:
        values=[r[metric] for r in by_pair]
        if all(x is not None for x in values):
            arr=np.asarray(values);bootstrap=arr[sample].mean(1)
            result[metric]=dict(mean=float(arr.mean()),conditional_pair_bootstrap95=np.quantile(bootstrap,[.025,.975]).tolist())
    return result

def main():
    summaries=[];inputs=[];coverage=[]
    panel=json.loads((OUT/'semantic_panel_v1.json').read_text())['rows']
    controls=[[1,0,0],[0,1,0],[0,0,1]];attrs=['Country','Continent','Language']
    for name in NAMES:
        run=ROOT/'runs'/name
        if not (run/'semantic_summary.json').exists():continue
        cfg=json.loads((run/'config.resolved.json').read_text());path=run/'metrics.raw.jsonl';rows=list(map(json.loads,path.read_text().splitlines()))
        inputs.append(dict(run_id=name,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if 'coverage' in name:
            for c in json.loads((run/'semantic_summary.json').read_text())['cells']:coverage.append(dict(run_id=name,**c))
            synthetic=[]
            for row in rows:
                if row['split']!='held_component_development' or row['method'] not in ['unedited','raw_donor'] or row['mode'] not in ['none','region_1']:continue
                p=panel[row['row_id']];a=attrs.index(p['task'])
                for control in controls:
                    cause=bool(control[a]);accepted=p['donor_expected_ids'] if cause else p['expected_ids']
                    synthetic.append(dict(**{k:row[k] for k in ['component','task','method','row_id']},
                        operation=''.join(map(str,control)),control=control,endpoint='Cause' if cause else 'Iso',
                        first_token_correct=row['predicted_token_id'] in accepted))
            rows=synthetic
        for method in sorted({r['method'] for r in rows}):
            rr=[r for r in rows if r['method']==method and sum(r['control'])==1]
            if rr:summaries.append(dict(run_id=name,layer=cfg['layer'],method=method,operation_family='three_single_controls',
                **summarize(rr),per_attribute={a:summarize([r for r in rr if r['task']==a]) for a in attrs}))
    output=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),inputs=inputs,source_semantic=summaries,coverage=coverage,
        scope='Conditional descriptive uncertainty over the eight disjoint held development city pairs. Reciprocal directions, templates and attributes stay clustered. Not uncertainty over model or SAE seeds; not independent confirmation. Baseline controls are rescored from actual recorded unedited/raw predictions without additional model calls.')
    (OUT/'SEMANTIC_DEVELOPMENT_SUMMARY.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    for row in summaries:print(json.dumps({k:row[k] for k in ['run_id','layer','method','cause','iso','disentangle']}))

if __name__=='__main__':main()
