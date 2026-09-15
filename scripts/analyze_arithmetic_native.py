"""Summarize the exposed-panel arithmetic target-code execution pilot."""
from pathlib import Path
from collections import defaultdict
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def answer(r):
    return ('number',r['answer']) if r['answer'] is not None else ('text',r['generated_text'].strip())

def main():
    run=ROOT/'runs/REFORM_R43_arithmetic_native_execution_pilot_v1_20260915'
    status=json.loads((run/'status.json').read_text());assert status['status']=='PASS'
    rows=[json.loads(s) for s in (run/'metrics.raw.jsonl').read_text().splitlines()]
    groups=defaultdict(list)
    for r in rows:
        if r['kind']=='source_patch':groups[r['seed'],r['method'],r['operation']].append(r)
    cells=[]
    for (seed,method,op),rr in groups.items():
        if (seed==1 and method.startswith('source')) or (seed==2 and not method.startswith('source')):
            cells.append(dict(seed=seed,method=method,operation=op,n=len(rr),
                **{name:100*float(np.mean([r[key] for r in rr])) for name,key in
                   [('H','exact_hybrid'),('T','target_digit_success'),('P','preserve_digit_success')]}))
    lookup={(r['seed'],r['method'],r['operation'],r['row_id']):r for r in rows if r['kind']=='source_patch'}
    fidelity=[]
    for method in ['member','assignment','raw_readout','activation_readout','native_code']:
        counts=np.zeros(4)
        for rid in range(32):
            base=answer(lookup[0,'no_edit','unit',rid])
            for op in ['unit','tens']:
                for part in [0,1]:
                    src=answer(lookup[1,f'source_part{part}_bank0',op,rid])
                    pred=answer(lookup[2,f'{method}_part{part}_bank0',op,rid])
                    changed=src!=base;counts[2*int(changed)]+=1;counts[2*int(changed)+1]+=src==pred
        fidelity.append(dict(method=method,balanced=50*(counts[1]/counts[0]+counts[3]/counts[2]),counts=counts.tolist()))
    d=json.loads((run/'NATIVE_EXECUTION.json').read_text())['records']
    out=dict(status=status,run=run.relative_to(ROOT).as_posix(),scope='16 exposed question clusters, two formats, source1 to target2; development signal only',
        cells=cells,fidelity=fidelity,native=dict(calls=len(d),min_edited_code=min(r['min_edited_code'] for r in d),
        max_members=max(r['max_changed_members'] for r in d),solver_seconds=sum(r['wall_seconds'] for r in d)),
        raw_sha256=hashlib.sha256((run/'metrics.raw.jsonl').read_bytes()).hexdigest())
    (ROOT/'artifacts/correspondence_reform_20260913/r43_arithmetic_native_pilot.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out['fidelity']))

if __name__=='__main__':main()
