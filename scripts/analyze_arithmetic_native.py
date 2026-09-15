"""Summarize the exposed-panel arithmetic target-code execution pilot."""
from pathlib import Path
from collections import defaultdict
import json,hashlib,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def answer(r):
    return ('number',r['answer']) if r['answer'] is not None else ('text',r['generated_text'].strip())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='runs/REFORM_R43_arithmetic_native_execution_pilot_v1_20260915');ap.add_argument('--output',default='artifacts/correspondence_reform_20260913/r43_arithmetic_native_pilot.json');ap.add_argument('--reference-run',action='append',default=[]);args=ap.parse_args()
    run=ROOT/args.run
    status=json.loads((run/'status.json').read_text());assert status['status']=='PASS'
    rows=[json.loads(s) for s in (run/'metrics.raw.jsonl').read_text().splitlines()]
    inputs=[]
    panel=json.loads((run/'panel.json').read_text())
    for name in args.reference_run:
        previous=ROOT/name
        assert json.loads((previous/'status.json').read_text())['status']=='PASS'
        old=json.loads((previous/'panel.json').read_text())
        assert {k:v for k,v in old.items() if k!='scope'}=={k:v for k,v in panel.items() if k!='scope'}
        rows += [json.loads(s) for s in (previous/'metrics.raw.jsonl').read_text().splitlines()]
        inputs.append(dict(run=name,raw_sha256=hashlib.sha256((previous/'metrics.raw.jsonl').read_bytes()).hexdigest()))
    keys=[(r['seed'],r['method'],r['operation'],r['row_id']) for r in rows if r['kind']=='source_patch']
    assert len(keys)==len(set(keys)), 'Repeated intervention keys'
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
    methods=sorted({r['method'].removesuffix('_part0_bank0') for r in rows if r['kind']=='source_patch' and r['seed']==2 and r['method'].endswith('_part0_bank0') and not r['method'].startswith('source')})
    row_ids=sorted({r['row_id'] for r in rows if r['kind']=='source_patch' and r['method']=='no_edit'})
    for method in methods:
        counts=np.zeros(4)
        for rid in row_ids:
            base=answer(lookup[0,'no_edit','unit',rid])
            for op in ['unit','tens']:
                for part in [0,1]:
                    src=answer(lookup[1,f'source_part{part}_bank0',op,rid])
                    pred=answer(lookup[2,f'{method}_part{part}_bank0',op,rid])
                    changed=src!=base;counts[2*int(changed)]+=1;counts[2*int(changed)+1]+=src==pred
        fidelity.append(dict(method=method,balanced=50*(counts[1]/counts[0]+counts[3]/counts[2]),counts=counts.tolist()))
    d=json.loads((run/'NATIVE_EXECUTION.json').read_text())['records']
    out=dict(status=status,run=run.relative_to(ROOT).as_posix(),scope=f'{len(row_ids)//2} exposed question clusters, two formats, source1 to target2; development signal only',
        cells=cells,fidelity=fidelity,native=dict(calls=len(d),min_edited_code=min(r['min_edited_code'] for r in d),
        max_members=max(r['max_changed_members'] for r in d),solver_seconds=sum(r['wall_seconds'] for r in d)),
        raw_sha256=hashlib.sha256((run/'metrics.raw.jsonl').read_bytes()).hexdigest())
    out['reference_runs']=inputs
    if (run/'RESPONSE_EXECUTION.json').exists():
        dd=json.loads((run/'RESPONSE_EXECUTION.json').read_text())['records'];flat=[r for d in dd for r in d['records']]
        out['response']=dict(calls=len(flat),initial_kl=float(np.mean([x for r in flat for x in r['initial']['kl']])),final_kl=float(np.mean([x for r in flat for x in r['final_kl']])),solver_seconds=sum(r['solver_seconds'] for r in flat),min_edited_code=min(r['min_edited_code'] for r in flat),max_changed_members=max(r['max_changed_members'] for r in flat),backward_batches=sum(r['backward_batches'] for r in flat))
        out['response_by_method']=[]
        for method in sorted({d['method'] for d in dd}):
            ff=[r for d in dd if d['method']==method for r in d['records']]
            out['response_by_method'].append(dict(method=method,calls=len(ff),initial_kl=float(np.mean([x for r in ff for x in r['initial']['kl']])),final_kl=float(np.mean([x for r in ff for x in r['final_kl']])),solver_seconds=sum(r['solver_seconds'] for r in ff),min_edited_code=min(r['min_edited_code'] for r in ff),max_changed_members=max(r['max_changed_members'] for r in ff)))
    (ROOT/args.output).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out['fidelity']))
    if 'response' in out:print(json.dumps(out['response']))

if __name__=='__main__':main()
