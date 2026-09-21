from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import numpy as np
from analyze_program_generalization import read_run


ROOT=Path(__file__).resolve().parents[1]
BULK=Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round01')
OUT=ROOT/'artifacts/reuse_generalization_20260921_round01'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--objective',choices=['local','local_parts','state_parts','state_writer'])
    parser.add_argument('--steps',type=int,default=4096)
    args=parser.parse_args()
    if args.prepare:
        assert args.objective
        c=json.loads((ROOT/'configs/fs05_generic_local_agreement_dev.json').read_text())
        run=BULK/f'RG01_{args.objective.upper()}{"_DEVELOPMENT" if args.objective.startswith("state_") else ""}_20260921'
        checkpoint=run/f'generic_program_{args.steps}/resid_4_seed3.pt'
        assert checkpoint.exists()
        c.update(run_id=f'RG01_{args.objective.upper()}_AGREEMENT_{args.steps}_20260921',
                 run_parent='REUSE_GENERALIZATION_01',run_storage_root=str(BULK),
                 purpose='Evaluate a frozen generic-action dictionary on the complete excluded agreement explanation.',
                 transfer_training_run=str(run),request_design=args.objective,
                 task_adapted_reference={'grammar':str(checkpoint)})
        if args.objective=='state_writer': c['program_writer_reference']=str(checkpoint.parent/'program_writers.pt')
        path=ROOT/f'configs/rg01_{args.objective}_agreement_{args.steps}.json'
        assert not path.exists()
        path.write_text(json.dumps(c,indent=2)+'\n')
        return
    records={}; audits=[]
    objectives=[args.objective] if args.objective else ['local','local_parts','state_parts','state_writer']
    for objective in objectives:
        run=BULK/f'RG01_{objective.upper()}{"_DEVELOPMENT" if objective.startswith("state_") else ""}_20260921'
        records[objective]={'infinitive':read_run(run)}
        for steps in [512,4096]:
            other=BULK/f'RG01_{objective.upper()}_AGREEMENT_{steps}_20260921'
            if other.exists(): records[objective][f'agreement_{steps}']=read_run(other)
        rows=json.loads((run/'program_audit.json').read_text())['rows']
        for context in ['fit','held_context']:
            for kind in ['aggregate','singleton']:
                initial=[r for r in rows if r['context']==context and r['request_kind']==kind and r['method']=='initial']
                trained=[r for r in rows if r['context']==context and r['request_kind']==kind and r['method']=='trained']
                assert len(initial)==len(trained)>0
                for a,b in zip(initial,trained):
                    for k in ['item','sequences','bank_indices','q','source_hidden_energy','source_distribution_energy']:
                        assert a[k]==b[k], k
                record=dict(objective=objective,context=context,request_kind=kind,items=len(initial),
                    sequences=len({i for r in initial for i in r['sequences']}))
                for metric in ['local_relative_mse','column_relative_mse','downstream_relative_mse','distribution_relative_kl']:
                    a=np.array([r[metric] for r in initial]); b=np.array([r[metric] for r in trained])
                    record[metric]=dict(initial=float(a.mean()),trained=float(b.mean()),differences=(b-a).tolist())
                audits.append(record)
    payload=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),evidence='Development. One shared target and dependent natural actions; function contexts have been exposed in previous rounds.',records=records,audits=audits)
    suffix='_'+args.objective if args.objective else ''
    dest=OUT/f'LEARNING_ANALYSIS{suffix}.json'
    assert not dest.exists()
    dest.write_text(json.dumps(payload,indent=2)+'\n')
    for objective,sets in records.items():
        for name,record in sets.items():
            print(json.dumps(dict(objective=objective,dataset=name,methods={m:v['families'] for m,v in record['methods'].items()})))
    for a in audits:
        print(json.dumps({k:({kk:vv for kk,vv in v.items() if kk!='differences'} if isinstance(v,dict) else v) for k,v in a.items()}))


if __name__=='__main__':
    main()
