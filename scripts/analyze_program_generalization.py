from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/final_science_20260921_round05'
BULK=Path('D:/CCAD_Storage/runs/final_science_20260921_round05')


def read_run(run):
    assert json.loads((run/'status.json').read_text())['status']=='PASS'
    membership=json.loads((run/'membership.json').read_text())
    names=membership['grammar_query_order']
    rows=membership['grammar_rows']
    data=np.load(run/'responses.npz')
    source=data['grammar__source'].astype(float)
    clean=data['grammar__none'].astype(float)
    effect=np.mean((source-clean)**2,axis=1)
    assert effect.min()>1e-12 and len(rows)>0
    families=dict(endpoints=[i for i,n in enumerate(names) if not n.startswith(('participation_','interior_','boundary_','member_'))],
                  participation=[i for i,n in enumerate(names) if n.startswith(('participation_','interior_','boundary_'))],
                  members=[i for i,n in enumerate(names) if n.startswith('member_')])
    assert all(families.values())
    methods={}
    for key in data.files:
        if key in ['grammar__none','grammar__source']:continue
        mse=np.mean((data[key].astype(float)-source)**2,axis=1)
        error=np.sqrt(mse/effect)
        methods[key.split('__',1)[1]]=dict(all=float(error.mean()),
            families={f:float(error[ix].mean()) for f,ix in families.items()},per_query=error.tolist())
    return dict(run=str(run),methods=methods,source_rms=np.sqrt(effect).tolist(),
                queries=names,contexts=len(rows),families=families,
                response_sha256=hashlib.sha256((run/'responses.npz').read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('kind',choices=['cross','generic'])
    args=parser.parse_args()
    if args.kind=='cross':
        records={d:read_run(BULK/f'FS05_CROSS_EXPLANATION_{d.upper()}_DEV_20260921') for d in ['semantic','balanced','independent']}
    else:
        records={d:dict(infinitive=read_run(BULK/f'FS05_GENERIC_{d.upper()}_DEV_20260921'),
                        agreement=read_run(BULK/f'FS05_GENERIC_{d.upper()}_AGREEMENT_DEV_20260921'))
                 for d in ['local','downstream','distribution']}
    path=OUT/f'{args.kind.upper()}_PROGRAM_GENERALIZATION.json'
    assert not path.exists()
    path.write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        evidence='One target initialization, previously exposed development contexts. All source requests retained. Point estimates guide mechanism selection; no population-level confirmation is implied.',
        records=records),indent=2)+'\n')
    if args.kind=='cross':
        print(json.dumps({d:{m:v['families'] for m,v in r['methods'].items()} for d,r in records.items()},indent=2))
    else:
        print(json.dumps({d:{s:{m:v['families'] for m,v in r['methods'].items()} for s,r in ss.items()} for d,ss in records.items()},indent=2))


if __name__=='__main__':
    main()
