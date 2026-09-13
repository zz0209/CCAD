"""Export an actually evaluated native member query from its frozen request file."""
import argparse,json,hashlib
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--objective',required=True);p.add_argument('--target',type=int,required=True);p.add_argument('--delete',type=int,choices=range(3),required=True);p.add_argument('--preserve',type=int,choices=range(3),required=True);p.add_argument('--family',default='response_query');p.add_argument('--strategy',default='conditional_path',choices=['full','difference','own_matched','soft_matched','unconditional','conditional_path','conditional_scalar','conditional_clean']);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert a.delete!=a.preserve and not a.out.exists()
    path=a.run/'QUERY_FREEZE.json';cfg=json.loads((a.run/'config.resolved.json').read_text());rows=json.loads(path.read_text())['rows']
    matches=[r for r in rows if r['objective']==a.objective and r['target']==a.target and r['operation']==f'{a.delete}>{a.preserve}' and r['family']==a.family and r['strategy']==a.strategy];assert len(matches)==1
    payload=dict(**matches[0],delete_function=cfg['source_tasks'][a.delete],preserve_function=cfg['source_tasks'][a.preserve],model_revision=cfg['model_revision'],request_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),execution='Encode the clean hook state with this exact target SAE. Subtract the sum of z_j d_j over these member IDs at the shared-prefix site. Preserve the original reconstruction residual and all other positions. Each member is removed once. Full model evaluation is implemented in scripts/relational_exclusion_queries.py.')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(dict(output=str(a.out),members=payload['actual_members'],delete_function=payload['delete_function'],preserve_function=payload['preserve_function'])))


if __name__=='__main__':main()
