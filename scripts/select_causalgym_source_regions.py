"""Frozen source-only region rule; no target SAE outcomes enter this selection."""
from pathlib import Path
import argparse,json,hashlib

def select(run):
    cfg=json.loads((run/'config.resolved.json').read_text());summary=json.loads((run/'screen_summary.json').read_text())
    assert json.loads((run/'status.json').read_text())['status']=='PASS'
    result={};evidence=[]
    for model in cfg['models']:
        positions={}
        for task in cfg['tasks']:
            rows=[r for r in summary['rows'] if r['model']==model['name'] and r['task']==task and r['n']==cfg['rows_per_task']]
            assert rows,('No complete-coverage source region',model['name'],task)
            winner=max(rows,key=lambda r:(r['iia'],r['log_odds_ratio'],-int(r['position'].split('_')[-1])))
            positions[task]=winner['position'];evidence.append(winner)
        result[model['name']]=positions
    return dict(positions=result,selection_evidence=evidence,rule='First declared32 original training rows; only full-coverage original non-null regions; maximize raw source IIA, then mean logodds, then lowest region number. Never inspect target SAE values/outcomes.',source_run=str(run),source_summary_sha256=hashlib.sha256((run/'screen_summary.json').read_bytes()).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=select(a.run);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
    for r in out['selection_evidence']:print(r['model'],r['task'],r['position'],r['iia'],round(r['log_odds_ratio'],3))
