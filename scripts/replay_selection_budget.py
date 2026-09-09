"""Evaluate declared calibration-only budget policies on retained candidates.

This is a development reanalysis. It never changes the original run choices.
Candidate held outcomes are joined only after a policy returns its choice.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ccad.selection_budget import budget_select

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def replay(run,output):
    cfg=json.loads((run/'config.resolved.json').read_text())
    choices=json.loads((run/'selection_choices.json').read_text())['choices']
    result=[];sources=[]
    for c in choices:
        path=run/(c['query']+'_calibration.npz')
        with np.load(path) as a:
            names=a['names'].tolist()
            assert set(names)==set(c['scores']['source_endpoint'])
            records=budget_select(names,a['actual_margin'],a['predicted_endpoint'],c['scores'],cfg['screening_budgets'],writer_forward_ratio=cfg.get('selection_writer_forward_ratio',0.),source_rows=cfg.get('selection_source_rows'))
        result.extend(dict(query=c['query'],objective=c['objective'],task=c['task'],source_seed=c['source_seed'],target_seed=c['target_seed'],**r,held_iia=c['held_summary'][r['selected']]['iia']) for r in records)
        sources.append(dict(path=str(path),sha256=sha(path)))
    out=dict(written_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=str(run),scope='Posthoc development reanalysis of an unchanged candidate inventory. Policies consume only retained calibration prefixes; held outcomes evaluate their already returned choice. Original choices are preserved. Work proxies are not measured equal wall time.',code=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in [Path(__file__).resolve(),ROOT/'src/ccad/selection_budget.py']],inputs=[dict(path=str(run/n),sha256=sha(run/n)) for n in ['selection_choices.json','config.resolved.json']]+sources,records=result)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(path=str(output),queries=len(choices),records=len(result))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();replay(a.run,a.output)
