"""One fixed support-objective comparison on existing paired assets."""
import json,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 cfg=json.loads((ROOT/'configs/f4_operation_metric_panel_v1.json').read_text());out=ROOT/'artifacts/operation_metric_20260906';out.mkdir(exist_ok=True);start=time.perf_counter();rows=[]
 for i,c in enumerate(cfg['cells']):
  run=ROOT/'runs'/c['run']
  if not run.exists():
   subprocess.run([sys.executable,str(ROOT/'scripts/run_long_source_pair.py'),'--config',str(ROOT/c['config'])],cwd=ROOT,check=True)
  assert json.loads((run/'status.json').read_text())['status']=='PASS'
  assert json.loads((run/'contract_validation.json').read_text())['ok']
  with np.load(run/'coefficients.npz') as new,np.load(ROOT/'runs'/c['parent']/'coefficients.npz') as old:
   for m in old.files:
    if m!='wrong_query':assert np.array_equal(new[m],old[m]),(c,m,'Legacy numerical path changed')
  d=json.loads((run/'calibration_diagnostic.json').read_text());d.update(query=c['query'],seed=c['seed'],run=c['run']);rows.append(d)
  p=dict(completed=i+1,total=len(cfg['cells']),seconds=time.perf_counter()-start)
  (out/'progress.json').write_text(json.dumps(p)+'\n');print(json.dumps(p),flush=True)
  if p['seconds']>600:raise TimeoutError('Ten-minute panel budget; completed cells preserved')
 (out/'calibration_results.json').write_text(json.dumps(dict(cells=rows,legacy_coefficients_exact=True,seconds=time.perf_counter()-start),indent=2)+'\n')
if __name__=='__main__':main()
