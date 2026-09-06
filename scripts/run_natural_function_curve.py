"""Resume fixed natural-checkpoint cells after an exact early-checkpoint witness."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
 cfg=json.loads((ROOT/'configs/f4_natural_function_curve_v1.json').read_text());out=ROOT/'artifacts/natural_function_curve_20260906';start=time.perf_counter()
 def run(config,run_id):
  status=ROOT/'runs'/run_id/'status.json'
  if status.exists():
   assert json.loads(status.read_text())['status']=='PASS',f'Existing non-PASS run retained: {run_id}';return
  result=subprocess.run([sys.executable,str(ROOT/'scripts/run_long_source_pair.py'),'--config',str(ROOT/config)],cwd=ROOT)
  if result.returncode:raise RuntimeError(f'Run failed and preserved: {run_id}')
 run(cfg['witness_config'],cfg['witness_run']);parent=ROOT/'runs'/cfg['witness_parent'];child=ROOT/'runs'/cfg['witness_run'];rows=lambda p:[json.loads(s) for s in (p/'metrics.raw.jsonl').read_text().splitlines()];ref={(r['case_id'],r['method'],r['operator']):r for r in rows(parent)}
 fields=['source_difference','predicted_difference','source_activation','common_dose','source_kl','candidate_kl','normalized_kl_error','vector_squared_error'];bad=[]
 for row in rows(child):
  key=row['case_id'],row['method'],row['operator']
  for field in fields:
   if row[field]!=ref[key][field]:bad.append(dict(key=key,field=field,new=row[field],old=ref[key][field]))
 (out/'early_checkpoint_replay.json').write_text(json.dumps(dict(exact=not bad,rows=len(rows(child)),fields=fields,differences=bad),indent=2)+'\n');assert not bad,bad[:3]
 for cell in cfg['cells']:
  if not cell['reuse']:run(cell['config'],cell['run'])
  print(json.dumps(dict(stage='CELL_READY',**cell,seconds=time.perf_counter()-start)),flush=True)
  if time.perf_counter()-start>900:raise TimeoutError('Trajectory wrapper budget; completed runs retained')
 print(json.dumps(dict(stage='ALL_CELLS_READY',seconds=time.perf_counter()-start)),flush=True)
if __name__=='__main__':main()
