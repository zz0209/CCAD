"""Run the four frozen confirmation cells sequentially under one GPU lease."""
import json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 cfg=json.loads((ROOT/'configs/f4_training_gain_confirmation_v1.json').read_text());start=time.perf_counter()
 for cell in cfg['cells']:
  status=ROOT/'runs'/cell['run']/'status.json'
  if status.exists():assert json.loads(status.read_text())['status']=='PASS',f'Existing non-PASS preserved: {cell["run"]}'
  else:
   result=subprocess.run([sys.executable,str(ROOT/'scripts/run_long_source_pair.py'),'--config',str(ROOT/cell['config'])],cwd=ROOT)
   if result.returncode:raise RuntimeError(f'Run failed and preserved: {cell["run"]}')
  print(json.dumps(dict(stage='CELL_READY',seed=cell['seed'],step=cell['step'],seconds=time.perf_counter()-start)),flush=True)
  if time.perf_counter()-start>600:raise TimeoutError('Confirmation budget exhausted; completed runs retained')
 print(json.dumps(dict(stage='ALL_CELLS_READY',seconds=time.perf_counter()-start)),flush=True)
if __name__=='__main__':main()
