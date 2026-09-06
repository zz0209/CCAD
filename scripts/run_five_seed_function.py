"""Resume the fixed fit/application cells without changing their membership."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['fit','apply']);args=ap.parse_args();cfg=json.loads((ROOT/'configs/f4_five_seed_function_v1.json').read_text());start=time.perf_counter();out=ROOT/'artifacts/five_seed_function_20260906'
 if args.phase=='fit':
  fields=['source_difference','predicted_difference','source_activation','common_dose','source_kl','candidate_kl','normalized_kl_error','vector_squared_error'];rows=lambda run:[json.loads(x) for x in (ROOT/'runs'/run/'metrics.raw.jsonl').read_text().splitlines()];a,b=rows(cfg['witness_parent']),rows(cfg['witness_run']);assert len(a)==len(b)==64
  assert all(x[k]==y[k] for x,y in zip(a,b) for k in fields)
  (out/'legacy_replay.json').write_text(json.dumps(dict(exact=True,rows=64,fields=fields))+'\n')
 for i,cell in enumerate(cfg['cells']):
  run=cell['run'] if args.phase=='fit' else cell['apply_run'];config=cell['config'] if args.phase=='fit' else cell['apply_config'];status=ROOT/'runs'/run/'status.json'
  if status.exists():assert json.loads(status.read_text())['status']=='PASS',f'Existing failed cell preserved: {run}'
  else:
   ret=subprocess.run([sys.executable,str(ROOT/'scripts/run_long_source_pair.py'),'--config',str(ROOT/config)],cwd=ROOT)
   if ret.returncode:raise RuntimeError(f'Failed cell preserved: {run}')
  progress=dict(phase=args.phase,completed=i+1,total=len(cfg['cells']),run=run,seconds=time.perf_counter()-start);(out/f'{args.phase}_progress.json').write_text(json.dumps(progress)+'\n');print(json.dumps(progress),flush=True)
  if time.perf_counter()-start>1800:raise TimeoutError('Wrapper30minute budget; completed cells retained')
if __name__=='__main__':main()
