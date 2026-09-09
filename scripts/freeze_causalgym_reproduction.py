"""Record a new, explicitly labeled reproduction freeze for relocated configs.

Historical freezes are never rewritten. Path changes and new run IDs change
their hashes, so a new execution must have its own prospective identity. This
helper does not decide method settings, inspect task data, download anything
or claim independent scientific confirmation.
"""
from pathlib import Path
import argparse,json,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
CODE=['scripts/run_causalgym_group_selector.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/native_group_selection.py','src/ccad/finite_native_group.py','src/ccad/axis_transfer_selection.py','src/ccad/native_operation.py','src/ccad/selection_budget.py','src/ccad/causalgym_interface.py','src/ccad/semantic_context_matching.py','src/ccad/nip_baselines.py','src/ccad/artifacts.py']
def main():
 p=argparse.ArgumentParser();p.add_argument('--configs',type=Path,nargs='+',required=True);p.add_argument('--output-directory',type=Path,required=True);p.add_argument('--historical-freeze',type=Path,required=True);a=p.parse_args()
 if a.output_directory.exists():raise FileExistsError('Use a fresh directory; no historical freeze or config is overwritten')
 original=json.loads(a.historical_freeze.read_text());a.output_directory.mkdir(parents=True);runs={};configs=[]
 for path in a.configs:
  cfg=json.loads(path.read_text());cfg.pop('freeze_manifest',None);cfg.pop('freeze_manifest_sha256',None)
  if cfg['run_id'] in original['runs'] or (ROOT/'runs'/cfg['run_id']).exists():raise ValueError('Each reproduction needs a new unused run ID')
  if cfg.get('evaluation_split')!='test' or cfg.get('audit_opened') is not True:raise ValueError('This helper is for explicitly declared test reproductions')
  h=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest();runs[cfg['run_id']]=dict(canonical_config_sha256=h,source_config=str(path.resolve()));configs.append((path,cfg))
 manifest=a.output_directory/'REPRODUCTION_FREEZE.json'
 result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),runs=runs,code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in CODE},historical_freeze=str(a.historical_freeze.resolve()),historical_freeze_sha256=hashlib.sha256(a.historical_freeze.read_bytes()).hexdigest(),scope='New user-prepared execution identity. No data read here; this is not independent confirmation merely because paths, host or run IDs changed. Compare scientific settings with the historical freeze explicitly.')
 manifest.write_text(json.dumps(result,indent=2)+'\n');h=hashlib.sha256(manifest.read_bytes()).hexdigest()
 for path,cfg in configs:
  cfg.update(freeze_manifest=str(manifest.resolve()),freeze_manifest_sha256=h);dest=a.output_directory/path.name
  if dest.exists():raise FileExistsError('Duplicate config basename')
  dest.write_text(json.dumps(cfg,indent=2)+'\n')
 print(json.dumps(dict(manifest=str(manifest.resolve()),sha256=h,configs=len(configs),scope=result['scope'])))
if __name__=='__main__':main()
