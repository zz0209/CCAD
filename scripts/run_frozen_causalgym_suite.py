"""Execute only a previously resolved, hashed CausalGym confirmation queue.

Uses the existing resource manager for each job. A failed experiment stops the
queue and leaves every file intact; it is never retried or relabeled here.
"""
from pathlib import Path
import json,hashlib,subprocess,sys,os,urllib.request
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/final_three_research_20260909/r21_confirmation'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
queue=json.loads((OUT/'SUITE_QUEUE.json').read_text());freeze=ROOT/queue['freeze_manifest'];assert sha(freeze)==queue['freeze_sha256']
frozen=json.loads(freeze.read_text())
for path,h in frozen['code_sha256'].items():assert sha(ROOT/path)==h,('Frozen code changed',path)
for item in queue['configs']:assert sha(ROOT/item['path'])==item['sha256'],('Config changed',item['path'])
dataset=Path('D:/CCAD_Storage/references/causalgym/95349c3a5e53e2506e8b212482ea6dd784978156');test=dataset/'test.json'
if not test.exists():
 start=now();url='https://huggingface.co/datasets/aryaman/causalgym/resolve/95349c3a5e53e2506e8b212482ea6dd784978156/test.json'
 with urllib.request.urlopen(url,timeout=60) as r:payload=r.read()
 assert payload.lstrip().startswith(b'['),'Unexpected test asset format'
 test.write_bytes(payload)
 write(OUT/'TEST_RETRIEVAL.json',dict(started_at_utc=start,completed_at_utc=now(),path=str(test),url=url,sha256=sha(test),bytes=len(payload),freeze_path=str(freeze),freeze_sha256=sha(freeze),scope='Downloaded after resolved procedure freeze; JSON content not yet parsed by this retrieval step.'))
assert (OUT/'TEST_RETRIEVAL.json').exists(),'A cached test needs an explicit provenance review before use'
assert json.loads((OUT/'TEST_RETRIEVAL.json').read_text())['sha256']==sha(test)
env=os.environ.copy();env['PYTHONPATH']=';'.join(['D:/CCAD_Storage/environments/r005a_sparsify_overlay','D:/CCAD_Storage/references/source/sparsify_42c0645',str(ROOT/'src')]);completed=[]
for item in queue['configs']:
 run=ROOT/'runs'/item['run_id']
 if run.exists():
  status=json.loads((run/'status.json').read_text());assert status['status']=='PASS',('Existing incomplete/failed run retained; no automatic retry',run,status)
  assert json.loads((run/'config.resolved.json').read_text())==json.loads((ROOT/item['path']).read_text())
  completed.append(item['run_id']);continue
 progress=dict(written_at_utc=now(),status='RUNNING',current_run=item['run_id'],completed=completed,queue_count=len(queue['configs']))
 write(OUT/'SUITE_PROGRESS.json',progress);print(json.dumps(progress),flush=True)
 cmd=[sys.executable,str(ROOT.parent/'.resource_manager/resource_manager.py'),'run','--resource','gpu-0','--project','CCAD','--task',item['run_id'],'--wait-sec','43200','--heartbeat-sec','20','--',sys.executable,'scripts/run_causalgym_group_selector.py','--config',item['path']]
 result=subprocess.run(cmd,cwd=ROOT,env=env)
 if result.returncode:
  write(OUT/'SUITE_PROGRESS.json',dict(written_at_utc=now(),status='STOPPED_AT_FAILURE',failed_run=item['run_id'],returncode=result.returncode,completed=completed,scope='All failure artifacts retained. No following task started.'))
  raise SystemExit(result.returncode)
 completed.append(item['run_id'])
write(OUT/'SUITE_PROGRESS.json',dict(written_at_utc=now(),status='COMPLETE',completed=completed,queue_count=len(queue['configs'])));print('FROZEN_SUITE_COMPLETE',flush=True)
