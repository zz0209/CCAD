"""Join the two original and three added trajectories, checking actual input order."""
import hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=ROOT/'artifacts/five_seed_training_20260906';out.mkdir(exist_ok=True);runs=[ROOT/'runs'/x for x in ['R012_same_stream_curve_v1_20260906','R012_same_stream_extra_seeds_v1_20260906']];configs=[json.loads((r/'config.resolved.json').read_text()) for r in runs];allowed={'run_id','init_seeds','bulk_output_dir','budget_seconds','purpose','scope_limit'}
 assert all(configs[0][k]==configs[1][k] for k in configs[0] if k not in allowed)
 traces=[json.loads((r/'training_trace.json').read_text()) for r in runs];assert traces[0]['input_hashes']==traces[1]['input_hashes'] and len(traces[0]['input_hashes'])==8192
 rows=[];identities=[];training=[]
 for run,trace in zip(runs,traces):
  assert json.loads((run/'status.json').read_text())['status']=='PASS';assert json.loads((run/'contract_validation.json').read_text())['ok']
  rows.extend(dict(parent_run=run.name,**json.loads(t)) for t in (run/'metrics.raw.jsonl').read_text().splitlines());identities.append(dict(run=run.name,config_sha256=sha(run/'config.resolved.json'),raw_sha256=sha(run/'metrics.raw.jsonl'),trace_sha256=sha(run/'training_trace.json'),wall_seconds=json.loads((run/'metrics.summary.json').read_text())['wall_seconds']))
  for name,rr in trace['traces'].items():
   for begin in range(0,len(rr),512):
    block=rr[begin:begin+512];training.append(dict(seed=int(name.rsplit('seed',1)[1]),step_begin=begin+1,step_end=begin+len(block),median_fvu=statistics.median(x['fvu'] for x in block),median_auxk=statistics.median(x['auxk_loss'] for x in block)))
 assert len(rows)==25 and {(x['seed'],x['step']) for x in rows}=={(i,j) for i in [1,2,3,4,5] for j in [256,1024,2048,4096,8192]}
 assert len({x['quality']['ce']['clean'] for x in rows})==1 and len({x['quality']['ce']['zero'] for x in rows})==1
 endpoints=[]
 for seed in [1,2,3,4,5]:
  early=next(x for x in rows if x['seed']==seed and x['step']==256);late=next(x for x in rows if x['seed']==seed and x['step']==8192)
  endpoints.append(dict(seed=seed,early=early['quality'],late=late['quality'],delta_fve=late['quality']['fve']-early['quality']['fve'],delta_ce_recovered=late['quality']['ce_recovered']-early['quality']['ce_recovered'],early_norm_max_error=early['decoder_norm_max_error'],late_norm_max_error=late['decoder_norm_max_error']))
 summary=dict(rows=rows,endpoints=endpoints,identities=identities,training_blocks=training,actual_train_input_hashes_identical=True,shared_base_forward_steps=8192,configuration_matches_except=sorted(allowed),new_seed_count=3,total_seed_count=5,scope='Five initialization seeds, same actual input order/architecture/optimizer/schedule/validation. Quality and source preparation only, not five-seed functional confirmation or proof of convergence. Training FVU blocks use different batches.')
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'checkpoints.json').write_text(json.dumps(dict(checkpoints=[{k:v for k,v in x.items() if k not in ['quality','decoder_norm_max_error']} for x in rows]),indent=2)+'\n')
 print(json.dumps(dict(endpoints=[dict(seed=x['seed'],early_fve=x['early']['fve'],late_fve=x['late']['fve'],early_ce=x['early']['ce_recovered'],late_ce=x['late']['ce_recovered'],late_alive=x['late']['alive_features'],late_l0=x['late']['actual_nonzero_l0'],late_norm_max_error=x['late_norm_max_error']) for x in endpoints],new_wall_seconds=identities[1]['wall_seconds'],actual_train_input_hashes_identical=True),indent=2))
if __name__=='__main__':main()
