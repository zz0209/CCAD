"""Summarize the completed frozen suite by model/objective and inference unit.

This reads retained outcomes only. It cannot change a candidate, selector or
run. Primary summaries keep the three declared model/objective conditions
separate. Reserved and developed task partitions remain explicitly labeled.
"""
from pathlib import Path
from collections import defaultdict
import argparse,json,hashlib,csv,time
from summarize_selection_dependence import load,analyze

def main():
 p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--draws',type=int,default=2000);a=p.parse_args();root=Path(__file__).resolve().parents[1];timer=time.perf_counter()
 q=json.loads(a.queue.read_text());freeze=json.loads((root/q['freeze_manifest']).read_text());stage1=json.loads((root/freeze['stage1_path']).read_text());partition=stage1['task_partition'];groups=defaultdict(list);inputs=[]
 for item in q['configs']:
  run=root/'runs'/item['run_id'];assert json.loads((run/'status.json').read_text())['status']=='PASS',run
  cfg=json.loads((run/'config.resolved.json').read_text());assert cfg['evaluation_split']=='test'
  result=load(run);assert len(result)==len(cfg['tasks'])*len(cfg['objectives'])*len(cfg['seed_pairs'])
  model='GPT2Medium' if 'gpt2' in item['run_id'] else 'Pythia1B'
  for row in result:groups[(model,row['objective'])].append(row)
  for name in ['config.resolved.json','metrics.raw.jsonl','selection_choices.json','panel.json']:
   file=run/name;inputs.append(dict(path=str(file.relative_to(root)),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
 a.output.mkdir(parents=True,exist_ok=True);summaries=[];comparisons=[]
 for (model,objective),rows in groups.items():
  assert len(rows)==145,(model,objective,len(rows));nodes={r[s] for r in rows for s in ['source_seed','target_seed']};assert len(nodes)==5
  for split,tasks in [('all29',partition['development_tasks']+partition['reserved_tasks']),('reserved6',partition['reserved_tasks']),('developed23',partition['development_tasks'])]:
   subset=[r for r in rows if r['task'] in tasks];result=analyze(subset,a.draws);result.update(model=model,objective=objective,task_partition=split)
   name=f'{model}_{objective}_{split}';(a.output/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
   summaries.append(dict(model=model,objective=objective,partition=split,queries=len(subset),path=name+'.json',macro_iia=result['macro_iia']))
   for r in result['comparisons']:
    loo=[v['difference'] for v in r['incident_seed_leave_out'] if v['difference'] is not None]
    comparisons.append(dict(model=model,objective=objective,partition=split,left=r['left'],right=r['right'],difference=r['difference'],node_task_frame_low=r['seed_task_frame_interval'][0],node_task_frame_high=r['seed_task_frame_interval'][1],node_frame_fixed_task_low=r['seed_frame_fixed_task_interval'][0],node_frame_fixed_task_high=r['seed_frame_fixed_task_interval'][1],incident_seed_min=min(loo),incident_seed_max=max(loo)))
   print(json.dumps(dict(condition=name,queries=len(subset),compiled=result['macro_iia']['fixed:compiled_functional_axis'],shared=result['macro_iia']['fixed:native_shared_axis'],raw_signed=result['macro_iia']['fixed:raw_signed_distill'],wall_seconds=time.perf_counter()-timer)),flush=True)
 with (a.output/'comparisons.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
 (a.output/'query_frame_values.json').write_text(json.dumps([r for rows in groups.values() for r in rows])+'\n')
 result=dict(conditions=summaries,comparisons=comparisons,input_files=inputs,wall_seconds=time.perf_counter()-timer,draws=a.draws,analysis_source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['scripts/summarize_frozen_axis.py','scripts/summarize_selection_dependence.py']},scope='Primary all29 summaries are separate by fixed model/objective. Five shared seed nodes, tasks and lexical frames receive crossed bootstrap weights; fixed-task sensitivity and incident-node deletion also shown. The small observed cycle is not a model/corpus population confidence claim. Lexical frames mask alternating regions and group the remaining exact context; they are not a universal named-entity or shared-token clustering. Reserved6 and developed23 are prespecified subsets; neither replaces all29.')
 (a.output/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
