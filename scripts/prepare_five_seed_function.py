"""Freeze fitted maps, then bind outcome-blind natural document prefixes."""
import argparse,datetime,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/five_seed_function_20260906'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['freeze','bind']);args=ap.parse_args();r=ROOT;index=r/'configs/f4_five_seed_function_v1.json';f=json.loads(index.read_text())
 if args.phase=='freeze':
  maps=[]
  for cell in f['cells']:
   run=r/'runs'/cell['run'];assert json.loads((run/'status.json').read_text())['status']=='PASS';assert json.loads((run/'metrics.summary.json').read_text())['forwards']==0
   maps.append(dict(query=cell['query'],seed=cell['seed'],step=cell['step'],path=str(run),config_sha256=sha(run/'config.resolved.json'),coefficients_sha256=sha(run/'coefficients.npz')))
  now=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ');write(OUT/'maps_freeze.json',dict(written_at_utc=now,maps=maps,design_sha256=sha(OUT/'design_freeze.json')))
  c=json.loads((r/'configs/f4_training_gain_corpus_v1.json').read_text());c.update(run_id='F4_five_function_corpus_v1_20260906',purpose='Fresh natural documents for frozen four-query controlled five-seed functional panel',selection_salt='ccad-five-function-corpus-20260906-v1',scope_limit=f['scope'],budget=f['budget']);seen={x['path'] for x in c['additional_exclusions']}
  for p in sorted(r.glob('runs/*/artifacts/documents.jsonl')):
   name=p.relative_to(r).as_posix()
   if name not in seen:c['additional_exclusions'].append(dict(path=name,sha256=sha(p)));seen.add(name)
  c['frozen_scope']=dict(queries=f['queries'],minimum_prefix_tokens=10,maximum_recipients=16,selection_salt='ccad-five-function-positions-20260906-v1',selection_rule=f['natural_selection'],output=OUT.relative_to(r).as_posix());write(r/'configs/f4_five_function_corpus_v1.json',c)
  with (r/'master_log.md').open('a',encoding='utf-8') as log:log.write(f'\n## {now} — 四组32映射全冻结，新文档准备\n写入时间（written_at_utc）：{now}。32fit-only均PASS/0LMforward，maps_freeze.json SHA {sha(OUT/"maps_freeze.json")}，全部方法/源与target早晚权重固定后才采新语料；排除{len(c["additional_exclusions"])}份既存文档清单。原输入回放64行八字段精确。新自然请求及九图规则不变。\n')
  print(json.dumps(dict(maps=len(maps),written_at=now,exclusion_files=len(c['additional_exclusions']))))
 else:
  import numpy as np
  from transformers import AutoTokenizer
  c=json.loads((r/'configs/f4_five_function_corpus_v1.json').read_text());rule=c['frozen_scope'];corpus=r/'runs'/c['run_id'];assert json.loads((corpus/'status.json').read_text())['status']=='PASS';ts=np.fromfile(corpus/'artifacts/calibration.uint16.bin',dtype='<u2').reshape(-1,128);seq=json.loads((corpus/'artifacts/sequence_records.json').read_text())['sequences'];tok=AutoTokenizer.from_pretrained(c['tokenizer_local_dir'],local_files_only=True);selected={};all_candidates={};counts={};used=set()
  for query in f['queries']:
   qi=query['query'];candidates=[];stats=dict(multidocument_sequences=0,short_or_eos=0,token_positions=0)
   assert tok.encode(query['token'],add_special_tokens=False)==[query['token_id']]
   for rec in seq:
    if len(rec['document_ids'])!=1:stats['multidocument_sequences']+=1;continue
    si=rec['sequence_index'];doc=rec['document_ids'][0]
    for position in np.flatnonzero(ts[si]==query['token_id']):
     pos=int(position);stats['token_positions']+=1;ids=ts[si,:pos+1].tolist()
     if len(ids)<rule['minimum_prefix_tokens'] or tok.eos_token_id in ids:stats['short_or_eos']+=1;continue
     key=hashlib.sha256(f"{rule['selection_salt']}\0{qi}\0{doc}\0{si}\0{pos}".encode()).hexdigest();candidates.append(dict(document_id=doc,sequence_index=si,token_index=pos,query=qi,selection_hash=key,token_ids=ids,text=tok.decode(ids),observed_next_token_id=int(ts[si,pos+1]) if pos<127 else None))
   chosen=[]
   for x in sorted(candidates,key=lambda x:x['selection_hash']):
    if x['document_id'] in used:continue
    used.add(x['document_id']);chosen.append(x)
    if len(chosen)==rule['maximum_recipients']:break
   unpaired=chosen[len(chosen)//2*2:];chosen=chosen[:len(chosen)//2*2]
   for j,x in enumerate(chosen):x.update(pair=j//2,role='left' if j%2==0 else 'right',verb='not_applicable')
   selected[qi]=chosen;all_candidates[qi]=candidates;stats.update(candidates=len(candidates),distinct_candidate_documents=len({x['document_id'] for x in candidates}),selected=len(chosen),unpaired=unpaired,missing=rule['maximum_recipients']-len(chosen));counts[qi]=stats;write(OUT/f'inputs_q{qi}.json',dict(cases=chosen,scope=f['scope']))
  write(OUT/'selection.json',dict(counts=counts,candidates=all_candidates,selected=selected,globally_unique_documents=len({x['document_id'] for items in selected.values() for x in items}),selection_rule=rule,inputs=[dict(path=str(p),sha256=sha(p)) for p in [corpus/'artifacts/documents.jsonl',corpus/'artifacts/calibration.uint16.bin',r/'configs/f4_five_function_corpus_v1.json',Path(__file__)]],forwards=0));maps=json.loads((OUT/'maps_freeze.json').read_text())['maps']
  for cell,mp in zip(f['cells'],maps):
   parent=r/'runs'/cell['run'];assert sha(parent/'config.resolved.json')==mp['config_sha256'];assert sha(parent/'coefficients.npz')==mp['coefficients_sha256'];cfg=json.loads((parent/'config.resolved.json').read_text());inp=OUT/f'inputs_q{cell["query"]}.json';run=f'F4_five_apply_q{cell["query"]}_s{cell["seed"]}_step{cell["step"]}_v1_20260906';cfg.update(probability_storage='observed_and_probe_tokens',probe_token_ids=tok.encode('t',add_special_tokens=False),run_id=run,fit_only=False,use_frozen_target_checkpoint=True,frozen_map={k:mp[k] for k in ['path','config_sha256','coefficients_sha256']},evaluation_inputs=inp.relative_to(r).as_posix(),evaluation_inputs_sha256=sha(inp),confirmation_inputs=[dict(path=str(p),sha256=sha(p)) for p in [OUT/'maps_freeze.json',OUT/'design_freeze.json',OUT/'selection.json']],purpose='Frozen map application to new natural prefixes, all fixed queries and targets retained',budget=f['budget'],scope=f['scope']);dest=f'configs/f4_five_apply_q{cell["query"]}_s{cell["seed"]}_step{cell["step"]}_v1.json';write(r/dest,cfg);cell.update(apply_run=run,apply_config=dest)
  write(index,f);print(json.dumps(counts))
if __name__=='__main__':main()
