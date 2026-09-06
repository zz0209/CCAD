"""Source-only lexical candidate census from a frozen checkpoint and cached hook."""
import argparse,json,math,os,platform,re,sys,time,traceback
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from run_r011s1_raw_hook_asset import ROOT,entry,aggregate,write_json as write
from ccad.artifacts import sha256,validate_run_directory

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args();cfg=json.loads(args.config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False);start=time.perf_counter();inputs=[];checks={};env={};error=None;rows=[]
 write(run/'config.resolved.json',cfg);code=[]
 for rel in ['scripts/discover_checkpoint_source_contexts.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py']:
  p=ROOT/rel;q=run/'source_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
 write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
 write(run/'manifest.json',dict(schema_version='source.census.v1',run_id=cfg['run_id'],run_parent='R012',purpose=cfg['purpose'],milestone='C2-C3',evidence_level='source_only_candidate_preparation',started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(code),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='not estimated: source activation ranking only',threshold_source_split='frozen rule before source census',statistics_unit='discovery positions/documents; no target metrics',device='cuda:0',seeds=[cfg['source_seed']],resource_lease='gpu-0 resource_manager.run',resource_lease_reason='One source encoder, no new LM forward; bounded cached hook reads'))
 for name in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/name).touch()
 write(run/'status.json',dict(status='RUNNING'))
 def checked(path,digest=None):
  p=Path(path);p=p if p.is_absolute() else ROOT/p;item=entry(p,'existing source checkpoint or discovery asset','source_only_input')
  if digest:assert item['sha256']==digest,(path,digest,item['sha256'])
  inputs.append(item);return p
 try:
  import numpy as np,torch,transformers
  from sparsify import SparseCoder
  checked(args.config);checked('.aris/compute/local-r006b1-env-spec.json');torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
  cp=cfg['source_checkpoint'];checked(Path(cp['path'])/'sae.safetensors',cp['sha256']);raw=json.loads(checked(cfg['raw_manifest']).read_text());spec=next(x for x in raw['splits'] if x['split']=='discovery');hook=np.memmap(checked(spec['path'],spec['sha256']),mode='r',dtype='<f4',shape=tuple(spec['shape']))
  asset=json.loads(checked(cfg['asset_config']).read_text());tm=json.loads(checked(asset['token_manifest_path'],asset['token_manifest_sha256']).read_text());tmeta=tm['outputs']['discovery'];tokens=np.memmap(checked(ROOT/'runs'/asset['paired_corpus_run']/tmeta['path'],tmeta['sha256']),dtype='<u2',mode='r').reshape(-1,128);assert tokens.size==len(hook)
  docs=[json.loads(x) for x in checked(ROOT/'runs'/asset['paired_corpus_run']/'artifacts/documents.jsonl').read_text().splitlines()];training=json.loads(checked(cfg['training_documents']).read_text())['documents'];checks['train_disjoint']=all(not({x[k] for x in docs}&{x[k] for x in training}) for k in ['document_id','text_sha256']);assert checks['train_disjoint']
  selected=np.linspace(0,tokens.size-1,cfg['positions'],dtype=int);assert len(set(selected))==len(selected);sae=SparseCoder.load_from_disk(cp['path'],device='cuda:0').eval();ii=[];aa=[]
  with torch.no_grad():
   for offset in range(0,len(selected),512):
    enc=sae.encode(torch.tensor(np.array(hook[selected[offset:offset+512]]),device='cuda:0'));ii.append(enc.top_indices.cpu().numpy());aa.append(enc.top_acts.cpu().numpy())
  ii=np.concatenate(ii);aa=np.concatenate(aa);np.savez_compressed(run/'source_codes.npz',positions=selected,indices=ii,activations=aa);checks['finite']=bool(np.isfinite(aa).all());checks['source_only']=True
  tok=transformers.AutoTokenizer.from_pretrained(asset['model_local_dir'],local_files_only=True);hits=[[] for _ in range(cfg['num_latents'])]
  for j,pos in enumerate(selected):
   for atom,value in zip(ii[j],aa[j]):
    if value>0:hits[int(atom)].append((float(value),int(pos)))
  for atom,records in enumerate(hits):
   top=sorted(records,reverse=True)[:cfg['top_contexts']]
   if len(top)<cfg['minimum_hits']:continue
   token,count=Counter(int(tokens.flat[p]) for value,p in top).most_common(1)[0];seqs=len({p//128 for value,p in top})
   if seqs<cfg['minimum_sequences']:continue
   text=tok.decode([token]);stripped=text.strip();family='lexical' if re.fullmatch('[A-Za-z]+',stripped) else ('punctuation' if stripped and not re.search('[A-Za-z0-9]',stripped) else 'other');examples=[]
   for value,p in top[:8]:
    seq,pos=divmod(p,128);prefix=tokens[seq,:pos].tolist();eos=[k for k,t in enumerate(prefix) if t==0];begin=max(pos-24,eos[-1]+1 if eos else 0,0);examples.append(dict(sequence=seq,position=pos,activation=value,token=tok.decode([int(tokens[seq,pos])]),left=tok.decode(tokens[seq,begin:pos].tolist()),right=tok.decode(tokens[seq,pos+1:min(128,pos+17)].tolist())))
   rows.append(dict(atom=atom,frequency=len(records),dominant_token=text,dominant_token_id=token,lexical_family=family,concentration=count/len(top),score=count/len(top)*math.log(seqs),examples=examples))
  rows.sort(key=lambda x:(-x['score'],x['atom']));chosen=[];requests=[]
  for family in ['punctuation','lexical']:
   token_ids=[]
   for x in rows:
    if x['lexical_family']!=family or x['dominant_token_id'] in token_ids:continue
    token_ids.append(x['dominant_token_id'])
   for token_id in token_ids:
    options=[x for x in rows if x['dominant_token_id']==token_id and x['lexical_family']==family];requested=dict(family=family,token_id=token_id,token=options[0]['dominant_token'],available_atoms=len(options),selected=False);requests.append(requested)
    if len(options)<2:continue
    if sum(x['family']==family for x in chosen)>=cfg['token_families_per_class']:continue
    requested['selected']=True;atoms=[x['atom'] for x in options[:2]];decoder=sae.W_dec.detach().cpu().numpy()[atoms];gram=decoder@decoder.T;chosen.append(dict(family=family,token_id=token_id,token=options[0]['dominant_token'],atoms=atoms,source_decoder_gram=gram.tolist(),decoder_gram_eigenvalues=np.linalg.eigvalsh(gram).tolist(),members=options[:2]))
  write(run/'candidates.json',dict(source_seed=cfg['source_seed'],checkpoint=cp,positions=selected.tolist(),all_frequencies=[len(x) for x in hits],candidates=rows));write(run/'selected_queries.json',dict(queries=chosen,requests=requests,scope=cfg['scope'],target_outcomes_read=False,selection_rule=cfg['selection_rule']))
  checks['positions_complete']=len(selected)==cfg['positions'];checks['no_target_selection']=True;env=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,numpy=np.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),peak_vram_bytes=torch.cuda.max_memory_allocated())
 except Exception as exc:error=f'{type(exc).__name__}: {exc}';(run/'stderr.log').write_text(traceback.format_exc())
 (run/'metrics.raw.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));write(run/'inputs.json',dict(inputs=inputs));write(run/'environment.json',env);status='PASS' if error is None and checks and all(checks.values()) else 'FAIL';summary=dict(status=status,error=error,checks=checks,candidates=len(rows),base_model_forwards=0,wall_seconds=time.perf_counter()-start,metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),scope=cfg['scope']);write(run/'metrics.summary.json',summary);write(run/'stdout.log',summary);write(run/'status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat()));v=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=v.ok,errors=list(v.errors)));print(json.dumps(dict(summary=summary,contract_ok=v.ok)),flush=True);return 0 if status=='PASS' and v.ok else 1
if __name__=='__main__':raise SystemExit(main())
