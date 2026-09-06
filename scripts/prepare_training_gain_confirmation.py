"""Freeze existing early/late maps before new corpus access, then bind inputs."""
import argparse, datetime, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/training_gain_confirmation_20260906'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','bind']);a=ap.parse_args();r=ROOT
 if a.stage=='freeze':
  OUT.mkdir(exist_ok=True);assert not (OUT/'freeze.json').exists()
  now=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ');cells=[]
  arc=r/'archive/research_workflow_20260906'/('training_gain_'+now.replace('-','').replace(':',''));arc.mkdir();arch=[]
  for name in ['EXPERIMENT_TRACKER.md','EXPERIMENT_PLAN.md','PAPER_SNAPSHOT_20260906.md','REFERENCE_REGISTRY.md']:
   p=r/name;(arc/name).write_bytes(p.read_bytes());arch.append(dict(path=name,sha256=sha(p)))
  write(arc/'manifest.json',arch)
  for seed in [1,2]:
   for step in [256,8192]:
    p=r/'runs'/f'F4_function_curve_s{seed}_step{step}_v1_20260906';c=json.loads((p/'config.resolved.json').read_text());cp=c['target_checkpoint'];assert sha(Path(cp['path'])/'sae.safetensors')==cp['sha256']
    cells.append(dict(seed=seed,step=step,parent=p.name,run=f'F4_training_gain_s{seed}_step{step}_v1_20260906',config=f'configs/f4_training_gain_s{seed}_step{step}_v1.json',reuse=False,frozen_map=dict(path=str(p),config_sha256=sha(p/'config.resolved.json'),coefficients_sha256=sha(p/'coefficients.npz')),target_checkpoint=cp))
  old=json.loads((r/'configs/f4_natural_function_curve_v1.json').read_text())
  f=dict(written_at_utc=now,cells=cells,original_methods=old['original_methods'],primary='Per target shared16: late8192 minus early256 of maximum across four operators of median absolute KL(source||candidate) over all selected recipients. Hypothesis: negative in both targets.',secondary='Original normalized KL with denominator>1e-12 missing rule; absolute observed-next-token NLL response error; all eight maps and all four operators.',uncertainty='2000 bootstrap resamples of entire reciprocal document pairs, RNG20260906, max-operator median recomputed per resample. Conditional on fixed source3 and two dependent target trajectories; no seed-population inference.',scope='New natural documents, previously frozen maps and lexical applicability. Source atoms1850/2897, four operators and dose0.1 unchanged. No target-based selection; retain weak/reverse/zero cases and all exclusions. Confirms limited early-late absolute functional error prediction, not monotonicity or semantics.',budget='At most3556LMforwards,10min16GB; three bounded corpus rowgroups~25-100MB; no new training or fitting.',archive=str(arc))
  c=json.loads((r/'configs/f4_fisher_refit_corpus_v1.json').read_text());c.update(run_id='F4_training_gain_corpus_v1_20260906',purpose='Fresh documents for frozen early-late compact function prediction',selection_salt='ccad-training-gain-natural-20260906-v1',budget=f['budget'],scope_limit=f['scope']);seen={x['path'] for x in c['additional_exclusions']}
  for p in sorted(r.glob('runs/*/artifacts/documents.jsonl')):
   rel=p.relative_to(r).as_posix()
   if rel not in seen:c['additional_exclusions'].append(dict(path=rel,sha256=sha(p)));seen.add(rel)
  c['frozen_scope'].update(target_seeds=[1,2],selection_salt='ccad-training-gain-positions-20260906-v1',output=OUT.relative_to(r).as_posix(),primary=f['primary'],uncertainty=f['uncertainty'])
  write(r/'configs/f4_training_gain_corpus_v1.json',c);f['corpus_config_sha256']=sha(r/'configs/f4_training_gain_corpus_v1.json');write(OUT/'freeze.json',f)
  t=r/'EXPERIMENT_TRACKER.md';s=t.read_text(encoding='utf-8');s=s.replace('当前GPU/CPU计算已结束、租约释放，automation ACTIVE，由本对话接续。','当前独立确认已冻结四父映射/早晚权重、absoluteKL主要终点及原八对照，配置configs/f4_training_gain_corpus_v1.json，冻结artifacts/training_gain_confirmation_20260906/freeze.json；下一新文档采样后四cell顺序推理，automation ACTIVE。');t.write_text(s,encoding='utf-8')
  with (r/'master_log.md').open('a',encoding='utf-8') as log:log.write(f'\n## {now} — 紧凑关系训练收益新文档确认启动\n写入时间（written_at_utc）：{now}。冻结四父coeff/config/targetSAE SHA、早256晚8192与source3；新主终点为shared16最差operator中位absoluteKL晚减早，两target均预期下降。旧开发主要relativeKL混合结论不变。原八对照、相对KL/NLL及所有弱例保留；2000整pair bootstrap不当seed总体推断。先冻结再新语料，排除{len(c["additional_exclusions"])}份既存document清单；预算3556forward/10分钟16GB，无新训练/refit。冻结SHA {sha(OUT/"freeze.json")}；改前四文件逐字归档{arc.relative_to(r).as_posix()}。\n')
  print(json.dumps(dict(frozen_at=now,freeze_sha256=sha(OUT/'freeze.json'),exclusion_files=len(c['additional_exclusions']))))
 else:
  import numpy as np
  f=json.loads((OUT/'freeze.json').read_text());assert sha(r/'configs/f4_training_gain_corpus_v1.json')==f['corpus_config_sha256'];corpus=r/'runs/F4_training_gain_corpus_v1_20260906';assert json.loads((corpus/'status.json').read_text())['status']=='PASS'
  ts=np.fromfile(corpus/'artifacts/calibration.uint16.bin',dtype='<u2').reshape(-1,128);prep=json.loads((OUT/'prepared_inputs.json').read_text())
  for c in prep['cases']:
   si,pos=c['sequence_index'],c['token_index'];assert c['token_ids']==ts[si,:pos+1].tolist();c['observed_next_token_id']=int(ts[si,pos+1]) if pos<127 else None
  dest=OUT/'prepared_inputs_with_next.json';write(dest,prep)
  for cell in f['cells']:
   p=r/'runs'/cell['parent'];assert sha(p/'config.resolved.json')==cell['frozen_map']['config_sha256'];assert sha(p/'coefficients.npz')==cell['frozen_map']['coefficients_sha256'];c=json.loads((p/'config.resolved.json').read_text());c.update(run_id=cell['run'],purpose='Frozen early-late function prediction on new natural documents',frozen_map=cell['frozen_map'],use_frozen_target_checkpoint=True,evaluation_inputs=dest.relative_to(r).as_posix(),evaluation_inputs_sha256=sha(dest),confirmation_inputs=[dict(path=str(p),sha256=sha(p)) for p in [OUT/'freeze.json',OUT/'selection.json',corpus/'artifacts/documents.jsonl']],evidence_level='fresh_document_frozen_early_late_confirmation',scope=f['scope'],budget_seconds=600,budget=f['budget']);write(r/cell['config'],c)
  write(r/'configs/f4_training_gain_confirmation_v1.json',f);print(json.dumps(dict(cases=len(prep['cases']),with_next=sum(c['observed_next_token_id'] is not None for c in prep['cases']),inputs_sha256=sha(dest))))
if __name__=='__main__':main()
