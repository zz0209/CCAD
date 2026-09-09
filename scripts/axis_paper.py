"""Export complete source-axis experiments; plots consume only exported data.

The manifest lists completed runs and their finite conditions. It is not a
task-selection rule. Every task in each run is retained in the displayed data.
"""
from pathlib import Path
from collections import defaultdict
import json,csv,hashlib
import numpy as np

METHODS={
 'native_shared_axis':'Shared-axis native',
 'compiled_shared_axis':'Compiled shared-axis native',
 'compiled_functional_axis':'Output-fitted compiled native',
 'native_task_ridge_axis':'Task-ridge native',
 'native_calibrated_shared_axis':'Scalar-calibrated native',
 'native_residual_task_ridge_axis':'Residual-ridge native',
 'native_natural_ridge_axis':'Natural-ridge native',
 'native_cosine_axis':'Cosine native',
 'native_global_pw_axis':'Complete PW native',
 'native_semantic_ot_axis':'Feature SemanticOT native',
 'fixed_native_mask':'Fixed native mask',
 'reencode_shared_axis':'Actual re-encoding',
 'native_reencode_count':'Native at encoder count',
 'native_random_support':'Random-support native',
 'native_wrong_axis':'Wrong-axis native',
 'no_op':'No edit',
 'source':'Source decoded axis',
 'shared_axis_reader':'Unconstrained shared reader',
 'full_target':'Complete decoded difference',
 'raw_das':'Raw DAS-style',
 'raw_signed_distill':'Unrestricted two-direction distillation',
 'raw':'Complete raw difference'}
SELECTORS={'source_endpoint':'Source endpoint','base_linear':'Base linear',
 'anchored_base_linear':'Source value + base gradient','source_path':'Source path',
 'task_mse':'Task vector error','task_cosine':'Task vector cosine',
 'natural_mse':'Natural read error','cosine':'Decoder cosine','pw_mcc':'Support PW score',
 'semantic_ot_group':'Group SemanticOT score','finite_margin':'Direct smooth trials',
 'finite_budget':'Direct IIA trials'}

def mean(xs):return float(np.mean(list(xs)))
def tex(s):return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def csvwrite(path,rows):
 if not rows:return
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def export(root,out,read,manifest_path='paper/axis_runs.json'):
 manifest=Path(manifest_path)
 if not (root/manifest).exists():return None
 from summarize_group_selection import describe
 spec=read(manifest);conditions=[];inputs=[manifest.as_posix()];flat=[];costs=[];compiled_costs=[]
 for item in spec['runs']:
  run=root/item['path'];status=json.loads((run/'status.json').read_text());assert status['status']=='PASS',(run,status)
  result=describe(run);choices=json.loads((run/'selection_choices.json').read_text())['choices']
  replay=None
  if item.get('budget_replay'):
   replay=read(item['budget_replay']);inputs.append(item['budget_replay'])
   assert {r['query'] for r in replay['records']}=={c['query'] for c in choices}
  inputs += [str(Path(item['path'])/n) for n in ['metrics.summary.json','metrics.raw.jsonl','selection_choices.json','config.resolved.json']]
  for objective in result['config']['objectives']:
   label=item['labels'][objective];cs=[c for c in choices if c['objective']==objective]
   rr=[r for r in result['records'] if r['objective']==objective]
   bb=[b for b in (replay['records'] if replay else result['budget_records']) if b['objective']==objective]
   methods=sorted(set.intersection(*(set(c['held_summary']) for c in cs)))
   fixed={m:mean(c['held_summary'][m]['iia'] for c in cs) for m in methods}
   method_records=[r for r in result['all_method_records'] if r['objective']==objective]
   fixed_metrics={m:{k:mean(r[k] for r in method_records if r['method']==m and r[k] is not None) for k in ['iia','iia_flip','base_correct_fraction','donor_ce','log_odds_ratio','source_kl'] if any(r[k] is not None for r in method_records if r['method']==m)} for m in methods}
   metrics={}
   for s in SELECTORS:
    q=[r for r in rr if r['selector']==s]
    if q:metrics[s]={k:mean(r[k] for r in q if r[k] is not None) for k in ['top1_iia','tie_uniform_top1_iia','top3_uniform_iia','ranking_spearman','regret'] if any(r[k] is not None for r in q)}
   budget={}
   for b in bb:budget.setdefault(str(b['budget']),{}).setdefault(b['policy'],[]).append(b['held_iia'])
   budget={b:{p:mean(v) for p,v in policies.items()} for b,policies in budget.items()}
   for c in cs:
    for m in methods:
     flat.append(dict(condition=label,stage=spec['stage'],query=c['query'],task=c['task'],source_seed=c['source_seed'],target_seed=c['target_seed'],method=m,iia=c['held_summary'][m]['iia']))
    diagpath=run/(c['query']+'_axis_writer_diagnostics.json')
    if diagpath.exists():
     diag=json.loads(diagpath.read_text())
     inputs.append(str(diagpath.relative_to(root)))
     for m,d in diag.items():
      if m.startswith('compiled_'):
       kernel=d.get('application_kernel',{})
       compiled_costs.append(dict(condition=label,query=c['query'],method=m,compile_seconds=d.get('compile_seconds'),output_fit_seconds=d.get('fit',{}).get('wall_seconds'),kernel_milliseconds_per_row=kernel.get('milliseconds_per_row'),kernel_rows=kernel.get('rows'),native_replay_max_abs=d.get('saved_native_replay_max_abs'),actual_changed_min=min(d.get('actual_changed_counts',[0])),actual_changed_max=max(d.get('actual_changed_counts',[0])),distinct_union_members=d.get('distinct_union_members'),scope='Setup and warm batched application are separate; kernel excludes encoding and language-model execution.'))
      if 'wall_seconds' in d:
       count=sum(len(s.get('relative_errors',[])) for s in d.get('solver',[]))
       # Every writer executes all calibration + held task rows. Use the
       # retained matrix row count, rather than infer undocumented solver keys.
       with np.load(run/(c['query']+'_'+m+'_write.npz')) as z:count=len(z['row_ids']) if 'row_ids' in z else int(z['shape'][0])
       costs.append(dict(condition=label,query=c['query'],method=m,rows=count,wall_seconds=d['wall_seconds'],milliseconds_per_row=1000*d['wall_seconds']/count))
   conditions.append(dict(label=label,objective=objective,run=item['path'],queries=len(cs),tasks=result['config']['tasks'],fixed_iia=fixed,fixed_metrics=fixed_metrics,method_records=method_records,selector_metrics=metrics,budget_metrics=budget,records=rr,budget_records=bb,budget_scope=replay['scope'] if replay else 'Budget choices retained in the original run.'))
 # Edge-isolated executions share one displayed model/objective condition.
 # Aggregate their retained query rows; do not display seed edges as different
 # model conditions or treat them as independent statistical repetitions.
 grouped={}
 for c in conditions:grouped.setdefault(c['label'],[]).append(c)
 merged=[]
 for label,parts in grouped.items():
  if len(parts)==1:merged.append(parts[0]);continue
  assert all(c['tasks']==parts[0]['tasks'] and c['objective']==parts[0]['objective'] for c in parts)
  mr=[r for c in parts for r in c['method_records']];rr=[r for c in parts for r in c['records']];bb=[r for c in parts for r in c['budget_records']]
  methods=sorted(set.intersection(*(set(c['fixed_iia']) for c in parts)))
  fixed={m:mean(r['iia'] for r in mr if r['method']==m) for m in methods}
  fm={m:{k:mean(r[k] for r in mr if r['method']==m and r[k] is not None) for k in ['iia','iia_flip','base_correct_fraction','donor_ce','log_odds_ratio','source_kl'] if any(r[k] is not None for r in mr if r['method']==m)} for m in methods}
  metrics={}
  for s in SELECTORS:
   q=[r for r in rr if r['selector']==s]
   if q:metrics[s]={k:mean(r[k] for r in q if r[k] is not None) for k in ['top1_iia','tie_uniform_top1_iia','top3_uniform_iia','ranking_spearman','regret'] if any(r[k] is not None for r in q)}
  budget={}
  for b in bb:budget.setdefault(str(b['budget']),{}).setdefault(b['policy'],[]).append(b['held_iia'])
  merged.append(dict(label=label,objective=parts[0]['objective'],runs=[c['run'] for c in parts],queries=sum(c['queries'] for c in parts),tasks=parts[0]['tasks'],fixed_iia=fixed,fixed_metrics=fm,method_records=mr,selector_metrics=metrics,budget_metrics={b:{p:mean(v) for p,v in values.items()} for b,values in budget.items()},records=rr,budget_records=bb,budget_scope='Original choices retained across edge-isolated runs; these edges share SAE nodes.'))
 conditions=merged
 calibration=[];refusal=[]
 for c in conditions:
  for s in ['source_endpoint','base_linear','finite_margin']:
   rr=[r for r in c['records'] if r['selector']==s]
   for lo,hi in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
    q=[r for r in rr if lo<=r['predicted_confidence'] and (r['predicted_confidence']<hi or hi==1)]
    if q:calibration.append(dict(condition=c['label'],selector=s,lower=float(lo),upper=float(hi),queries=len(q),mean_preference=mean(r['predicted_confidence'] for r in q),selected_iia=mean(r['top1_iia'] for r in q)))
   for threshold in [0,.25,.5,.6,.7,.8,.9]:
    q=[r for r in rr if r['predicted_confidence']>=threshold]
    refusal.append(dict(condition=c['label'],selector=s,threshold=threshold,accepted_queries=len(q),total_queries=len(rr),coverage=len(q)/len(rr),selected_iia=mean(r['top1_iia'] for r in q) if q else None))
 training=[]
 for item in spec.get('training_runs',[]):
  run=root/item['path'];assert read(Path(item['path'])/'status.json')['status']=='PASS'
  inputs.extend(str(Path(item['path'])/n) for n in ['metrics.raw.jsonl','metrics.summary.json','config.resolved.json'])
  for line in (run/'metrics.raw.jsonl').read_text().splitlines():
   r=json.loads(line)
   if 'fve' in r:training.append(dict(cohort=item['label'],run=item['path'],**{k:r[k] for k in ['step','tokens','objective','seed','fve','ce_recovered','l0','alive','dead','decoder_norm_max_error']}))
 if training:
  csvwrite(out/'data/axis_material_by_seed_step.csv',training)
  lines=[]
  for item in spec['training_runs']:
   last=max(r['step'] for r in training if r['run']==item['path'])
   for objective in ['topk','matryoshka']:
    rr=[r for r in training if r['run']==item['path'] and r['step']==last and r['objective']==objective]
    def summary(k,scale=1,digits=2):
     v=[r[k]*scale for r in rr]
     return f'{mean(v):.{digits}f} [{min(v):.{digits}f}, {max(v):.{digits}f}]'
    lines.append(' & '.join([tex(item['label']),tex(objective),summary('fve',100),summary('ce_recovered',100),summary('l0'),summary('alive',digits=0)])+r' \\')
  (out/'tables/axis_material.tex').write_text('\n'.join(lines)+'\n')
 pilot=[]
 for item in spec.get('development_pilots',[]):
  run=root/item['path'];assert read(Path(item['path'])/'status.json')['status']=='PASS'
  result=describe(run)
  inputs.extend(str(Path(item['path'])/n) for n in ['metrics.raw.jsonl','selection_choices.json','config.resolved.json'])
  pilot.extend(dict(run=item['path'],stage=item['stage'],**r) for r in result['all_method_records'])
 if pilot:
  csvwrite(out/'data/axis_development_pilot.csv',pilot)
  lines=[r'Method & TopK & Matryoshka \\',r'\midrule']
  for m in ['source','shared_axis_reader','reencode_shared_axis','native_reencode_count','native_shared_axis','native_calibrated_shared_axis','native_residual_task_ridge_axis','compiled_shared_axis','compiled_functional_axis','raw_signed_distill','raw_das']:
   lines.append(tex(METHODS[m])+' & '+' & '.join(f"{100*mean(r['iia'] for r in pilot if r['objective']==o and r['method']==m):.2f}" for o in ['topk','matryoshka'])+r' \\')
  (out/'tables/axis_development_pilot.tex').write_text('\n'.join(lines)+'\n')
 data=dict(stage=spec['stage'],conditions=conditions,method_labels=METHODS,selector_labels=SELECTORS,rows=flat,writer_costs=costs,compiled_costs=compiled_costs,calibration=calibration,refusal=refusal,training=training,development_pilot=pilot,input_summary_paths=inputs,scope=spec['scope'])
 csvwrite(out/'data/axis_compilation_costs.csv',compiled_costs)
 (out/'data/axis_transfer.json').write_text(json.dumps(data,indent=2)+'\n')
 csvwrite(out/'data/axis_task_edge_results.csv',flat);csvwrite(out/'data/axis_writer_costs.csv',costs);csvwrite(out/'data/axis_choice_calibration.csv',calibration);csvwrite(out/'data/axis_refusal.csv',refusal)
 csvwrite(out/'data/axis_all_method_metrics.csv',[dict(condition=c['label'],**r) for c in conditions for r in c['method_records']])
 headers=' & '.join(tex(c['label']) for c in conditions)
 def table(path,order,values):
  lines=['Method & '+headers+r' \\',r'\midrule']
  for k,label in order.items():
   nums=[values(c,k) for c in conditions]
   if all(x is None for x in nums):continue
   lines.append(tex(label)+' & '+' & '.join('--' if x is None else f'{100*x:.2f}' for x in nums)+r' \\')
  (out/'tables'/path).write_text('\n'.join(lines)+'\n')
 table('axis_methods.tex',METHODS,lambda c,k:c['fixed_iia'].get(k))
 lines=['Method & '+headers+r' \\',r'\midrule']
 for m,label in METHODS.items():
  if all(m not in c['fixed_metrics'] for c in conditions):continue
  lines.append(tex(label)+' & '+' & '.join(f"{100*c['fixed_metrics'][m]['iia']:.2f} / {100*c['fixed_metrics'][m]['iia_flip']:.2f}" if m in c['fixed_metrics'] else '--' for c in conditions)+r' \\')
 (out/'tables/axis_methods_flip.tex').write_text('\n'.join(lines)+'\n')
 table('axis_selectors.tex',SELECTORS,lambda c,k:c['selector_metrics'].get(k,{}).get('top1_iia'))
 lines=[]
 for c in conditions:
  for s,a in c['selector_metrics'].items():
   lines.append(' & '.join([tex(c['label']),tex(SELECTORS[s]),f"{100*a['top1_iia']:.2f}",f"{100*a['tie_uniform_top1_iia']:.2f}",f"{100*a['top3_uniform_iia']:.2f}",f"{a.get('ranking_spearman',float('nan')):.3f}",f"{100*a['regret']:.2f}"])+r' \\')
 (out/'tables/axis_selector_details.tex').write_text('\n'.join(lines)+'\n')
 values={'AxisQueries':sum(c['queries'] for c in conditions)}
 for label,key in [('Native','native_task_ridge_axis'),('Reencode','reencode_shared_axis'),('Source','source'),('RawDAS','raw_das'),('Shared','native_shared_axis')]:values['Axis'+label]=100*mean(c['fixed_iia'][key] for c in conditions)
 for label,key in [('Endpoint','source_endpoint'),('Finite','finite_margin'),('Cosine','cosine')]:values['Axis'+label]=100*mean(c['selector_metrics'][key]['top1_iia'] for c in conditions)
 (out/'tables/axis_values.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+(str(v) if k=='AxisQueries' else f'{v:.2f}')+'}' for k,v in values.items())+'\n')
 return data

def task_label(task):
 words={'agr_gender':'Gender agreement','agr_refl_num':'Reflexive number','agr_sv_num':'Verb number',
 'filler_gap':'Filler gap','garden_mvrr':'MV/RR','garden_npz':'NP/Z','npi_any':'NPI any','npi_ever':'NPI ever',
 'gss_subord':'Subordination','cleft':'Cleft'}
 for prefix,label in words.items():
  if task.startswith(prefix):return label+task[len(prefix):].replace('_',' ').replace('subj-relc','S-RC').replace('obj-relc','O-RC').replace('v-trans','trans.').replace('hierarchy','hier.').replace('embed','depth')
 return task.replace('_',' ')

def plot(data,save):
 if not data:return
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 GREEN='#286956';PURPLE='#785481';INK='#262626';GREY='#a3a3a3'
 conditions=data['conditions'];tasks=conditions[0]['tasks'];assert all(c['tasks']==tasks for c in conditions)
 fig,axs=plt.subplots(1,len(conditions),figsize=(7,1.2+.205*len(tasks)),squeeze=False)
 fig.subplots_adjust(left=.24,right=.985,bottom=.10,top=.88,wspace=.12)
 for j,(ax,c) in enumerate(zip(axs[0],conditions)):
  rows=[r for r in data['rows'] if r['condition']==c['label']]
  for i,t in enumerate(tasks):
   get=lambda m:100*mean(r['iia'] for r in rows if r['task']==t and r['method']==m)
   a,b=get('reencode_shared_axis'),get('native_shared_axis')
   ax.plot([a,b],[i,i],color=GREEN if b>=a else PURPLE,lw=1.0)
   ax.plot(a,i,'o',ms=3,mec=GREY,mfc='white',mew=.75);ax.plot(b,i,'o',ms=3,color=GREEN)
   ax.plot(get('raw_das'),i,marker='|',ms=6,color=INK,mew=.8)
  ax.set(xlim=(-3,103),ylim=(len(tasks)-.4,-.8),yticks=range(len(tasks)),yticklabels=[task_label(t) for t in tasks] if j==0 else [],xticks=[0,50,100],xlabel='Interchange accuracy (%)')
  ax.set_title(c['label'],loc='left',fontsize=9);ax.grid(axis='x',color='#e7e7e7',lw=.45);ax.tick_params(axis='y',length=0,labelsize=7.6)
 handles=[Line2D([],[],color=GREEN,marker='o',ms=3,lw=.7,label='Shared-axis native'),Line2D([],[],color=GREY,marker='o',ms=3,mfc='white',lw=0,label='Actual re-encoding'),Line2D([],[],color=INK,marker='|',ms=6,lw=0,label='Raw DAS-style')]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.57,.98),ncol=3,frameon=False,fontsize=8)
 save(fig,'axis_all_tasks')
 fig,axs=plt.subplots(1,2,figsize=(7,3.1));fig.subplots_adjust(left=.08,right=.985,bottom=.33,top=.87,wspace=.29)
 colors={'source_screen':GREEN,'direct_balanced':INK,'direct_halving':PURPLE,'cosine_screen':GREY}
 labels={'source_screen':'Source screen','direct_balanced':'Balanced direct','direct_halving':'Direct halving','cosine_screen':'Cosine screen'}
 budgets=sorted(set.intersection(*(set(c['budget_metrics']) for c in conditions)),key=int)
 for off,(p,col) in enumerate(colors.items()):
  y=[100*mean(c['budget_metrics'][b][p] for c in conditions) for b in budgets]
  axs[0].plot(range(len(budgets)),y,color=col,marker=['o','s','^','D'][off],ms=3.5,lw=.9,label=labels[p])
 axs[0].set(xticks=range(len(budgets)),xticklabels=budgets,xlabel='Declared native-trial budget',ylabel='Selected IIA (%)');axs[0].grid(axis='y',color='#eeeeee',lw=.5);axs[0].set_title('(a) Candidate choice at fixed accounting budget',loc='left',fontsize=8)
 styles=[('source_endpoint','Source endpoint',GREEN,'o'),('finite_margin','Direct trials',INK,'s'),('cosine','Cosine',GREY,'^'),('natural_mse','Natural error',PURPLE,'D')]
 for j,(s,label,col,marker) in enumerate(styles):
  for i,c in enumerate(conditions):
   m=c['selector_metrics'][s];y=i+(j-1.5)*.16
   axs[1].plot([100*m['top1_iia'],100*m['tie_uniform_top1_iia']],[y,y],color=col,lw=.8)
   axs[1].plot(100*m['top1_iia'],y,marker=marker,color=col,ms=3.5)
   axs[1].plot(100*m['tie_uniform_top1_iia'],y,marker=marker,mfc='white',mec=col,mew=.8,ms=3.5)
 axs[1].set(yticks=range(len(conditions)),yticklabels=[c['label'] for c in conditions],ylim=(len(conditions)-.4,-.6),xlabel='Selected IIA (%)');axs[1].tick_params(axis='y',labelsize=7);axs[1].grid(axis='x',color='#eeeeee',lw=.5);axs[1].set_title('(b) Filled: fixed ties; open: tie average',loc='left',fontsize=8)
 axs[0].legend(frameon=False,fontsize=7,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.27))
 axs[1].legend(handles=[Line2D([],[],color=c,marker=m,lw=0,ms=3,label=l) for _,l,c,m in styles],frameon=False,fontsize=7,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.27))
 save(fig,'axis_selection')
 fig,axs=plt.subplots(2,len(conditions),figsize=(7,4.4),squeeze=False);fig.subplots_adjust(left=.075,right=.985,bottom=.13,top=.9,wspace=.33,hspace=.4)
 for j,c in enumerate(conditions):
  for s,label,col,marker in [('source_endpoint','Source endpoint',GREEN,'o'),('base_linear','Base linear',PURPLE,'D'),('finite_margin','Direct trials',INK,'s')]:
   rows=[r for r in data['calibration'] if r['condition']==c['label'] and r['selector']==s]
   axs[0,j].plot([r['mean_preference'] for r in rows],[r['selected_iia'] for r in rows],color=col,marker=marker,ms=3,lw=.8,label=label)
   rows=[r for r in data['refusal'] if r['condition']==c['label'] and r['selector']==s and r['accepted_queries']]
   axs[1,j].plot([r['coverage'] for r in rows],[r['selected_iia'] for r in rows],color=col,marker=marker,ms=3,lw=.8)
  axs[0,j].plot([0,1],[0,1],lw=.5,ls='--',color=GREY);axs[0,j].set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1],xlabel='Predicted answer preference',ylabel='Selected IIA' if j==0 else '');axs[0,j].set_title(c['label'],loc='left',fontsize=8)
  axs[1,j].set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1],xlabel='Accepted query fraction',ylabel='Accepted-query IIA' if j==0 else '')
  for ax in axs[:,j]:ax.grid(color='#eeeeee',lw=.4)
 fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.54,1),ncol=3,frameon=False,fontsize=8)
 save(fig,'axis_calibration')
 if all('compiled_functional_axis' in c['fixed_iia'] for c in conditions):
  plot_compiled(data,save)


def plot_compiled(data,save):
 """Show the fixed compiled contrast, retaining shared edges and every task."""
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
 GREEN='#286956';PURPLE='#785481';INK='#262626';GREY='#777777'
 conditions=data['conditions'];methods=[('native_shared_axis','Dynamic native',GREEN),('raw_signed_distill','Unrestricted two-direction',PURPLE)]
 fig,axs=plt.subplots(1,2,figsize=(7,2.6),sharey=True)
 fig.subplots_adjust(left=.20,right=.985,bottom=.27,top=.77,wspace=.18)
 for ax,(baseline,label,col) in zip(axs,methods):
  for i,c in enumerate(conditions):
   rows=[r for r in data['rows'] if r['condition']==c['label']]
   edges=sorted({(r['source_seed'],r['target_seed']) for r in rows});values=[]
   for edge in edges:
    vals={m:mean(r['iia'] for r in rows if r['method']==m and (r['source_seed'],r['target_seed'])==edge) for m in ['compiled_functional_axis',baseline]}
    values.append(100*(vals['compiled_functional_axis']-vals[baseline]))
   # Offsets identify distinct edges; neither their extent nor their count is
   # a confidence interval or a count of independent model replications.
   ax.scatter(values,i+np.linspace(-.11,.11,len(values)),s=15,facecolors='white',edgecolors=col,linewidths=.7,zorder=3)
   ax.scatter([mean(values)],[i],s=27,marker='D',color=col,zorder=4)
  ax.axvline(0,color=GREY,lw=.7,ls='--');ax.grid(axis='x',color='#eeeeee',lw=.45)
  ax.set(yticks=range(len(conditions)),yticklabels=[c['label'] for c in conditions],ylim=(len(conditions)-.55,-.45),xlabel='Compiled native minus comparator (points)')
  ax.set_title(label,loc='left',fontsize=9);ax.tick_params(axis='y',length=0,labelsize=8)
 lo=min(ax.get_xlim()[0] for ax in axs);hi=max(ax.get_xlim()[1] for ax in axs)
 for ax in axs:ax.set_xlim(min(-1,lo),max(1,hi))
 fig.legend(handles=[Line2D([],[],marker='o',mfc='white',mec=INK,lw=0,ms=4,label='Each shared-seed edge'),Line2D([],[],marker='D',color=INK,lw=0,ms=4,label='Mean over all tasks and edges')],frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.58,.99),fontsize=8)
 save(fig,'axis_compiled_edges')
 tasks=conditions[0]['tasks'];values=[]
 for task in tasks:
  line=[]
  for c in conditions:
   rows=[r for r in data['rows'] if r['condition']==c['label'] and r['task']==task]
   compiled=mean(r['iia'] for r in rows if r['method']=='compiled_functional_axis')
   line.extend(100*(compiled-mean(r['iia'] for r in rows if r['method']==m)) for m,_,_ in methods)
  values.append(line)
 values=np.asarray(values);bound=max(5,5*np.ceil(np.max(np.abs(values))/5))
 cmap=LinearSegmentedColormap.from_list('native_difference',[PURPLE,'#ffffff',GREEN])
 fig,axs=plt.subplots(1,len(conditions),figsize=(7,max(3.4,1.25+.19*len(tasks))),sharey=True,squeeze=False)
 fig.subplots_adjust(left=.235,right=.915,bottom=.21 if len(tasks)<8 else .10,top=.83 if len(tasks)<8 else .91,wspace=.25)
 for j,(ax,c) in enumerate(zip(axs[0],conditions)):
  im=ax.imshow(values[:,2*j:2*j+2],cmap=cmap,norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),aspect='auto',interpolation='nearest')
  ax.set(xticks=[0,1],xticklabels=['Dynamic\nnative','Unrestricted\ntwo-direction'],yticks=range(len(tasks)),yticklabels=[task_label(t) for t in tasks])
  ax.set_title(c['label'],loc='left',fontsize=8.5);ax.tick_params(length=0,labelsize=7.4)
  for i in range(len(tasks)):
   for k in range(2):
    value=values[i,2*j+k];ax.text(k,i,f'{value:+.1f}',ha='center',va='center',fontsize=7.1,color='white' if abs(value)>.58*bound else INK)
  for spine in ax.spines.values():spine.set_visible(False)
 cax=fig.add_axes([.939,.26,.012,.46]);cb=fig.colorbar(im,cax=cax);cb.ax.tick_params(labelsize=7,length=2)
 fig.text(.24,.972,'Compiled native minus comparator: mean IIA difference (points)',ha='left',va='top',fontsize=9)
 save(fig,'axis_compiled_task_contrasts')
