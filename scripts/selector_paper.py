"""Paper exports for actual source-conditioned candidate selection experiments."""
from pathlib import Path
import json,csv,hashlib
import numpy as np

BASE=Path('artifacts/final_three_research_20260909')
LABELS={'cosine':'Decoder cosine','pw_mcc':'Support PW-MCC','semantic_ot_group':'Group SemanticOT','natural_mse':'Natural regression error','base_linear':'Base linear','anchored_base_linear':'Source value + base gradient','source_endpoint':'Source-endpoint linear','source_path':'Source-path cubic','finite_budget':'Finite trials: IIA','finite_margin':'Finite trials: smooth margin'}
ORDER=list(LABELS)
def export(root,out,read):
 p=BASE/'r19_selection_analysis/selection_summary.json'
 if not (root/p).exists():return None
 data=read(p);train=Path('runs/FINAL3_R19_gpt2_dual_objective_five_seed_16m_v1_20260909');ts=read(train/'metrics.summary.json');tc=read(train/'config.resolved.json')
 tr=[json.loads(l) for l in (root/train/'metrics.raw.jsonl').read_text().splitlines()]
 for r in tr:r.pop('sequence_ce',None)
 data.update(training=tr,training_summary=ts,training_config=tc,input_summary_paths=[p.as_posix(),(train/'metrics.summary.json').as_posix(),(train/'config.resolved.json').as_posix()])
 main=data['runs'][-1];m=main['metrics']
 lines=[]
 for s in ORDER:
  if s in m:
   a=m[s];lines.append(f"{LABELS[s]} & {100*a['top1_iia']:.2f} & {100*a['top3_uniform_iia']:.2f} & {a.get('ranking_spearman',float('nan')):.3f} & {100*a['regret']:.2f} \\")
 (out/'tables/selector_main.tex').write_text('\n'.join(l+'\\' for l in lines)+'\n')
 vals={'SelectorEndpoint':100*m['source_endpoint']['top1_iia'],'SelectorFiniteMargin':100*m['finite_margin']['top1_iia'],'SelectorCosine':100*m['cosine']['top1_iia'],'SelectorAnchoredBase':100*m['anchored_base_linear']['top1_iia'],'SelectorRaw':100*main['raw_iia'],'SelectorSource':100*main['source_iia'],'SelectorFull':100*main['full_target_iia']}
 (out/'tables/selector_values.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+f'{v:.2f}'+'}' for k,v in vals.items())+'\n')
 lines=[]
 for o in ['topk','matryoshka']:
  rr=[r for r in tr if r['objective']==o and r['step']==tc['steps']]
  for r in rr:lines.append(f"{o.title()} & {r['seed']} & {100*r['fve']:.2f} & {100*r['ce_recovered']:.2f} & {r['l0']:.2f} & {r['alive']} & {r['decoder_norm_max_error']:.1e} \\")
 (out/'tables/selector_quality.tex').write_text('\n'.join(l+'\\' for l in lines)+'\n')
 rr=main['records'];lines=[]
 for o in ['topk','matryoshka']:
  for task in main['config']['tasks']:
   lines.append('\\multicolumn{5}{l}{'+o.title()+': '+task.replace('_',r'\_')+r'} \\')
   for s in ORDER:
    selected=[r for r in rr if r['objective']==o and r['task']==task and r['selector']==s]
    lines.append(f"{LABELS[s]} & {100*np.mean([r['top1_iia'] for r in selected]):.2f} & {100*min(r['top1_iia'] for r in selected):.2f} & {100*max(r['top1_iia'] for r in selected):.2f} & {np.mean([r['ranking_spearman'] for r in selected if r['ranking_spearman'] is not None]):.3f} \\")
 (out/'tables/selector_strata.tex').write_text('\n'.join(l if l.endswith('\\\\') else l+'\\' for l in lines)+'\n')
 (out/'data/group_selection.json').write_text(json.dumps(data,indent=2)+'\n')
 with (out/'data/group_selection_by_query.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 return data

def plot(data,save):
 if not data:return
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 GREEN='#286956';PURPLE='#785481';INK='#262626';GREY='#929292'
 main=data['runs'][-1];records=main['records'];styles=[('source_endpoint','Source endpoint',GREEN,'o'),('anchored_base_linear','Anchored base',INK,'s'),('finite_margin','Finite trials',PURPLE,'D'),('cosine','Cosine',GREY,'^')]
 fig,axs=plt.subplots(1,2,figsize=(7,2.8));fig.subplots_adjust(left=.09,right=.98,bottom=.20,top=.75,wspace=.28)
 for ax,o in zip(axs,['topk','matryoshka']):
  for j,(s,label,color,marker) in enumerate(styles):
   for i,task in enumerate(main['config']['tasks']):
    x=np.array([r['top1_iia']*100 for r in records if r['objective']==o and r['task']==task and r['selector']==s]);y=i+(j-1.5)*.16
    ax.plot([x.min(),x.max()],[y,y],color=color,lw=.7,alpha=.7);ax.scatter(x,np.full_like(x,y),s=12,marker=marker,facecolor='white',edgecolor=color,linewidth=.6);ax.plot(x.mean(),y,marker='|',ms=9,color=color,mew=1.5)
  ax.set(xlim=(0,80),ylim=(2.6,-.6),yticks=range(3),yticklabels=['Gender','NPI','Filler-gap'] if o=='topk' else [],xlabel='Selected native intervention accuracy (%)');ax.set_title(o.title(),loc='left');ax.grid(axis='x',color='#eeeeee',lw=.5)
 fig.legend(handles=[Line2D([],[],color=c,marker=m,lw=.8,label=l,markersize=4,markerfacecolor='white') for _,l,c,m in styles],loc='upper center',bbox_to_anchor=(.53,.99),ncol=4,frameon=False,fontsize=8)
 save(fig,'selector_choices')
 fig,axs=plt.subplots(1,3,figsize=(7,2.35));fig.subplots_adjust(left=.075,right=.985,bottom=.23,top=.85,wspace=.40)
 tr=data['training']
 for ax,key,label,scale in zip(axs,['ce_recovered','l0','alive'],['CE recovery (%)',r'Mean $L_0$','Active validation members'],[100,1,1]):
  for o,c,m in [('topk',INK,'s'),('matryoshka',GREEN,'o')]:
   steps=sorted(set(r['step'] for r in tr));x=[];y=[];lo=[];hi=[]
   for step in steps:
    rr=[r for r in tr if r['step']==step and r['objective']==o];x.append(rr[0]['tokens']/1e6);a=np.array([r[key]*scale for r in rr]);y.append(a.mean());lo.append(a.min());hi.append(a.max())
   ax.plot(x,y,color=c,marker=m,ms=3,lw=.85,label=o.title());ax.fill_between(x,lo,hi,color=c,alpha=.14,lw=0)
  ax.set(xscale='log',xlabel='Natural training tokens (M)',ylabel=label,xticks=[.26,1.05,4.19,16.78],xticklabels=['.26','1','4','16']);ax.grid(axis='y',color='#eeeeee',lw=.5)
 axs[0].legend(frameon=False,fontsize=7,loc='lower right');save(fig,'selector_material')
 fig,axs=plt.subplots(1,2,figsize=(7,2.9),gridspec_kw={'width_ratios':[1,1.85]});fig.subplots_adjust(left=.05,right=.985,bottom=.1,top=.88,wspace=.32)
 ax=axs[0];ax.set_aspect('equal');ax.set(xlim=(-.45,1.6),ylim=(-1.3,1.4));ax.axhline(0,color=PURPLE,lw=2);ax.axvline(0,color='#dddddd',lw=.7)
 for end,c in [((0,1),GREEN),((1,-1),GREEN),((1,0),INK)]:ax.annotate('',xy=end,xytext=(0,0),arrowprops=dict(arrowstyle='->',color=c,lw=1.2))
 ax.text(.1,1.07,r'$q_s=e_2$',color=GREEN);ax.text(.88,-1.13,r'$e_1-e_2$',ha='center',color=GREEN);ax.text(1.05,.15,r'$h_d=e_1$');ax.text(.4,-.42,'Target span',color=PURPLE);ax.set_xticks([]);ax.set_yticks([])
 for sp in ax.spines.values():sp.set_visible(False)
 ax.set_title('(a) Exact data reconstruction, impossible write',loc='left',fontsize=8)
 ax=axs[1];ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1));ax.set_title('(b) Three tests answer different questions',loc='left',fontsize=8)
 for y,name,obj,test,col in [(.78,'Source validity',r'$q_s$','Does the requested behavior occur?',INK),(.45,'Reading fidelity',r'$\hat q_t$','Does the target predict the source effect?',GREEN),(.12,'Native execution',r'$D_t\Delta z_t^{\,edit}$','Can actual target codes realize that effect?',PURPLE)]:
  ax.text(0,y,name,fontsize=9,color=col);ax.text(.45,y,obj,fontsize=11,ha='center');ax.text(0,y-.12,test,fontsize=8)
 save(fig,'correspondence_tests')
