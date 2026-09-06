"""Quality and functional recovery across every fixed natural-input checkpoint."""
from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.aris/plot_runtime_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 out=ROOT/'artifacts/natural_function_curve_20260906';s=json.loads((out/'summary.json').read_text());methods=['dynamic_pair_ridge','geometric_pair_ridge','shared16','full','raw'];labels=['Dynamic pair (<=2)','Geometric pair (<=2)','Shared (<=16)','Full','Raw'];colors=['#777777','#c17a18','#0072b2','#009e73','#8c5fb5'];styles=['--',':','-','-.','--'];plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
 for i,seed in enumerate([1,2]):
  rows=[x for x in s['records'] if x['seed']==seed];xx=[x['tokens']/1e6 for x in rows];ax=axes[i,0]
  for key,label,col in [('fve','FVE','#333333'),('ce_recovered','CE recovered','#0072b2')]:ax.plot(xx,[x['quality'][key] for x in rows],marker='o',ms=3,label=label,color=col)
  ax.set(ylim=(.88,1.005),title=f'Target {seed}: fixed-validation quality',ylabel='Recovered fraction');ax.legend(frameon=False,fontsize=8)
  for j,key,title in [(1,'worst_median_kl_ratio','Natural-input relative KL'),(2,'worst_median_absolute_kl','Natural-input absolute KL')]:
   ax=axes[i,j]
   for m,label,col,ls in zip(methods,labels,colors,styles):ax.plot(xx,[x['methods'][m][key] for x in rows],label=label,color=col,linestyle=ls,marker='o',ms=3)
   ax.set(yscale='log',title=f'Target {seed}: {title}',ylabel='Worst-operator median error')
  for ax in axes[i]:
   ax.set(xscale='log',xticks=xx,xticklabels=['.131','.524','1.05','2.10','4.19']);ax.grid(alpha=.15)
   if i==1:ax.set_xlabel('Packed train tokens (M; log scale)')
 fig.legend(*axes[0,1].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.64,.935),ncol=3,frameon=False,fontsize=9)
 fig.suptitle('High reconstruction quality does not settle functional recovery',fontsize=14)
 fig.text(.5,.02,'Same two target trajectories and fixed source3; all five checkpoints, 24 exposed natural prefixes / 12 reciprocal document pairs.\nFrozen discovery maps, no refitting on these inputs. Lines connect dependent stages; source and raw outputs replay exactly.\nQuality uses separate fixed validation. Early checkpoints follow the full-run LR schedule; this is not a short-budget optimization comparison.',ha='center',fontsize=8)
 fig.tight_layout(rect=[0,.10,1,.85])
 for ext in ['png','pdf','svg']:fig.savefig(out/f'natural_training_curve.{ext}',dpi=180,bbox_inches='tight')
 fig2,axes2=plt.subplots(2,4,figsize=(14,7),sharex=True,sharey=True)
 for i,seed in enumerate([1,2]):
  rr=[x for x in s['records'] if x['seed']==seed];xx=[x['tokens']/1e6 for x in rr]
  for j,op in enumerate(['field_component','clause_component','sum','difference']):
   ax=axes2[i,j]
   for m,label,col,ls in zip(methods,labels,colors,styles):ax.plot(xx,[x['methods'][m]['per_operator'][op]['median_kl_ratio'] for x in rr],label=label,color=col,linestyle=ls,marker='o',ms=3)
   ax.set(xscale='log',yscale='log',title=f'Target {seed}: {op}',xticks=xx,xticklabels=['.131','.524','1.05','2.10','4.19']);ax.grid(alpha=.15)
   if i==1:ax.set_xlabel('Packed train tokens (M)')
   if j==0:ax.set_ylabel('Median normalized KL')
 fig2.legend(*axes2[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.95),ncol=5,frameon=False);fig2.suptitle('All prescribed operations retained',fontsize=14);fig2.tight_layout(rect=[0,.03,1,.89])
 for ext in ['png','pdf','svg']:fig2.savefig(out/f'natural_training_operators.{ext}',dpi=180,bbox_inches='tight')
if __name__=='__main__':main()
