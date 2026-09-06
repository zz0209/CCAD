"""Frozen absolute endpoint and secondary metrics, including all eight controls."""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.aris/plot_runtime_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
 out=ROOT/'artifacts/training_gain_confirmation_20260906';s=json.loads((out/'summary.json').read_text());methods=['raw','full','best_atom','geometric_atom','shared16','same_support_ridge','dynamic_pair_ridge','geometric_pair_ridge'];labels=['Raw','Full','Dynamic atom','Geometric atom','Shared <=16','Same support ridge','Dynamic pair','Geometric pair'];colors=['#8c5fb5','#009e73','#777777','#ba6c1b','#0072b2','#56b4e9','#555555','#aa4499']
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 fig,axs=plt.subplots(2,3,figsize=(13,7))
 for row,seed in enumerate([1,2]):
  for col,(field,title) in enumerate([('candidate_kl','Primary: absolute KL'),('normalized_kl_error','Secondary: relative KL'),('nll_error','Secondary: observed-token NLL')]):
   ax=axs[row,col]
   for m,label,color in zip(methods,labels,colors):
    v=next(x for x in s['changes'] if x['seed']==seed and x['method']==m)['metrics'][field];ax.plot([0,1],[v['initial'],v['final']],label=label,color=color,marker='o',ms=4,lw=2.3 if m=='shared16' else 1,ls='-' if m in ['shared16','full','raw'] else '--')
   ax.set(yscale='log',xticks=[0,1],xticklabels=['Early: 0.131M','Late: 4.19M'],title=f'Target {seed} | {title}',ylabel='Worst-operator median error',xlabel='Packed training tokens');ax.grid(alpha=.15)
 fig.suptitle('Frozen early-late prediction on new natural documents',fontsize=14)
 fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.95),ncol=4,frameon=False)
 fig.text(.5,.015,'24 new document prefixes / 12 reciprocal pairs; fixed source3 and two target trajectories. All eight maps retained.\nAbsolute KL is primary only for this predeclared confirmation; earlier development retained its original relative-KL primary.\nLines show two endpoints, not a monotonic learning curve. Pair-bootstrap intervals are reported in FINDINGS.md.',ha='center',fontsize=8)
 fig.tight_layout(rect=[0,.10,1,.85])
 for ext in ['png','pdf','svg']:fig.savefig(out/f'training_gain_confirmation.{ext}',dpi=180,bbox_inches='tight')
if __name__=='__main__':main()
