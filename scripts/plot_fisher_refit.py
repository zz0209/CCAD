"""Show primary and secondary fixed-method outcomes together."""
from pathlib import Path
import json,os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.aris/plot_runtime_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 out=ROOT/'artifacts/fisher_refit_20260906';s=json.loads((out/'summary.json').read_text());methods=['best_atom','shared16','euclidean_refit','constant_fisher_refit','pointwise_fisher_refit','full','raw'];labels=['Best atoms (2)','Original sparse (15)','Euclidean refit (15)','Constant Fisher (15)','Pointwise Fisher (15)','Full (2223)','Raw (768 hook dims)'];colors=['#777777','#0072b2','#d18b15','#7d65ab','#cf5963','#009e73','#555555'];metrics=['primary_worst_operator_median_kl_ratio','worst_operator_median_absolute_kl','worst_operator_median_observed_nll_error'];titles=['Primary: relative KL','Absolute KL','Observed-token NLL error']
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 fig,axes=plt.subplots(1,3,figsize=(12,4.8),sharey=True)
 for ax,key,title in zip(axes,metrics,titles):
  for i,(m,color) in enumerate(zip(methods,colors)):
   value=s['methods'][m][key];ax.plot(value,i,'o',color=color,ms=7);ax.annotate(f'{value:.4g}',(value,i),xytext=(7,0),textcoords='offset points',va='center',fontsize=8)
  ax.set(xscale='log',title=title,xlabel='Worst-operator median',yticks=range(len(methods)),yticklabels=labels);ax.grid(axis='x',alpha=.15);lo,hi=ax.get_xlim();ax.set_xlim(lo*.8,hi*2)
 axes[0].invert_yaxis();fig.suptitle('Fresh natural inputs: functional refitting exposes a trade-off',fontsize=14)
 fig.text(.5,.02,'24 prefixes / 12 reciprocal document pairs; one fixed source3 -> target1. All three refits share support and fitting inputs.\nPrimary KL does not favor pointwise Fisher over Euclidean. Full / raw use larger interfaces; all 11 methods and pair-bootstrap intervals are in the report.',ha='center',fontsize=9)
 fig.subplots_adjust(top=.83,bottom=.22,left=.19,wspace=.25)
 for ext in ['png','pdf','svg']:fig.savefig(out/f'refit_tradeoff.{ext}',dpi=180,bbox_inches='tight')
if __name__=='__main__':main()

