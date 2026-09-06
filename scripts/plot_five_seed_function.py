"""Four fixed queries: all methods, early/late errors, and paired uncertainty."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.aris/plot_runtime_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
 out=ROOT/'artifacts/five_seed_function_20260906';s=json.loads((out/'summary.json').read_text());methods=['raw','full','best_atom','geometric_atom','shared16','same_support_ridge','dynamic_pair_ridge','geometric_pair_ridge','wrong_query'];labels=['Raw','Full','Dynamic atom','Geometric atom','Shared <=16','Same support ridge','Dynamic pair','Geometric pair','Swap + oracle energy'];colors=['#8c5fb5','#009e73','#888888','#ba6c1b','#0072b2','#56b4e9','#555555','#aa4499','#d55e00'];titles=['Hyphen (-)','Curved apostrophe','Article token (the)','Preposition token (on)'];plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'});fig,axes=plt.subplots(2,2,figsize=(11,7))
 for qi,ax in enumerate(axes.flat):
  for m,label,col in zip(methods,labels,colors):
   v=s['per_query'][m][str(qi)];ax.plot([0,1],[v['early'],v['late']],color=col,label=label,marker='o',ms=4,lw=2.3 if m=='shared16' else 1,ls='-' if m in ['raw','full','shared16'] else '--')
  ax.set(yscale='log',xticks=[0,1],xticklabels=['Early: 0.131M tokens','Late: 4.19M tokens'],title=titles[qi],ylabel='Mean across four targets of\nworst-operator median absolute KL');ax.grid(alpha=.15)
 fig.suptitle('Four source-defined queries on 64 new document prefixes',fontsize=14);fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,frameon=False)
 fig.text(.5,.015,'Source1 fixed; targets2/3/4/5 share the same controlled training stream and schedule. All four queries retained.\nSwap control receives oracle source energy; it diagnoses composition and is not an equal-information method baseline.\nEach query has 8 reciprocal document pairs; different queries use distinct documents. Intervals shown separately.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.12,1,.80])
 for ext in ['png','pdf','svg']:fig.savefig(out/f'four_query_function.{ext}',dpi=180,bbox_inches='tight')
 fig2,ax=plt.subplots(figsize=(9,4));rows=[(titles[q],s['per_query']['shared16'][str(q)]) for q in range(4)]+[('Prespecified panel mean',s['aggregate']['shared16']['candidate_kl'])]
 for y,(label,v) in enumerate(rows):
  lo,hi=v['bootstrap_95'];ax.plot([lo,hi],[y,y],color='#0072b2',lw=2);ax.scatter([v['delta']],[y],color='#0072b2',s=45 if y==4 else 25)
 ax.axvline(0,color='#555555',lw=1,ls='--');ax.set(yticks=range(5),yticklabels=[x[0] for x in rows],xlabel='Late minus early absolute KL (negative = improvement)',title='Shared compact maps: paired uncertainty');ax.invert_yaxis();ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0));fig2.text(.5,.015,'2000 whole-pair bootstrap draws within each query, shared across targets/stages; conditional on this fixed panel.',ha='center',fontsize=8);fig2.tight_layout(rect=[0,.06,1,1])
 for ext in ['png','pdf','svg']:fig2.savefig(out/f'compact_training_change.{ext}',dpi=180,bbox_inches='tight')
if __name__=='__main__':main()
