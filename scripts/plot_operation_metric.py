"""Show all cells and component trade-offs, including worsened cases."""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.aris/plot_runtime_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
 o=ROOT/'artifacts/operation_metric_20260906';s=json.loads((o/'summary.json').read_text());fig,ax=plt.subplots(1,2,figsize=(10,4.8));names=['Hyphen','Apostrophe','the','on'];colors=['#0072b2','#d55e00','#009e73','#aa4499']
 for q in range(4):
  rr=[r for r in s['rows'] if r['query']==q]
  for i,r in enumerate(rr):
   ax[0].scatter(q+(i-1.5)*.13,r['ratio'],c=colors[q],s=40);ax[0].annotate(str(r['seed']),(q+(i-1.5)*.13,r['ratio']),xytext=(3,3),textcoords='offset points',fontsize=8)
   ax[1].scatter(*r['component_ratios'],c=colors[q],s=40,label=names[q] if i==0 else None)
 ax[0].axhline(1,color='gray',ls='--');ax[0].set(xticks=range(4),xticklabels=names,ylabel='New / old mean iid-donor hook error',title='All 16 query-target cells');ax[0].grid(alpha=.15)
 ax[1].axhline(1,color='gray',ls='--');ax[1].axvline(1,color='gray',ls='--');ax[1].set(xlabel='Component A variance ratio',ylabel='Component B variance ratio',title='Component trade-offs');ax[1].legend(frameon=False);ax[1].grid(alpha=.15)
 fig.suptitle('Output weighting changes compact support, with mixed held-out surrogate gains');fig.tight_layout(rect=(0,.10,1,.93));fig.text(.5,.025,'Existing paired calibration; lower ratios are better. Labels are target seeds sharing source 1.\nSame solver/path/support cap; actual support sizes vary. No LM or independent behavioral confirmation.',ha='center',fontsize=9)
 for ext in ['png','pdf','svg']:fig.savefig(o/f'operation_metric.{ext}',dpi=180,bbox_inches='tight')
if __name__=='__main__':main()
