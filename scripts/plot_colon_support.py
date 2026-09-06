"""Display all exposed support-selection panels without hiding tradeoffs."""
from pathlib import Path
import json,os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.aris/plot_runtime_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
def main():
    out=ROOT/'artifacts/colon_support_tradeoff_20260906';s=json.loads((out/'summary.json').read_text())
    methods=['same_support_ridge','separate8_ridge','union16_ridge','full','raw']
    labels=['Shared support + ridge','Separate supports','Separate union + ridge','Full','Raw']
    colors=['#0072b2','#c17a18','#d45f5f','#009e73','#8c5fb5']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(12,7.5),sharey=True)
    values=[c['methods'][m]['median_ratio'] for c in s['cells'] for m in methods];limits=(min(values)*.7,max(values)*1.4)
    for i,panel in enumerate(['authored_development','natural_development']):
        for j,seed in enumerate([1,2]):
            ax=axes[i,j];rows=[c for c in s['cells'] if c['seed']==seed and c['panel']==panel];x=np.arange(4)
            for off,(m,label,col) in enumerate(zip(methods,labels,colors)):
                ax.plot(x+(off-2)*.08,[c['methods'][m]['median_ratio'] for c in rows],ls='none',marker='o',color=col,label=label,ms=5)
            ax.set(xticks=x,xticklabels=['Component 1','Component 2','Sum','Difference'],yscale='log',ylim=limits,title=f"Target {seed}: {'authored' if i==0 else 'natural'} development")
            if j==0:ax.set_ylabel('Median normalized output-KL error')
            ax.grid(axis='y',alpha=.15)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.925),ncol=5,frameon=False,fontsize=9)
    fig.suptitle('Support selection changes which component is recovered',fontsize=15)
    fig.text(.5,.025,'Same <=16 total-member ceiling; shared / separate-union actual sizes: target1 15 / 12, target2 14 / 14.\nOriginal 12 authored and 24 natural inputs are both exposed development; shared source and reciprocal directions are dependent.\nAll old controls replay exactly. Medians do not imply casewise dominance; absolute errors and all ten methods are in the linked summary.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.82,bottom=.19,hspace=.35,wspace=.22)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'support_tradeoff.{ext}',dpi=190,bbox_inches='tight')
if __name__=='__main__':main()
