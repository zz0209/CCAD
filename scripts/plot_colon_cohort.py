"""Show all four operators in the frozen five-SAE/four-target cohort."""
from pathlib import Path
import json,os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.aris/plot_runtime_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
def main():
    out=ROOT/'artifacts/colon_cohort_20260906';s=json.loads((out/'summary.json').read_text())
    methods=['shared16','dynamic_pair_ridge','separate8_ridge','union16_ridge','full','raw']
    labels=['Shared group','Dynamic pair','Separate supports','Separate union','Full','Raw']
    colors=['#0072b2','#777777','#c17a18','#d45f5f','#009e73','#8c5fb5']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(12,7.6),sharey=True)
    vals=[c['methods'][m]['median_ratio'] for c in s['cells'] for m in methods];assert all(v is not None and v>0 for v in vals)
    lim=(min(vals)*.7,max(vals)*1.4)
    for ax,op,title in zip(axes.flat,['field_component','clause_component','sum','difference'],['Component 1','Component 2 (primary)','Sum','Difference']):
        rows=[c for c in s['cells'] if c['operator']==op];x=np.arange(4)
        for k,(m,label,col) in enumerate(zip(methods,labels,colors)):
            ax.plot(x+(k-2.5)*.09,[c['methods'][m]['median_ratio'] for c in rows],ls='none',marker='o',ms=5,color=col,label=label)
        ax.set(xticks=x,xticklabels=['Target 1','Target 2','Target 4','Target 5'],yscale='log',ylim=lim,title=title)
        ax.axhline(1,color='#777777',lw=.7,ls=':');ax.grid(axis='y',alpha=.15)
    for ax in axes[:,0]:ax.set_ylabel('Median normalized output-KL error')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.925),ncol=6,frameon=False,fontsize=9)
    fig.suptitle('Frozen component recovery across four target SAEs',fontsize=15)
    fig.text(.5,.025,'24 fresh documents / 12 cross-document donor pairs; shared source3 and reciprocal directions dependent.\nAll ten methods frozen before new text; component2 selected as primary on earlier development. All four operations retained.\nMedians are conditional on this source and sample; absolute errors, document-pair bootstrap and all controls are in summary.json.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.82,bottom=.19,hspace=.35,wspace=.22)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'cohort_confirmation.{ext}',dpi=190,bbox_inches='tight')
if __name__=='__main__':main()
