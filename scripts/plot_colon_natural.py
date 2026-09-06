"""Render frozen natural-context findings with the isolated plotting runtime."""
from pathlib import Path
import json,os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.aris/plot_runtime_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    out=ROOT/'artifacts/colon_natural_20260906';s=json.loads((out/'summary.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for j,atom in enumerate([1850,2897]):
        ax=axes[0,j]
        for p in s['pairs']:
            vals=[p['field_activation'][j],p['clause_activation'][j]]
            ax.plot([0,1],vals,color='#0072b2' if p['predictions'][j] else '#c17a18',alpha=.7,marker='o',markersize=3,lw=1)
        ax.set(xticks=[0,1],xticklabels=['Short-field proxy','Long-clause proxy'],ylabel='Source activation',title=f"Source {atom}: predicted ordering {s['source_hypothesis_successes'][j]}/12")
        ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.15)
    methods=['dynamic_pair_ridge','geometric_pair_ridge','shared16','full','raw']
    labels=['Dynamic pair','Geometric pair','Shared group','Full','Raw']
    colors=['#777777','#c17a18','#0072b2','#009e73','#8c5fb5']
    for j,seed in enumerate([1,2]):
        ax=axes[1,j];rows=[r for r in s['cells'] if r['seed']==seed];x=np.arange(4)
        for off,(m,label,color) in enumerate(zip(methods,labels,colors)):
            ax.plot(x+(off-2)*.07,[r['methods'][m]['median_ratio'] for r in rows],linestyle='none',marker='o',color=color,label=label,markersize=5)
        ax.axhline(1,color='#555555',lw=.7,ls=':')
        ax.set(xticks=x,xticklabels=['Component 1','Component 2','Sum','Difference'],yscale='log',ylim=(.003,2),ylabel='Median normalized output-KL error',title=f'Target seed {seed}; lower is better')
        ax.grid(axis='y',alpha=.15)
    fig.legend(*axes[1,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.55),ncol=5,frameon=False)
    fig.suptitle('Frozen two-component correspondence on new natural contexts',fontsize=15)
    fig.text(.5,.025,'24 documents / 12 cross-document donor pairs; all retained, shared source and reciprocal directions dependent.\nTop: blue matches the frozen input prediction; orange is contrary. Bottom: medians across 24 recipients, not independent seed estimates.\nLexical proxies include time/citation cases; source-aligned operations do not establish target-native or semantic mechanisms.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.89,bottom=.19,hspace=.65,wspace=.25)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'natural_confirmation.{ext}',dpi=190,bbox_inches='tight')
if __name__=='__main__':main()

