"""Equal-energy signed response versus whole-distribution perturbation."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.aris/plot_runtime_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
def main():
 out=ROOT/'artifacts/that_controls_20260906';s=json.loads((out/'summary.json').read_text());plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 fig,(a,b)=plt.subplots(1,2,figsize=(12,5.2));names=['native','atom_988','random_0','random_1'];labels=['Source2379','Same-token988','Random0','Random1'];families=['attitude','report','noun_control'];colors=['#0072b2','#d17823','#009e73']
 for i,name in enumerate(names):
  for j,(family,col) in enumerate(zip(families,colors)):
   rr=[r for r in s['detail'] if r['direction']==name and r['family']==family];x=i+(j-1)*.16+np.linspace(-.04,.04,8);a.scatter(x,[r['signed_slope'] for r in rr],color=col,s=18,label=family if i==0 else None);b.scatter(x,[r['mean_kl'] for r in rr],color=col,s=18)
  group=next(r for r in s['cells'] if r['direction']==name and r['family']=='all');a.hlines(group['median_signed_slope'],i-.3,i+.3,color='#222',lw=2);b.hlines(group['median_mean_kl'],i-.3,i+.3,color='#222',lw=2)
 for ax in [a,b]:ax.set(xticks=range(4),xticklabels=labels);ax.grid(axis='y',alpha=.15)
 a.axhline(0,color='#555',lw=.8);a.set(title='Signed lexical response at equal hook energy',ylabel='(Add response - remove response) / 2')
 b.set(title='Whole-distribution response',ylabel='Mean KL to baseline over add / remove',yscale='log')
 fig.suptitle('Same token preference does not imply the same output function',fontsize=15)
 fig.legend(*a.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.12),ncol=3,frameon=False)
 fig.text(.5,.025,'24 exposed natural documents; three frozen controls, no direction/sign search. Dots: documents; black bars: medians.\nSame hook norm does not match output KL. Two random samples do not establish uniqueness or a null distribution.',ha='center',fontsize=9)
 fig.subplots_adjust(top=.84,bottom=.27,wspace=.3)
 for ext in ['png','pdf','svg']:fig.savefig(out/f'direction_controls.{ext}',dpi=180,bbox_inches='tight')
if __name__=='__main__':main()
