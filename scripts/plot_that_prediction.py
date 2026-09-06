"""Natural input/output prediction and all-target recovery, including challenge set."""
from pathlib import Path
import json,os
ROOT=Path(__file__).resolve().parents[1];os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.aris/plot_runtime_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
def main():
 out=ROOT/'artifacts/that_prediction_20260906';s=json.loads((out/'summary.json').read_text());families=['attitude','report','noun_control'];colors=['#0072b2','#d17823','#009e73'];plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 fig,axes=plt.subplots(2,2,figsize=(12,8));a,b,c,d=axes.flat
 for j,(family,col) in enumerate(zip(families,colors)):
  rr=[r for r in s['native_details'] if r['family']==family];x=j+np.linspace(-.1,.1,8);a.scatter(x,[r['activation'] for r in rr],color=col,s=22);a.hlines(s['native'][family]['activation_median'],j-.2,j+.2,color=col,lw=2)
  for op,shift,marker in [('add',-.14,'o'),('remove',.14,'x')]:b.scatter(x+shift,[r[op] for r in rr],color=col,marker=marker,s=24,label=op if j==0 else None)
 for ax in [a,b]:ax.set(xticks=range(3),xticklabels=['Attitude words','Report words','Noun challenge']);ax.grid(axis='y',alpha=.15)
 a.set(title='A  Source input preference',ylabel='Source1:2379 activation')
 b.set(title='B  Frozen source-native output prediction',ylabel='Change in personal-token / there log ratio');b.axhline(0,color='#555',lw=.8);b.legend(frameon=False)
 methods=['best_atom','geometric_atom','sparse16','full','raw'];labels=['Best atom','Geometric atom','Sparse FCC (14-16)','Full','Raw'];cols=['#777777','#d17823','#0072b2','#009e73','#9966aa']
 vals=[]
 for ax,family,title in [(c,'attitude_report','C  Attitude / report donor pairs'),(d,'noun_challenge','D  Noun challenge donor pairs')]:
  rows=[r for r in s['cells'] if r['family']==family]
  for k,(m,label,col) in enumerate(zip(methods,labels,cols)):
   y=[r['methods'][m]['mean_absolute_contrast_error'] for r in rows];vals.extend(y);ax.plot(np.arange(4)+(k-2)*.10,y,ls='none',marker='o',color=col,label=label)
  ax.set(xticks=range(4),xticklabels=['Target 2','Target 3','Target 4','Target 5'],title=title,ylabel='Mean absolute lexical-response error');ax.grid(axis='y',alpha=.15)
 for ax in [c,d]:ax.set_ylim(0,max(vals)*1.15)
 fig.suptitle('A testable output tendency and its cross-seed recovery',fontsize=15)
 fig.legend(*c.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.092),ncol=5,frameon=False,fontsize=9)
 fig.text(.5,.025,'24 fresh documents; fixed lexical groups, all zero/reverse/challenge cases retained. Native operations and donor transport differ.\nFour targets share one source; reciprocal donor requests are dependent. Lexical contrast is not semantic uniqueness or target-native attribution.',ha='center',fontsize=9)
 fig.subplots_adjust(top=.90,bottom=.19,hspace=.36,wspace=.26)
 for ext in ['png','pdf','svg']:fig.savefig(out/f'that_prediction.{ext}',dpi=180,bbox_inches='tight')
if __name__=='__main__':main()
