"""Standalone source-backed research figures; uses an isolated plotting runtime."""
import os,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.aris/plot_runtime_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pair','curve']);args=ap.parse_args();plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 if args.mode=='pair':
  run=ROOT/'runs/F4_colon_pair_shared_v1_20260906';data=json.loads((run/'comparison.json').read_text());grams={r['method']:r for r in json.loads((run/'gram_summary.json').read_text())}
  methods=['geometric_pair_ridge','sparse16','shared16','full','raw'];labels=['Geometric pair (2)','Independent sparse (28)','Shared support (15)','Full (2211)','Raw hook (768)'];colors=['#777777','#c17a18','#0072b2','#009e73','#8c5fb5'];styles=['--',':','-','-.','--']
  fig,axes=plt.subplots(1,2,figsize=(11.8,4.6),gridspec_kw={'width_ratios':[1.65,1]})
  for m,label,col,style in zip(methods,labels,colors,styles):
   vals=[g['methods'][m]['median_kl_ratio'] for g in data['groups']];axes[0].plot(range(4),vals,label=label,color=col,linestyle=style,marker='o',markersize=5)
  axes[0].set(xticks=range(4),xticklabels=['Component 1','Component 2','Sum','Difference'],ylabel='Median normalized output-KL error',yscale='log',title='A   Separate and mixed source-aligned operations');fig.legend(*axes[0].get_legend_handles_labels(),fontsize=8,loc='upper center',bbox_to_anchor=(.35,.89),ncol=3,frameon=False);axes[0].grid(axis='y',alpha=.2)
  vals=[max(grams[m]['eigenvalues']) for m in methods];axes[1].barh(range(5),vals,color=colors);axes[1].set(yticks=range(5),yticklabels=['Pair (2)','Sparse (28)','Shared (15)','Full','Raw'],xlabel='Largest eigenvalue of mean error Gram',title='B   Empirical vector error');axes[1].invert_yaxis();axes[1].set_xlim(0,max(vals)*1.3)
  for i,v in enumerate(vals):axes[1].text(v+.001,i,f'{v:.4f}',va='center',fontsize=9)
  fig.suptitle('Two source contributions: recovery and compactness trade-offs',fontsize=14)
  fig.text(.5,.025,'One source-target seed pair; 12 related inputs. Parenthesized values are active input counts, not matched capacities.\nSame dose per input. Development panel; empirical vector error does not guarantee unseen behavioral effects.',ha='center',fontsize=8)
  fig.tight_layout(rect=[0,.13,1,.78]);base=run/'pair_operation_figure'
 else:
  run=ROOT/'runs/R012_same_stream_curve_v1_20260906';rows=[json.loads(s) for s in (run/'metrics.raw.jsonl').read_text().splitlines()];assert len(rows)==10
  fig,axes=plt.subplots(1,3,figsize=(12,4.1));specs=[('fve','Fraction of variance explained'),('ce_recovered','CE recovered'),('alive_features','Features observed active')]
  for seed,col,style in [(1,'#0072b2','-'),(2,'#d55e00','--')]:
   sub=sorted([r for r in rows if r['seed']==seed],key=lambda r:r['step']);x=[r['packed_train_tokens']/1e6 for r in sub]
   for ax,(key,title) in zip(axes,specs):
    ax.plot(x,[r['quality'][key] for r in sub],color=col,linestyle=style,marker='o',label=f'Seed {seed}');ax.set(xscale='log',xticks=x,xticklabels=['0.131','0.524','1.049','2.097','4.194'],xlabel='Packed training tokens (million)',title=title);ax.grid(alpha=.2);ax.tick_params(axis='x',labelsize=8)
  axes[0].legend();fig.suptitle('Same-stream, fixed-validation training trajectory',fontsize=14)
  fig.text(.5,.02,'Two initializations; 64 fixed validation sequences. Prefix checkpoints share the full 8192-step LR schedule.\nActive features are measured on this validation batch, not permanent alive/dead status. Quality is not FCC or semantic validity.',ha='center',fontsize=8)
  fig.tight_layout(rect=[0,.13,1,.9]);base=run/'training_curve'
 for ext in ['png','pdf','svg']:fig.savefig(str(base)+'.'+ext,dpi=220,bbox_inches='tight')
 print(base)
if __name__=='__main__':main()
