"""Fixed-validation quality and actual training traces for all five seeds."""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.aris/plot_runtime_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
 out=ROOT/'artifacts/five_seed_training_20260906';s=json.loads((out/'summary.json').read_text());fig,axs=plt.subplots(2,2,figsize=(10,7));colors=['#0072b2','#e69f00','#009e73','#cc79a7','#555555'];plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none'})
 for seed,color in zip(range(1,6),colors):
  rr=sorted([r for r in s['rows'] if r['seed']==seed],key=lambda r:r['step']);xx=[r['packed_train_tokens']/1e6 for r in rr]
  for ax,key,title in [(axs[0,0],'fve','Fixed-validation FVE'),(axs[0,1],'ce_recovered','Fixed-validation CE recovery'),(axs[1,0],'alive_features','Features firing on fixed validation')]:
   ax.plot(xx,[r['quality'][key] for r in rr],marker='o',ms=3,color=color,label=f'Seed {seed}');ax.set(title=title,xlabel='Packed training tokens (M)',xscale='log',xticks=xx,xticklabels=['.131','.524','1.05','2.10','4.19']);ax.grid(alpha=.15)
  tr=[r for r in s['training_blocks'] if r['seed']==seed];axs[1,1].plot([r['step_end']*512/1e6 for r in tr],[r['median_fvu'] for r in tr],color=color,label=f'Seed {seed}')
 axs[1,1].set(title='Training FVU: 512-step block medians',xlabel='Packed training tokens (M)',ylabel='FVU');axs[1,1].grid(alpha=.15)
 fig.suptitle('Same stream and schedule, five controlled initialization seeds',fontsize=13);fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.94),ncol=5,frameon=False)
 fig.text(.5,.015,'Seeds1/2 reused; seeds3/4/5 added. All8192 actual input hashes match; 64 fixed validation sequences.\nTopK128 / 3072 latents, Pythia160M layer5. Training blocks use different batches; no convergence claim. Quality is separate from atom/FCC functionality.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.08,1,.87])
 for ext in ['png','pdf','svg']:fig.savefig(out/f'five_seed_quality.{ext}',dpi=180,bbox_inches='tight')
if __name__=='__main__':main()
