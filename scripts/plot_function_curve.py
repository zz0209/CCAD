"""Aggregate all fixed function-curve cells and render comparable operation panels."""
from pathlib import Path
import os,json,statistics,hashlib
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.aris/plot_runtime_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
 out=ROOT/'artifacts/function_curve_20260906';out.mkdir(exist_ok=True)
 old=[json.loads(s) for s in (ROOT/'runs/F4_colon_pair_shared_v1_20260906/metrics.raw.jsonl').read_text().splitlines()];ref={(r['case_id'],r['operator'],r['method']):r for r in old};records=[];identities=[];total=0.
 for step in [256,1024,2048,4096,8192]:
  for seed in [1,2]:
   run=ROOT/'runs'/f'F4_function_curve_s{seed}_step{step}_v1_20260906';st=json.loads((run/'metrics.summary.json').read_text());assert st['status']=='PASS';total+=st['wall_seconds'];rows=[json.loads(s) for s in (run/'metrics.raw.jsonl').read_text().splitlines()]
   assert all(all(r[k]==ref[r['case_id'],r['operator'],r['method']][k] for k in ['source_difference','common_dose','source_kl']) for r in rows)
   assert all(r['normalized_kl_error']==ref[r['case_id'],r['operator'],r['method']]['normalized_kl_error'] for r in rows if r['method']=='raw')
   with np.load(run/'coefficients.npz') as co:supports={m:int(np.count_nonzero(np.linalg.norm(co[m],axis=1))) for m in co.files if m!='source_decoder'}
   identities.append(dict(run=run.name,raw_sha256=hashlib.sha256((run/'metrics.raw.jsonl').read_bytes()).hexdigest(),wall_seconds=st['wall_seconds'],supports=supports))
   lookup={(r['case_id'],r['operator'],r['method']):r for r in rows}
   for op in ['field_component','clause_component','sum','difference']:
    rec=dict(seed=seed,step=step,packed_tokens=step*512,operator=op,methods={})
    for m in sorted({r['method'] for r in rows}):
     vals=[r for r in rows if r['method']==m and r['operator']==op];assert all(r['normalized_kl_error'] is not None for r in vals)
     rec['methods'][m]=dict(median=statistics.median(r['normalized_kl_error'] for r in vals),mean_vector_error=statistics.mean(r['vector_squared_error'] for r in vals),support=supports[m])
    rec['shared_vs_dynamic_pair_wins']=sum(r['normalized_kl_error']<lookup[r['case_id'],op,'dynamic_pair_ridge']['normalized_kl_error'] for r in rows if r['operator']==op and r['method']=='shared16')
    rec['shared_vs_geometric_pair_wins']=sum(r['normalized_kl_error']<lookup[r['case_id'],op,'geometric_pair_ridge']['normalized_kl_error'] for r in rows if r['operator']==op and r['method']=='shared16');records.append(rec)
 payload=dict(records=records,runs=identities,total_wall_seconds=total,total_forwards=4450,source_and_raw_exact_replay=True,scope='All ten fixed cells; one source pair, shared recipients/donors and dependent training checkpoints. Medians descriptive, not independent replicates.')
 (out/'summary.json').write_text(json.dumps(payload,indent=2))
 methods=['dynamic_pair_ridge','geometric_pair_ridge','shared16','full','raw'];labels=['Dynamic pair (<=2)','Geometric pair (<=2)','Shared (<=16)','Full','Raw'];colors=['#777777','#c17a18','#0072b2','#009e73','#8c5fb5'];styles=['--',':','-','-.','--']
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 fig,axes=plt.subplots(2,4,figsize=(14,7.2),sharex=True,sharey=True)
 allvals=[r['methods'][m]['median'] for r in records for m in methods];lo=min(allvals)*.7;hi=max(allvals)*1.4
 for i,seed in enumerate([1,2]):
  for j,op in enumerate(['field_component','clause_component','sum','difference']):
   ax=axes[i,j];subset=[r for r in records if r['seed']==seed and r['operator']==op];x=[r['packed_tokens']/1e6 for r in subset]
   for m,label,col,ls in zip(methods,labels,colors,styles):ax.plot(x,[r['methods'][m]['median'] for r in subset],color=col,linestyle=ls,marker='o',markersize=3,label=label)
   ax.set(xscale='log',yscale='log',ylim=(lo,hi),xticks=x,xticklabels=['.131','.524','1.05','2.10','4.19'],title=f"Target {seed}: {['Component 1','Component 2','Sum','Difference'][j]}");ax.grid(alpha=.17)
   if i==1:ax.set_xlabel('Packed train tokens (M; log scale)')
   if j==0:ax.set_ylabel('Median normalized output-KL error')
 fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.935),ncol=5,frameon=False)
 fig.suptitle('Fixed-function recovery across same-stream training checkpoints',fontsize=14)
 fig.text(.5,.02,'Fixed source3 pair; target1/2, five checkpoints, same 12 development inputs. Four operations share a dose per recipient.\nLines connect dependent checkpoints; all methods refit on the same discovery rows. Raw is fixed. No independent-context confirmation.',ha='center',fontsize=8)
 fig.tight_layout(rect=[0,.09,1,.875])
 for ext in ['png','pdf','svg']:fig.savefig(out/f'function_curve.{ext}',dpi=200,bbox_inches='tight')
 print(json.dumps(dict(output=str(out),total_wall_seconds=total,cells=len(records))))
if __name__=='__main__':main()
