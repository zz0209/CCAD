"""Source-backed core paper tables and figures; no fit or model inference."""
from pathlib import Path
import json,hashlib,statistics,collections,sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/core_contribution_20260906'
def read(p):return json.loads((ROOT/p).read_text())
def lines(p):return [json.loads(s) for s in (ROOT/p).read_text().splitlines() if s]
def median(xs):
 a=[x for x in xs if x is not None];return statistics.median(a) if a else None
def main():
 fine=lines('runs/F4_contribution_completion_v1_20260906/metrics.raw.jsonl');fine=[r for r in fine if r['source_seed']!=r['target_seed']];grain=lines('runs/F4_contribution_grain_v1_20260906/metrics.raw.jsonl');natural=lines('runs/F4_contribution_natural_v2_20260906/metrics.raw.jsonl');cases=read('runs/F4_contribution_natural_v2_20260906/source_cases.json');bank=collections.defaultdict(list)
 for r in natural:bank[(r['source_seed'],r['query_id'],r['target_seed'],r['size'],r['kind'],r['method'])].append(r)
 agg=[]
 for k,v in bank.items():
  energy=sum(x['source_energy'] for x in v);kl=sum(x['source_kl'] for x in v);nll=sum(x['source_nll_change']**2 for x in v)
  agg.append(dict(zip(['source_seed','query_id','target_seed','size','kind','method'],k))|dict(pairs=len(v),hook=sum(x['vector_squared_error'] for x in v)/energy if energy>1e-20 else None,kl=sum(x['candidate_kl'] for x in v)/kl if kl>1e-12*len(v) else None,nll=sum(x['nll_change_squared_error'] for x in v)/nll if nll>1e-20 else None,absolute_kl=statistics.mean(x['candidate_kl'] for x in v),absolute_hook=statistics.mean(x['vector_squared_error'] for x in v),source_kl=kl/len(v),source_energy=energy/len(v)))
 (OUT/'natural_aggregates.json').write_text(json.dumps(agg,indent=2)+'\n')
 def tables(rows,field):
  b=collections.defaultdict(list)
  for r in rows:b[(r['size'],r['kind'],r['method'])].append(r)
  return [dict(size=k[0],kind=k[1],method=k[2],cells=len(v),**{f:median(x[f] for x in v) for f in field},by_source={s:{f:median(x[f] for x in v if x['source_seed']==s) for f in field} for s in range(1,6)}) for k,v in b.items()]
 flattened=[dict(source_seed=r['source_seed'],size=r['size'],kind=r['kind'],method=m,error=v['relative_error']) for r in grain for m,v in r['methods'].items()]
 ft={m:dict(median=median(r['methods'][m]['relative_error'] for r in fine),by_source={s:median(r['methods'][m]['relative_error'] for r in fine if r['source_seed']==s) for s in range(1,6)}) for m in fine[0]['methods']}
 lookup={(r['query_id'],r['target_seed'],r['size'],r['kind'],r['method']):r for r in agg};contrasts=[]
 for tag,ak,bk in [('fine_readout_vs_native',(1,'coherent','readout_same64'),(1,'coherent','native64')),('coherent16_vs_random16',(16,'coherent','native64'),(16,'random','native64')),('coherent_native64_vs_marginal64',(16,'coherent','native64'),(16,'coherent','marginal64')),('coherent_native64_vs_matching',(16,'coherent','native64'),(16,'coherent','matching_refit'))]:
  pair=[(a,lookup[(q,t,*bk)]) for (q,t,z,k,m),a in lookup.items() if (z,k,m)==ak];row=dict(contrast=tag,cells=len(pair),metrics={})
  for f in ['hook','kl','nll']:
   valid=[(a,b) for a,b in pair if a[f] is not None and b[f] is not None];diff=[a[f]-b[f] for a,b in valid];row['metrics'][f]=dict(valid=len(valid),improved=sum(x<0 for x in diff),worse=sum(x>0 for x in diff),median_paired_difference=median(diff),by_source={s:dict(a=median(a[f] for a,b in valid if a['source_seed']==s),b=median(b[f] for a,b in valid if a['source_seed']==s)) for s in range(1,6)})
  contrasts.append(row)
 docs={c[k] for c in cases['cases'] for k in ['recipient_document','donor_document']};payload=dict(written_at_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),fine_cells=len(fine),grain_cells=len(grain),natural_raw_rows=len(natural),natural_queries=len({c['query_id'] for c in cases['cases']}),natural_pairs=len(cases['cases']),natural_distinct_documents=len(docs),missing_requests=cases['missing'],fine=ft,grain=tables(flattened,['error']),natural=tables(agg,['hook','kl','nll','absolute_kl','absolute_hook','source_kl','source_energy']),contrasts=contrasts,scope='All statistics descriptive. Natural endpoints sum over2source-selected pairs before query-target aggregation. 160cells share40queries/five seeds/documents; no independent-direction inference. Coherent vs random source groups are matched in size and anchor, not semantic label/covariance rank or source effect magnitude.',inputs=[dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in ['runs/F4_contribution_completion_v1_20260906/metrics.raw.jsonl','runs/F4_contribution_grain_v1_20260906/metrics.raw.jsonl','runs/F4_contribution_natural_v2_20260906/metrics.raw.jsonl']])
 (OUT/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
 # Plot-only interpreter and isolated matplotlib target, separate from SAE runtime.
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 colors=['#176B87','#BB4430','#658C48','#855C93','#C28E2C']
 def save(fig,name):
  fig.savefig(OUT/(name+'.png'),dpi=190,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
 witness=read('artifacts/core_contribution_20260906/mask_algebra_witness/results.json')['rows'];fig,ax=plt.subplots(1,2,figsize=(10.4,3.5))
 for key,label,c in [('fine','Fine source operation',colors[1]),('true_block','Correct 2-feature block',colors[0]),('wrong_coarsening','Wrong 2-feature merge',colors[3])]:
  v=[x for x in witness if x['query']==key];style='--x' if key=='wrong_coarsening' else '-o';ax[0].plot([x['rho'] for x in v],[x['native_relative_hook_error'] for x in v],style,label=label,color=c);ax[1].plot([x['rho'] for x in v],[max(0,x['native_relative_kl']) for x in v],style,label=label,color=c)
 for a in ax:a.set_xlabel('Within-block mixing strength');a.set_ylim(bottom=-.008);a.grid(alpha=.16)
 ax[0].set_ylabel('Relative hook error');ax[1].set_ylabel('Relative nonlinear-output KL');ax[0].legend(fontsize=8);fig.suptitle('Exact reconstruction and readout do not fix intervention granularity',fontsize=12);fig.tight_layout();save(fig,'figure1_mask_algebra')
 fig,ax=plt.subplots(1,2,figsize=(11.1,4))
 for s,c in zip(range(1,6),colors):
  v=[r for r in fine if r['source_seed']==s];ax[0].scatter([r['methods']['readout_native64_support']['relative_error'] for r in v],[r['methods']['calibration_native_oracle']['relative_error'] for r in v],s=12,alpha=.55,color=c,label=f'Source {s}')
 ax[0].set_xscale('log');ax[0].set_xlabel('Readout error using selected 64 codes');ax[0].set_ylabel('Full native calibration-oracle error');ax[0].set_ylim(0,1.03);ax[0].legend(fontsize=7,loc='lower right');ax[0].set_title('640 cross-seed atom queries')
 for kind,style,label,c in [('coherent','-','Source-coherent',colors[0]),('random','--','Size-matched random',colors[1])]:
  sizes=[1,4,16,64] if kind=='coherent' else [4,16,64];ys=[median(r['methods']['native64']['relative_error'] for r in grain if r['size']==z and r['kind']==kind) for z in sizes];ax[1].plot(sizes,ys,style+'o',color=c,label=label)
  for s in range(1,6):ax[1].plot(sizes,[median(r['methods']['native64']['relative_error'] for r in grain if r['size']==z and r['kind']==kind and r['source_seed']==s) for z in sizes],style,color=c,alpha=.22,linewidth=.8)
 ax[1].set_xscale('log',base=4);ax[1].set_xticks([1,4,16,64],['1','4','16','64']);ax[1].set_xlabel('Number of source members');ax[1].set_ylabel('Relative target-native error (64 members)');ax[1].set_ylim(.45,1.02);ax[1].set_title('Coarsening helps; larger is not always better');ax[1].legend(fontsize=8)
 for a in ax:a.grid(alpha=.14)
 fig.tight_layout();save(fig,'figure2_real_granularity')
 fig,ax=plt.subplots(1,2,figsize=(10.5,4.1));panels=[[(1,'coherent','readout_same64','Readout\n64 inputs'),(1,'coherent','native64','Native\n64 members'),(1,'coherent','best_native_atom','Best native\natom')],[(16,'coherent','native64','Coherent\n16 source'),(16,'random','native64','Random\n16 source'),(16,'coherent','marginal64','Coherent\nmarginal-64')]]
 for a,panel,title in zip(ax,panels,['Fine: output class matters','Coarse: source grouping matters']):
  for s,c in zip(range(1,6),colors):
   y=[median(r['kl'] for r in agg if (r['size'],r['kind'],r['method'],r['source_seed'])==(z,k,m,s)) for z,k,m,_ in panel];a.plot(range(len(panel)),y,'o-',color=c,label=f'Source {s}',alpha=.8,linewidth=1)
  a.set_xticks(range(len(panel)),[q[3] for q in panel]);a.set_ylim(0,1.08);a.set_ylabel('Median relative KL on new natural inputs');a.set_title(title);a.grid(axis='y',alpha=.18)
 ax[0].legend(fontsize=7,loc='lower right');fig.suptitle('Frozen maps: 40 source queries, 80 donor pairs, all 20 seed directions',fontsize=12);fig.tight_layout();save(fig,'figure3_natural_operations')
 print(json.dumps(dict(counts={k:payload[k] for k in ['natural_queries','natural_pairs','natural_distinct_documents','missing_requests']},contrasts=contrasts),indent=2))
if __name__=='__main__':main()
