"""Mechanism/instance diagnostic figure; no inference from plotting quality."""
import os,json,hashlib
from pathlib import Path
os.environ['MPLBACKEND']='Agg'
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'.aris/mplconfig')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main():
 out=ROOT/'artifacts/contribution_structure_20260906'
 p=out/'data.json';data=json.loads(p.read_text(encoding='utf-8'));summ=data['summary'];inst=data['instance']
 colors=['#0072B2','#777777','#B35400'];names=['Source-coherent','Uniform random','Nuisance-matched random'];kinds=['coherent','random','matched_random']
 with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'}):
  fig=plt.figure(figsize=(13.0,9.8),facecolor='white')
  gs=fig.add_gridspec(3,3,left=.065,right=.94,bottom=.13,top=.87,height_ratios=[.78,1.5,1],hspace=.73,wspace=.48)
  fig.text(.045,.965,'What did coherent grouping buy?',fontsize=20,weight='bold',color='#172938')
  fig.text(.045,.935,'A source-only matched control removes most of the original advantage',fontsize=12,color='#49555E')
  a=fig.add_subplot(gs[0,:]);a.axis('off')
  a.text(-.02,1.15,'A',transform=a.transAxes,weight='bold',fontsize=12)
  boxes=[(.0,.28,'SOURCE QUERY','16 contribution members\n'+r'$Y(x)=\sum_{i\in I} w_i z_i(x)d_i$'),(.35,.28,'CONTROL CONSTRUCTION','Same anchor; source-only matching\nenergy, frequency, spectrum, raw PCs'),(.70,.28,'FIXED TARGET OPERATION','64 target decoder contributions\nSame signed fit; held-out error')]
  for x,y,h,t in boxes:
   a.add_patch(FancyBboxPatch((x,y),.29,.69,boxstyle='round,pad=0.012',linewidth=.8,edgecolor='#C8D1D7',facecolor='#F5F8FA',transform=a.transAxes,clip_on=False))
   a.text(x+.015,y+.54,h,fontsize=9,weight='bold',transform=a.transAxes,color='#172938')
   a.text(x+.015,y+.37,t,fontsize=9,va='top',linespacing=1.7,transform=a.transAxes)
  for x in [.30,.65]:a.annotate('',xy=(x+.045,.62),xytext=(x,.62),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#657681',lw=1.2))
  a.text(0,-.17,'Example below: first frozen anchor, seed 1 / feature 1155. Selected before transfer outcomes.',transform=a.transAxes,fontsize=9)
  a.text(0,-.39,'Natural high-activation prefixes: “The step by” and “With every new album ... era after”. These are observations, not concept labels.',transform=a.transAxes,fontsize=9,color='#49555E')
  heat=[]
  limit=max(abs(np.array(g['correlation'])[~np.eye(16,dtype=bool)]).max() for g in inst['groups'])
  for j,(g,name,col) in enumerate(zip(inst['groups'],names,colors)):
   ax=fig.add_subplot(gs[1,j]);matrix=np.array(g['correlation']);np.fill_diagonal(matrix,np.nan)
   cm=plt.colormaps['RdBu_r'].copy();cm.set_bad('#EEEEEE')
   im=ax.imshow(matrix,vmin=-limit,vmax=limit,cmap=cm,interpolation='nearest',aspect='equal')
   labels=[f"{r['atom']}: {r['top_tokens'][0].strip() if r['top_tokens'][0].strip() else '[space]'}" for r in g['members']]
   labels=[s.replace('<|endoftext|>','[EOS]') for s in labels]
   ax.set_yticks(range(16),labels,fontsize=7);ax.set_xticks([0,4,8,12,15],[1,5,9,13,16],fontsize=8)
   ax.set_xlabel('Source member index',fontsize=8);ax.tick_params(length=0)
   ax.set_title(name+f"\nenergy {g['energy']:.3f} | effective rank {g['effective_rank']:.2f}",fontsize=10,color=col,pad=12)
   ax.text(-.25,1.16,chr(ord('B')+j),transform=ax.transAxes,weight='bold',fontsize=12)
   heat.append(ax)
  cbax=fig.add_axes([.95,.445,.009,.16]);cbar=fig.colorbar(im,cax=cbax,ticks=[-.03,0,.03]);cbar.ax.tick_params(labelsize=7);cbar.ax.set_title('r',fontsize=8)
  ax=fig.add_subplot(gs[2,0]);ax.text(-.17,1.16,'E',transform=ax.transAxes,weight='bold',fontsize=12)
  for g,name,col,style in zip(inst['groups'],names,colors,['-','--',':']):
   spec=np.array(g['spectrum']);ax.plot(np.arange(1,17),spec.cumsum()/spec.sum(),style,color=col,lw=1.8,label=name)
  ax.set(xlim=(1,16),ylim=(0,1.025),xlabel='Source covariance eigenvalue index',ylabel='Cumulative fraction of energy',title='Same example: matched source spectrum')
  ax.set_xticks([1,4,8,12,16]);ax.legend(fontsize=7,loc='lower right',frameon=False);ax.grid(axis='y',alpha=.15)
  ax=fig.add_subplot(gs[2,1]);ax.text(-.17,1.16,'F',transform=ax.transAxes,weight='bold',fontsize=12)
  for kind,name,col,style in zip(kinds,names,colors,['-','--',':']):
   vals=np.sort([r['methods']['native64']['relative_error'] for r in data['all_rows'] if r['kind']==kind]);ax.plot(vals,np.arange(1,len(vals)+1)/len(vals),style,color=col,lw=1.8)
   med=summ['method_medians'][kind]['native64'];ax.axvline(med,color=col,lw=.7,alpha=.4)
   ax.text(.035,.92-.105*kinds.index(kind),name+f': {med:.3f}',transform=ax.transAxes,fontsize=7.5,color=col)
  ax.set(xlim=(0,1.05),ylim=(0,1),xlabel='Relative native-64 contribution error',ylabel='Fraction of query–target cells',title='All anchors; no transfer-based exclusion')
  ax.grid(alpha=.15)
  ax=fig.add_subplot(gs[2,2]);ax.text(-.17,1.16,'G',transform=ax.transAxes,weight='bold',fontsize=12)
  for kind,col,label,marker in [('random',colors[1],'versus uniform random','o'),('matched_random',colors[2],'versus matched random','s')]:
   ys=[summ['per_source'][str(s)][kind]['paired_median_difference'] for s in range(1,6)]
   ax.plot(range(1,6),ys,marker+'-',color=col,label=label,lw=1.2,markersize=5)
  ax.axhline(0,color='#555555',lw=.8);ax.set(xlabel='Source seed',ylabel='Coherent error minus control error',title='The original gap shrinks in every seed')
  ax.set_xticks(range(1,6));ax.legend(fontsize=7,frameon=False);ax.grid(axis='y',alpha=.15)
  fig.text(.045,.047,'16 members / source group; 64 target members. All 160 anchors and 20 directed seed pairs retained. Shared seeds and documents are dependent.',fontsize=8,color='#49555E')
  fig.text(.045,.025,'B–D: diagonal omitted (always 1); labels show tokens with most activation mass, not semantic annotations. Matching is approximate; all balance errors are reported.',fontsize=8,color='#49555E')
  for ext in ['png','pdf','svg']:fig.savefig(out/f'figure_control_mechanism.{ext}',dpi=240,facecolor='white')
  plt.close(fig)
 manifest=dict(source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   selection=inst['selection'],figure_dimensions_inches=[13,9.8],dpi=240,purpose='Internal scientific diagnostic, not submission-ready claim',
   normalization=['group spectra divided by own energy','relative error divided by corresponding source covariance energy','diagonal correlation masked consistently and labeled','common symmetric correlation scale uses maximum absolute off-diagonal across all three displayed groups; no clipping'],
   statistics='ECDF of dependent cells and five per-source paired medians; no independent-direction confidence intervals',
   missing='No source queries dropped; diagonal correlations omitted by definition',
   alt_text='The first frozen source group and its two random controls show distinct member relations. Matching source energy, spectrum, frequency and raw-PC content raises the random-control performance nearly to coherent grouping. The paired error gap shrinks in each of five source seeds.')
 (out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2))
 print(out/'figure_control_mechanism.png')
if __name__=='__main__':main()
