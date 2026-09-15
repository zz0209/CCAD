"""Export the frozen member-choice results at paper size."""
from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'paper';ART=ROOT/'artifacts/correspondence_reform_20260913'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    r=json.loads((ART/'r57_member_confirmation_analysis.json').read_text());g=r['results']['all_forms']
    labels={'geometry':'Contribution magnitude','source_cached':'Cached source gradient','source_fullgrad':'Source endpoint gradient','source_path':'Reused source path','source_pooled':'Pooled source path','raw_path':'Reused raw path','target_path':'New target path','source_selected':'Selected source','source_full':'Full source','target_full':'Full target'}
    commands={f'BindingMember{name}':g['methods'][method+'/k16']['complete']['percent'] for name,method in [('Source','source_path'),('Geometry','geometry'),('Cached','source_cached'),('Raw','raw_path'),('Pooled','source_pooled'),('Direct','target_path')]}
    commands['BindingMemberFull']=g['methods']['target_full/k64']['complete']['percent']
    commands['BindingMemberGain']=g['contrasts']['source_path/k16 - geometry/k16']['complete']['percent']
    (P/'tables/binding_member_values.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+f'{v:.2f}'+'}' for k,v in commands.items())+'\n')
    table=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrrrrrr}\toprule',r'Method & Members & Entity & Restoration & Complete & Fit forms & Held-out forms \\\midrule']
    for key,v in g['methods'].items():
        m,k=key.split('/k');values=[v[e]['percent'] for e in ['entity','restoration','complete']]+[r['results'][f]['methods'][key]['complete']['percent'] for f in ['fit_forms','heldout_forms']]
        table.append(labels[m]+' & '+k+' & '+' & '.join(f'{x:.2f}' for x in values)+r' \\')
    table += [r'\bottomrule\end{tabular}',r'\caption{Frozen member selection on 32 new worlds and four target SAE dictionaries. Members are a per-state allowance at each of two entity-role locations, with an input-dependent 64-candidate bank. Entity and restoration each require both cities; complete requires both operations.}',r'\label{tab:binding_member_selection}',r'\end{anchoredtable}']
    (P/'tables/binding_member_selection.tex').write_text('\n'.join(table)+'\n')
    table=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrrrr}\toprule',r'Method & Target 2 & Target 3 & Target 4 & Target 5 \\\midrule']
    for m in ['geometry','source_cached','source_path','source_pooled','raw_path','target_path']:
        vals=[a['methods'][m+'/k16']['complete']['percent'] for a in r['per_target'].values()]
        table.append(labels[m]+' & '+' & '.join(f'{v:.2f}' for v in vals)+r' \\')
    table += [r'\bottomrule\end{tabular}',r'\caption{Complete rule accuracy at the frozen allowance of 16 members per changed state. All targets reuse source 1.}',r'\label{tab:binding_member_targets}',r'\end{anchoredtable}']
    (P/'tables/binding_member_targets.tex').write_text('\n'.join(table)+'\n')
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    with plt.rc_context({'font.family':'Times New Roman','font.size':8.5,'mathtext.fontset':'stix','axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(1,2,figsize=(6.8,2.75));fig.subplots_adjust(left=.08,right=.975,bottom=.27,top=.87,wspace=.43)
        styles=[('geometry','#787168','s'),('source_cached','#77566d','^'),('source_path','#286956','o')]
        a,b=axes
        for m,color,marker in styles:
            vals=[g['methods'][f'{m}/k{k}']['complete'] for k in [8,16,32]]
            xs=[0,1,2];ys=[v['percent'] for v in vals]
            a.plot(xs,ys,marker=marker,color=color,lw=.8,ms=3.8,label=labels[m])
            for x,v in zip(xs,vals):a.plot([x,x],v['ci95'],color=color,lw=.8)
        full=g['methods']['target_full/k64']['complete']['percent'];a.axhline(full,color='#333333',ls='--',lw=.7,label='Full target (64)')
        a.set(xticks=[0,1,2],xticklabels=['8','16','32'],xlabel='Changed members per state',ylabel='Complete rule accuracy (%)',ylim=(0,101),xlim=(-.12,2.12))
        a.set_title('(a) Functional member choice',loc='left',fontsize=9)
        for i,(name,analysis) in enumerate(r['per_target'].items()):
            for offset,m,color,marker in [(-.10,'geometry','#787168','s'),(.10,'source_cached','#77566d','^')]:
                v=analysis['contrasts'][f'source_path/k16 - {m}/k16']['complete'];b.plot(v['ci95'],[i+offset]*2,color=color,lw=.8);b.plot(v['percent'],i+offset,marker,color=color,ms=3.8)
        b.axvline(0,color='#333333',lw=.6);b.set(yticks=range(4),yticklabels=['Target 2','Target 3','Target 4','Target 5'],xlabel='Source-path gain at 16 members (points)',ylim=(3.5,-.5));b.set_title('(b) Reuse across retraining',loc='left',fontsize=9)
        for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='x' if ax==b else 'y',color='#dddddd',lw=.4);ax.set_axisbelow(True)
        h,l=a.get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,frameon=False,fontsize=8,bbox_to_anchor=(.5,-.01))
        for ext in ['pdf','svg','png']:fig.savefig(P/f'figures/binding_member_selection.{ext}',dpi=240,facecolor='white')
        plt.close(fig)
    data=P/'data/binding_member_selection.json';data.write_text(json.dumps(r,indent=2)+'\n')
    with (P/'data/binding_member_selection.csv').open('w',newline='') as stream:
        wr=csv.writer(stream);wr.writerow(['forms','method_budget','endpoint','percent','ci95_low','ci95_high'])
        for forms,analysis in r['results'].items():
            for m,v in analysis['methods'].items():
                for endpoint in ['entity','restoration','complete']:wr.writerow([forms,m,endpoint,v[endpoint]['percent'],*v[endpoint]['ci95']])
    path=P/'figures/FIGURE_MANIFEST.json';manifest=json.loads(path.read_text());newpaths=[f'figures/binding_member_selection.{e}' for e in ['pdf','svg','png']]
    manifest['outputs']=[x for x in manifest['outputs'] if x['path'] not in newpaths]+[dict(path=x,bytes=(P/x).stat().st_size,sha256=hashlib.sha256((P/x).read_bytes()).hexdigest()) for x in newpaths]
    manifest.setdefault('additional_sources',{})['binding_member_selection']=dict(path='data/binding_member_selection.json',sha256=hashlib.sha256(data.read_bytes()).hexdigest(),generator='scripts/binding_member_paper.py');path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(commands))


if __name__=='__main__':main()
