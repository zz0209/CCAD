"""Paper-sized role-composition figure and tables from completed confirmation."""
from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'paper';ART=ROOT/'artifacts/correspondence_reform_20260913'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    r=json.loads((ART/'r56_role_confirmation_analysis.json').read_text());loc=json.loads((ART/'r56_context_role_analysis.json').read_text())
    methods=['all_context','source64','target64','nearest64','direct64']
    labels=['Raw role program','Source code response','Target recoding','Nearest decoder','Direct target recoding']
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    with plt.rc_context({'font.family':'Times New Roman','font.size':8.5,'mathtext.fontset':'stix','axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig=plt.figure(figsize=(6.8,2.9));a=fig.add_axes([.015,.17,.40,.66]);a.axis('off');a.set(xlim=(0,1),ylim=(0,1))
        a.text(0,1.15,'(a) Roles across the context layers',fontsize=9)
        for x,label in zip([.57,.74,.92],['Entity','Attribute','Joint']):a.text(x,.94,label,ha='center')
        a.plot([0,1],[.84,.84],color='#444444',lw=.6)
        for y,key,label in zip([.68,.46,.24],['all_context','existing_pair','all_except_pair'],['All layers','SAE locations','Other layers']):
            a.text(0,y,label,va='center')
            for x,op in zip([.57,.74,.92],['entity','attribute','both']):
                v=loc['methods'][key][op+'_complete']['percent'];a.text(x,y,f'{v:.1f}',ha='center',va='center',color='#202020' if v>1 else '#777777')
        a.plot([0,1],[.10,.10],color='#444444',lw=.6)
        a.text(0,-.03,'Both-city rule accuracy (%)',fontsize=8)
        b=fig.add_axes([.705,.26,.26,.64]);b.text(-.77,1.05,'(b) All three rules after SAE retraining',transform=b.transAxes,fontsize=9)
        for i,m in enumerate(methods):
            for off,g,color,marker,label in [(-.11,'known_forms','#555555','o','Known forms'),(.11,'new_forms','#286956','D','New forms')]:
                v=r['results'][g][m]['all_three'];lo,hi=v['ci95'];b.plot([lo,hi],[i+off]*2,lw=.85,color=color);b.plot(v['percent'],i+off,marker,ms=3.1,color=color,label=label if i==0 else None)
        b.set(yticks=range(5),yticklabels=labels,xlim=(40,101),ylim=(4.6,-.6),xticks=[40,60,80,100],xlabel='Both cities, all three rules (%)')
        b.tick_params(axis='y',length=0,pad=4,labelsize=8);b.spines[['top','right','left']].set_visible(False);b.grid(axis='x',color='#dddddd',lw=.5);b.set_axisbelow(True)
        handles,legend_labels=b.get_legend_handles_labels()
        fig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.77,.015),ncol=2,frameon=False,fontsize=8,handlelength=1,columnspacing=1.2)
        for ext in ['pdf','svg','png']:fig.savefig(P/f'figures/binding_roles.{ext}',dpi=240,facecolor='white')
        plt.close(fig)
    t=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrrrrrr}',r'\toprule',r' & \multicolumn{3}{c}{Complete rule, all forms} & \multicolumn{3}{c}{All three rules} \\',r'Method & Entity & Attribute & Joint & Known & New & All \\',r'\midrule']
    for m,label in zip(methods,labels):
        v=r['results']['all_forms'][m];vals=[v[op]['percent'] for op in ['entity','attribute','both']]+[r['results'][g][m]['all_three']['percent'] for g in ['known_forms','new_forms','all_forms']]
        t.append(label+' & '+' & '.join(f'{x:.2f}' for x in vals)+r' \\')
    t += [r'\bottomrule\end{tabular}',r'\caption{Frozen role-composition confirmation: 32 new worlds, two orderings, four forms and five dependent SAE directions. Every single-rule score requires both city answers; all-three requires all six answers within the same context. Updates at the two SAE locations use at most 64 coefficients per changed state. Other layers retain the common raw program.}',r'\label{tab:binding_roles}',r'\end{anchoredtable}']
    (P/'tables/binding_roles.tex').write_text('\n'.join(t)+'\n')
    data=P/'data/binding_roles.json';data.write_text(json.dumps(dict(confirmation=r,localization=loc),indent=2)+'\n')
    with (P/'data/binding_roles.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['forms','method','endpoint','percent','ci95_low','ci95_high'])
        for g,gg in r['results'].items():
            for m,mm in gg.items():
                for op,v in mm.items():wr.writerow([g,m,op,v['percent'],*v['ci95']])
    commands={'BindingRoleSource':r['results']['all_forms']['source64']['all_three']['percent'],'BindingRoleTarget':r['results']['all_forms']['target64']['all_three']['percent'],'BindingRoleNearest':r['results']['all_forms']['nearest64']['all_three']['percent'],'BindingRoleDirect':r['results']['all_forms']['direct64']['all_three']['percent'],'BindingRoleGain':r['contrasts']['all_forms']['nearest64']['percent']}
    (P/'tables/binding_role_values.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+f'{v:.2f}'+'}' for k,v in commands.items())+'\n')
    fm=P/'figures/FIGURE_MANIFEST.json';manifest=json.loads(fm.read_text());paths=['figures/binding_roles.'+e for e in ['pdf','svg','png']]
    manifest['outputs']=[x for x in manifest['outputs'] if x['path'] not in paths]+[dict(path=x,bytes=(P/x).stat().st_size,sha256=hashlib.sha256((P/x).read_bytes()).hexdigest()) for x in paths]
    manifest.setdefault('additional_sources',{})['binding_roles']=dict(path='data/binding_roles.json',sha256=hashlib.sha256(data.read_bytes()).hexdigest(),generator='scripts/binding_role_paper.py');fm.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(commands))


if __name__=='__main__':main()
