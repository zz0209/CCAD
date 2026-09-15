"""Export the human-decision reuse figure and complete development tables."""
from pathlib import Path
import os,json,csv,hashlib,datetime
os.environ['MPLBACKEND']='Agg'
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT/'paper';ART=ROOT/'artifacts/correspondence_reform_20260913'

def main():
    data=json.loads((ART/'r58_shift_reuse_analysis.json').read_text());results=data['results']
    (PAPER/'data/human_explanation_reuse.json').write_text(json.dumps(data,indent=2)+'\n')
    with (PAPER/'data/human_explanation_reuse.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['condition','request','method','profession_percent','worst_group_percent','gender_percent'])
        for run,rows in results.items():
            for cell,r in rows.items():
                method,q=cell.split('/');writer.writerow([run,q,method,*[100*r[k] for k in ('profession','worst_group','gender')]])
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    mpl.rcParams.update({'font.family':'Times New Roman','font.size':10,'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7,3.4));fig.subplots_adjust(left=.20,right=.98,bottom=.19,top=.76)
    series=[('Unedited','source','none/none','#906443','x'),('Published source','source','source/full','#222222','o'),('Geometry','curve_8192','geometry/full','#777777','^'),('Member relation','curve_8192','native/full','#216451','s'),('Source-direction readout','curve_8192','raw/full','#78548b','D')]
    for offset,(name,key,cell,color,marker) in zip(np.linspace(.28,-.28,5),series):
        row=results[key][cell];v=100*np.array(row['groups']);ci=100*np.array(row['group_ci'])
        ax.errorbar(v,np.arange(4)+offset,xerr=np.maximum(np.stack([v-ci[:,0],ci[:,1]-v]),0),fmt=marker,color=color,ms=4.5,lw=.75,capsize=2,label=name,mfc='white' if name=='Published source' else color)
    ax.set(yticks=np.arange(4),yticklabels=['Male professors','Female professors','Male nurses','Female nurses'],xlim=(0,102),ylim=(3.55,-.55),xlabel='Profession accuracy (%)',xticks=np.arange(0,101,20))
    ax.grid(axis='x',color='.9',lw=.5);ax.set_axisbelow(True)
    ax.legend(loc='lower center',bbox_to_anchor=(.42,1.02),ncols=2,frameon=False,fontsize=9,handletextpad=.5,columnspacing=1.3)
    for ext in ('pdf','svg','png'):fig.savefig(PAPER/f'figures/human_explanation_reuse.{ext}',dpi=240)
    plt.close(fig)
    rows=['\\begin{tabular}{llrrrr}\\toprule','Request & Measure & Source & Geometry & Relation & Raw \\\\','\\midrule']
    for query,label in [('full','All selected'),('pronouns','Pronouns'),('names','Names'),('associated_words','Associated words')]:
        for metric,mlab in [('profession','Profession'),('worst_group','Worst group')]:
            cells=[results['curve_8192'][m+'/'+query][metric]*100 for m in ('source','geometry','native','raw')]
            rows.append(f'{label if metric=="profession" else ""} & {mlab} & '+ ' & '.join(f'{x:.2f}' for x in cells)+' \\\\')
    rows+=['\\bottomrule\\end{tabular}'];(PAPER/'tables/human_explanation_reuse.tex').write_text('\n'.join(rows)+'\n')
    rows=['\\begin{tabular}{llrrrr}\\toprule','Material & Fitted quantity & Geometry & Gain & Relation & Raw \\\\','\\midrule']
    for run,label,obj in [('field_1m','1M pilot','Vector'),('token_1m','1M pilot','Token response'),('context_1m','1M pilot','Context response'),('curve_1024','1M curve','Context response'),('curve_4096','4M curve','Context response'),('curve_8192','8M curve','Context response')]:
        cells=[results[run][m+'/full'] for m in ('geometry','geometry_gain','native','raw')]
        rows.append(f'{label} & {obj} & '+' & '.join(f'{100*r["profession"]:.2f} / {100*r["worst_group"]:.2f}' for r in cells)+' \\\\')
    rows+=['\\bottomrule\\end{tabular}'];(PAPER/'tables/human_explanation_comparisons.tex').write_text('\n'.join(rows)+'\n')
    rows=['\\begin{tabular}{lrrrrrr}\\toprule',' & \\multicolumn{2}{c}{1M tokens} & \\multicolumn{2}{c}{4M tokens} & \\multicolumn{2}{c}{8M tokens} \\\\','Site & FVE & CE recovery & FVE & CE recovery & FVE & CE recovery \\\\','\\midrule']
    quality={(q['task'],q['step']):q for q in data['quality']}
    for site in dict.fromkeys(q['task'] for q in data['quality']):
        cells=[quality[site,step][metric] for step in (1024,4096,8192) for metric in ('fve','ce_recovery')]
        rows.append(site.replace('_','\\_')+' & '+' & '.join(f'{v:.3f}' for v in cells)+' \\\\')
    rows+=['\\bottomrule\\end{tabular}'];(PAPER/'tables/human_explanation_material.tex').write_text('\n'.join(rows)+'\n')
    paths=[PAPER/'data/human_explanation_reuse.json',PAPER/'data/human_explanation_reuse.csv',*[PAPER/f'figures/human_explanation_reuse.{ext}' for ext in ('pdf','svg','png')],*[PAPER/f'tables/{name}.tex' for name in ('human_explanation_reuse','human_explanation_comparisons','human_explanation_material')]]
    receipt=dict(written_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=str(ART/'r58_shift_reuse_analysis.json'),figure_width_inches=7,uncertainty=data['scope'],files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
    (PAPER/'data/HUMAN_EXPLANATION_MANIFEST.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt['files'][:2]))

if __name__=='__main__':main()
