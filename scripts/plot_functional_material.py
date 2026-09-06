"""Vector, data-backed material and intervention figures (provisional manuscript size)."""
import hashlib,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.aris/mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch,Rectangle

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/seven_round_rebuild_20260906'
S=json.loads((OUT/'material_summary.json').read_text(encoding='utf-8'))
BLUE='#245881';TEAL='#087F82';ORANGE='#BC631F';GRAY='#66717A';INK='#172B3A';LIGHT='#EDF3F6'
SEQ=LinearSegmentedColormap.from_list('paper_teal',['#F4F7F8','#72B8B6','#075D61'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
    'xtick.labelsize':7.5,'ytick.labelsize':7.5,'axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':GRAY,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
    'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

def title(ax,letter,text):
    ax.set_title(letter+'  '+text,loc='left',fontweight='bold',fontsize=9,pad=21)

def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=220)
    plt.close(fig)

def row(model,layer,site,factor,template):
    return next(r for r in S['patches'] if (r['model'],r['layer'],r['site'],r['factor'],r['template'])==(model,layer,site,factor,template))

def main():
    fig=plt.figure(figsize=(178/25.4,173/25.4))
    gs=fig.add_gridspec(3,2,left=.085,right=.96,top=.93,bottom=.17,height_ratios=[1.0,1.12,1.28],hspace=.57,wspace=.5)
    a=fig.add_subplot(gs[0,:]);a.set_xlim(0,1);a.set_ylim(0,1);a.axis('off')
    title(a,'a','A variable and its distractor, with the same words')
    a.text(.01,.88,'Subject',color=BLUE,fontweight='bold');a.text(.22,.88,'Distractor',color=ORANGE,fontweight='bold')
    a.text(.39,.88,'Actual prompt');a.text(.995,.88,'log P(are) / P(is)',ha='right')
    baseline={r['row_id']:r for r in S['example']['baseline']}
    for k,r in enumerate(S['example']['inputs']):
        y=.68-k*.205;b=baseline[r['id']]
        if k%2==0:a.add_patch(Rectangle((0,y-.08),1,.18,fc=LIGHT,ec='none',zorder=-1))
        a.text(.01,y,'plural' if r['subject_number'] else 'singular',color=BLUE,va='center')
        a.text(.22,y,'plural' if r['distractor_number'] else 'singular',color=ORANGE,va='center')
        a.text(.39,y,r['text'].replace('<|endoftext|>',''),va='center',fontsize=7.7)
        a.text(.995,y,f"{b['plural_margin']:+.2f}",ha='right',va='center',fontweight='bold',color=BLUE if b['plural_margin']<0 else TEAL)
    b=fig.add_subplot(gs[1,:]);b.axis('off');b.set_xlim(0,1);b.set_ylim(0,1)
    title(b,'b','A donor vector changes the answer at the subject position')
    b.text(.01,.86,'Donor',color=TEAL,va='center',fontweight='bold')
    b.text(.01,.37,'Recipient',color=BLUE,va='center',fontweight='bold')
    xs=[.185,.345,.525,.71];ws=[.09,.18,.14,.15]
    for y,words,c in [(.86,['The','authors','near the','teacher'],TEAL),(.37,['The','author','near the','teacher'],BLUE)]:
        for i,(x,w,word) in enumerate(zip(xs,ws,words)):
            b.add_patch(Rectangle((x-w/2,y-.10),w,.20,fc='#E5F3F1' if i==1 else '#F1F4F6',ec=c if i==1 else '#B7C4CC',lw=1))
            b.text(x,y,word,ha='center',va='center',color=c if i==1 else INK)
    b.add_patch(FancyArrowPatch((.345,.735),(.345,.48),arrowstyle='-|>',mutation_scale=11,color=TEAL,lw=1.6))
    b.text(.365,.605,'Replace resid_post at layer 3',fontsize=7.6,va='center')
    base=baseline[0]['plural_margin']
    pp=next(r for r in S['example']['patches'] if r['layer']==3 and r['factor']=='subject' and r['site']=='changed_region_end')
    last=next(r for r in S['example']['patches'] if r['layer']==3 and r['factor']=='subject' and r['site']=='last_region_end')
    b.text(.86,.86,'are',fontweight='bold',color=TEAL,ha='center')
    b.text(.86,.37,'are',fontweight='bold',color=TEAL,ha='center')
    b.text(.96,.37,f"{base+pp['plural_margin_change']:+.2f}",ha='right',va='center',fontweight='bold',color=TEAL)
    b.text(.01,.03,f"Recipient: {base:+.2f} (is)     Subject patch: {base+pp['plural_margin_change']:+.2f} (are)     Final-token patch: {base+last['plural_margin_change']:+.2f} (is)",fontsize=7.8)
    c=fig.add_subplot(gs[2,0]);title(c,'c','Patch location matters')
    templates=['pp','subject_relative','object_relative']
    vals=np.array([[row('1b',l,site,'subject',t)['correct'] for site in ['changed_region_end','last_region_end'] for t in templates] for l in [3,7,11]])
    im=c.imshow(vals,vmin=0,vmax=64,cmap=SEQ,aspect='auto',interpolation='nearest')
    for i in range(3):
        for j in range(6):c.text(j,i,str(vals[i,j]),ha='center',va='center',fontsize=8,color='white' if vals[i,j]>39 else INK)
    c.axvline(2.5,color='white',lw=3)
    c.set_xticks(range(6),['PP','SR','OR','PP','SR','OR']);c.set_yticks(range(3),['Layer 3','Layer 7','Layer 11'])
    c.text(.23,1.005,'Subject token',transform=c.transAxes,ha='center',fontsize=7.5)
    c.text(.77,1.005,'Final token',transform=c.transAxes,ha='center',fontsize=7.5)
    c.set_xlabel('Successful donor-label transfers / 64',labelpad=8)
    d=fig.add_subplot(gs[2,1]);title(d,'d','Off-target output change')
    leakage=np.array([[row('1b',l,'changed_region_end','distractor',t)['mean_abs_margin_change'] for t in templates] for l in [3,7,11]])
    di=d.imshow(leakage,vmin=0,vmax=max(3,float(leakage.max())),cmap='Oranges',aspect='auto',interpolation='nearest')
    for i in range(3):
        for j in range(3):
            r=row('1b',[3,7,11][i],'changed_region_end','distractor',templates[j])
            d.text(j,i,f"{leakage[i,j]:.2f}\n{r['correct']}/64",ha='center',va='center',fontsize=7.8,color=INK)
    d.set_xticks(range(3),['PP','SR','OR']);d.set_yticks(range(3),['Layer 3','Layer 7','Layer 11'])
    d.set_xlabel('Mean |log-odds change| (nats)\nand retained correct answers',labelpad=8)
    fig.text(.085,.025,'Pythia 1B · 16 lexical blocks × 3 syntax frames × 4 number combinations. Pair directions share inputs.\nPP: prepositional phrase; SR / OR: subject / object relative clause. All answers are is-versus-are comparisons.',fontsize=7,color=GRAY)
    save(fig,'figure_material_intervention')

    fig=plt.figure(figsize=(178/25.4,191/25.4))
    ax=fig.add_axes([.47,.16,.25,.74]);tasks=sorted({r['task'] for r in S['public_tasks']})
    table={(r['task'],r['model']):r for r in S['public_tasks']}
    values=np.array([[table[t,m]['accuracy'] for m in ['160m','1b']] for t in tasks])
    im=ax.imshow(values,vmin=0,vmax=1,cmap=SEQ,aspect='auto',interpolation='nearest')
    for i in range(len(tasks)):
        for j in range(2):ax.text(j,i,f'{values[i,j]*100:.0f}',ha='center',va='center',fontsize=7.7,color='white' if values[i,j]>.61 else INK)
    ax.set_yticks(range(len(tasks)),[t.replace('_',' ') for t in tasks]);ax.tick_params(length=0)
    ax.set_xticks([0,1],['160M','1B']);ax.xaxis.tick_top();ax.set_title('Correct label (%)',pad=25)
    for y in np.arange(.5,len(tasks),1):ax.axhline(y,color='white',lw=.7)
    dx=fig.add_axes([.79,.16,.13,.74],sharey=ax)
    delta=100*(values[:,1]-values[:,0]);dx.axvline(0,c='#9BA9B2',lw=.8)
    dx.scatter(delta,np.arange(len(tasks)),c=TEAL,s=15,zorder=3)
    dx.set_xlim(min(-10,float(delta.min())-3),max(45,float(delta.max())+3));dx.set_xticks([0,20,40]);dx.set_xlabel('1B − 160M\npercentage points');dx.tick_params(axis='y',left=False,labelleft=False)
    dx.spines['left'].set_visible(False);dx.set_title('Model difference',pad=25)
    fig.text(.075,.975,'Model competence across all 29 released development tasks',fontweight='bold',fontsize=10)
    fig.text(.075,.04,'CausalGym dev: 100 released pairs per task, evaluated in both directions (200 recorded sides).\nReciprocal prompt reuse is retained; these are descriptive task totals, not 200 independent trials.\nFixed revisions; no test data. Overall: 160M 4290/5800 (74.0%); 1B 4978/5800 (85.8%).',fontsize=7.6,color=GRAY)
    save(fig,'figure_public_task_material')
    paths=list(OUT.glob('figure_material_intervention.*'))+list(OUT.glob('figure_public_task_material.*'))
    (OUT/'figure_manifest.json').write_text(json.dumps(dict(source_data_sha256=hashlib.sha256((OUT/'material_summary.json').read_bytes()).hexdigest(),
        generator='scripts/plot_functional_material.py',physical_width_mm=178,venue_template='Provisional; venue not selected',
        figures=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]),indent=2)+'\n')
    print(json.dumps(dict(figures=[str(p) for p in paths])))

if __name__=='__main__':main()
