"""Within-task response checks and a compact development figure for SCIENCE01."""
from pathlib import Path
from datetime import datetime, timezone
import json, csv, hashlib, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/science_upgrade_20260919'
FIRST=ROOT/'runs/SCIENCE01_response_space_tasks_v1_20260919'
SECOND=Path('D:/CCAD_Storage/runs/science_upgrade_20260919/SCIENCE01_response_continuous_tasks_v2_20260919')
METHODS=['initial','head_parts','pooled_parts','white_parts','pooled_whole',
         'head_continuous','pooled_continuous','geometry','geometry_gain','raw','raw_reconstruction']
FAMILIES={'Single parts':['pronouns','names','associated_words'],
          'Complete and pairs':['full','pronouns+names','pronouns+associated_words','names+associated_words'],
          'Fractional requests':['mix_balanced','mix_pronouns','mix_names','mix_words']}

def main():
    output=OUT/'WITHIN_TASK_ANALYSIS.json'
    if output.exists(): raise FileExistsError(output)
    provenance={}; cache={}
    def arr(name):
        if name not in cache:
            p=next(p/name for p in [FIRST,SECOND] if (p/name).exists())
            provenance[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            cache[name]=np.load(p).astype(np.float64)
        return cache[name]
    rows=json.loads((FIRST/'evaluation_membership.json').read_text())['rows']
    assert rows==json.loads((SECOND/'evaluation_membership.json').read_text())['rows']
    for query in sum(FAMILIES.values(),[]):
        a=np.load(FIRST/f'source__{query}__pooled.npy')
        b=np.load(SECOND/f'source__{query}__pooled.npy')
        assert np.allclose(a,b,atol=2e-5,rtol=1e-5)
    clean=arr('none__full__pooled.npy'); cells=[]; replicates={}
    taskdef=[('composer_surgeon',(5,25)),('model_software_engineer',(12,24))]
    rng=np.random.default_rng(20260919)
    # The same stratified document bootstrap is reused across all methods,
    # requests and the two fixed orientations of each profession pair.
    for task,professions in taskdef:
        ix=np.array([i for i,r in enumerate(rows) if r['profession'] in professions])
        buckets=[np.array([j for j,i in enumerate(ix) if rows[i]['profession']==p and rows[i]['gender']==g])
                 for p in professions for g in [0,1]]
        boot=np.concatenate([rng.choice(v,size=(2000,len(v)),replace=True) for v in buckets],axis=1)
        for orientation in [0,1]:
            head=f'{task}_orientation{orientation}'
            p=ROOT/'runs/IR04_shift_consumer_seed2_v1_20260916'/f'none__full__{head}__probe42.npz'
            provenance[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            w=np.load(p)['weight'].ravel().astype(np.float64)
            for family,queries in FAMILIES.items():
                for method in METHODS:
                    scores=[]; bs=[]
                    for query in queries:
                        source=arr(f'source__{query}__pooled.npy')[ix]
                        target=arr(f'{method}__{query}__pooled.npy')[ix]
                        numer=((target-source)@w)**2
                        denom=((source-clean[ix])@w)**2
                        scores.append(float(np.sqrt(numer.mean()/denom.mean())))
                        bs.append(np.sqrt(numer[boot].mean(1)/denom[boot].mean(1)))
                    cells.append(dict(head=head,family=family,method=method,nrmse=float(np.mean(scores)),documents=len(ix)))
                    replicates[head,family,method]=np.mean(bs,axis=0)
    summary=[]; differences=[]
    heads=sorted({r['head'] for r in cells})
    for family in FAMILIES:
        for method in METHODS:
            value=float(np.mean([r['nrmse'] for r in cells if r['family']==family and r['method']==method]))
            summary.append(dict(family=family,method=method,nrmse=value))
            if method!='head_parts':
                diff=np.mean([replicates[h,family,method]-replicates[h,family,'head_parts'] for h in heads],axis=0)
                differences.append(dict(family=family,method=method,reference='head_parts',
                                        interval=np.quantile(diff,[.025,.975]).tolist()))
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
                evidence='Previously exposed development, one target seed, four fixed task readouts; each head uses only its profession pair',
                intervals='2000 paired profession/gender-stratified document resamples; no head/seed population inference',
                summary=summary,cells=cells,differences=differences,inputs=provenance)
    output.write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'WITHIN_TASK_ANALYSIS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['family','method','nrmse']);writer.writeheader();writer.writerows(summary)
    print(json.dumps(summary,indent=2))

def plot():
    summary=json.loads((OUT/'WITHIN_TASK_ANALYSIS.json').read_text())['summary']
    sys.path.insert(0,str(ROOT/'.aris/plot_runtime_v1'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'mathtext.fontset':'stix',
                         'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    show=['initial','head_parts','pooled_parts','head_continuous','pooled_continuous','raw_reconstruction']
    labels=['Initial relation','Old head / parts','Full response / parts','Old head / continuous',
            'Full response / continuous','Reconstruction readout']
    colors=['#777777','#784352','#24746b','#784352','#24746b','#191919']
    fig,axes=plt.subplots(1,3,figsize=(7.05,2.65),sharey=True)
    for ax,family in zip(axes,FAMILIES):
        for i,(m,col) in enumerate(zip(show,colors)):
            value=next(r['nrmse'] for r in summary if r['method']==m and r['family']==family)
            ax.plot(value,i,'D' if 'continuous' in m else 'o',color=col,markersize=4)
            ax.annotate(f'{value:.3f}',(value,i),xytext=(5,0),textcoords='offset points',va='center',fontsize=7)
        ax.set_title(family,fontsize=9,fontweight='normal')
        ax.set_xlim(0,.39);ax.set_xticks([0,.1,.2,.3]);ax.grid(axis='x',color='#dddddd',lw=.4)
        ax.set_xlabel('Response nRMSE (lower is better)')
    axes[0].set_yticks(range(len(show)),labels);axes[0].invert_yaxis()
    fig.subplots_adjust(left=.26,right=.99,bottom=.21,top=.83,wspace=.18)
    fig.savefig(OUT/'request_response_coverage.pdf')
    fig.savefig(OUT/'request_response_coverage.png',dpi=200)

if __name__=='__main__':
    plot() if '--plot-only' in sys.argv else main()
