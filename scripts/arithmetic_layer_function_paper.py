"""Summarize matched carry-site interventions and keep query/preservation separate."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import numpy as np
from analyze_arithmetic_functional_rules import main as analyze

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/correspondence_reform_20260913'
LAYERS = [7, 13, 18, 23]
METHODS = ['raw_initial', 'raw_fitted', 'carry_oracle']
CONDITIONS = ['same_answer_opposite_carry', 'same_carry_different_answer']


def main():
    analyses = {}; all_rows = []; costs = []; baseline_reference = None
    arrays = {}; metadata = {}; fits = []; readout = []; writer_norms = []
    for layer in LAYERS:
        run = ROOT / f'runs/REFORM_R55_carry_layer{layer}_v2_20260915'
        summary = json.loads((run / 'metrics.summary.json').read_text())
        assert summary['status'] == 'PASS'
        records = [json.loads(s) for s in (run / 'metrics.raw.jsonl').read_text().splitlines()]
        panel = json.loads((run / 'RULE_PANEL.json').read_text())
        baseline = {r['row_id']: (r['answer'], r['correct']) for r in records if r['kind'] == 'base'}
        assert len(baseline) == 917
        if baseline_reference is not None:
            assert baseline == baseline_reference
        baseline_reference = baseline
        meta = {p['component']: p for p in panel['canonical_pairs']}
        with np.load(run/'LAYER_FUNCTION.npz') as data:
            reader=data['reader'].copy();initial=data['initial'].copy();writer=data['writer'].copy()
        with np.load(run/'states.npz') as data:
            hidden=data['hidden'].copy()
        for arity in [2,3]:
            a=arity-2
            writer_norms.append(dict(layer=layer,arity=arity,initial=float(np.linalg.norm(initial[a])),
                fitted=float(np.linalg.norm(writer[a])),cap_fraction=float(np.linalg.norm(writer[a])/(4*np.linalg.norm(initial[a])))))
            for template in [0,1]:
                for condition in CONDITIONS:
                    pp=[p for p in panel['pairs'] if p['template']==template and p['condition']==condition
                        and int('c' in panel['rows'][p['recipient']])+2==arity]
                    q=np.array([np.dot(hidden[p['donor'],template]-hidden[p['recipient'],template],reader[a]) for p in pp])
                    truth=np.array([panel['rows'][p['donor']]['carry']-panel['rows'][p['recipient']]['carry'] for p in pp])
                    readout.append(dict(layer=layer,arity=arity,template=template,condition=condition,n=len(pp),
                        mean_absolute_error=float(np.abs(q-truth).mean()),rounded_accuracy=float((np.round(q)==truth).mean())))
        for arity in [2, 3]:
            out = ART / f'r55_layer{layer}_{arity}.json'
            analyze(run, out, [0], arity)
            result = json.loads(out.read_text())
            analyses[f'layer{layer}_arity{arity}'] = result
            pp = [p for p in panel['pairs'] if len([v for v in ['a','b','c'] if v in panel['rows'][p['recipient']]]) == arity]
            for condition in CONDITIONS:
                ids = sorted({p['component'] for p in pp if p['condition'] == condition})
                labels = [meta[i].get('carry_group', meta[i].get('stratum', 'all')) for i in ids]
                metadata[arity, condition] = ids, labels
                for cid in ids:
                    group = [p for p in pp if p['component'] == cid]
                    assert len(group) == 4 and {p['template'] for p in group} == {0,1}
                    questions = lambda index: tuple(panel['rows'][index][v] for v in ['a','b','c'] if v in panel['rows'][index])
                    orientations = {(questions(p['recipient']), questions(p['donor'])) for p in group}
                    assert len(orientations) == 2 and all((b,a) in orientations for a,b in orientations)
                with np.load(out.with_suffix('.npz')) as data:
                    for method in METHODS:
                        arrays[layer, arity, method, condition] = data[method + '|' + condition][:,0].copy()
            for method in METHODS:
                row = dict(layer=layer, arity=arity, method=method)
                for condition, label in zip(CONDITIONS, ['change','preserve']):
                    cell = next(c for c in result['cells'] if c['method']==method and c['condition']==condition)
                    row[label] = 100*cell['success']['mean']
                    row[label+'_interval'] = [100*v for v in cell['success']['interval']]
                all_rows.append(row)
        costs.append(dict(run=run.relative_to(ROOT).as_posix(),status=json.loads((run/'status.json').read_text()),
            **{k:summary[k] for k in ['wall_seconds','process_cpu_seconds','peak_allocated_bytes','sequence_forwards','token_forwards','rows']}))
        fits.append(json.loads((run/'LAYER_FIT.json').read_text()))
    contrasts=[]
    for arity in [2,3]:
        for condition in CONDITIONS:
            ids,labels=metadata[arity,condition];rng=np.random.default_rng(9491502)
            weights=np.zeros((10000,len(ids)))
            for label in sorted(set(labels)):
                jj=[i for i,x in enumerate(labels) if x==label];n=len(jj)
                weights[:,jj]=rng.multinomial(n,np.full(n,1/n),10000)/len(ids)
            comparisons=[((layer,method),(23,method)) for layer in LAYERS[:-1] for method in METHODS]
            comparisons += [((layer,a),(layer,b)) for layer in LAYERS for a,b in [('raw_fitted','raw_initial'),('carry_oracle','raw_fitted')]]
            for (la,ma),(lb,mb) in comparisons:
                delta=100*(arrays[la,arity,ma,condition]-arrays[lb,arity,mb,condition])
                contrasts.append(dict(arity=arity,condition=condition,layer=la,method=ma,reference_layer=lb,reference=mb,
                    difference_points=float(delta.mean()),interval_points=np.quantile(weights@delta,[.025,.975]).tolist()))
    failed=ROOT/'runs/REFORM_R55_carry_layer7_v1_20260915'
    fs=json.loads((failed/'metrics.summary.json').read_text())
    costs.append(dict(run=failed.relative_to(ROOT).as_posix(),status=json.loads((failed/'status.json').read_text()),
        **{k:fs[k] for k in ['wall_seconds','process_cpu_seconds','peak_allocated_bytes','sequence_forwards','token_forwards','rows']}))
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),table=all_rows,analyses=analyses,contrasts=contrasts,
        fits=fits,costs=costs,readout=readout,writer_norms=writer_norms,baseline_replay='All 917 unedited answers agree across the four highest-precision sites.',
        canonical_identity='Statistics use component and carry-stratum labels and verified reciprocal question identities. Original canonical metadata donor/recipient integers retain R51 coordinates; actual R55 request indices are in pairs.',
        total_driver_seconds=sum(c['wall_seconds'] for c in costs),total_process_cpu_seconds=sum(c['process_cpu_seconds'] for c in costs),
        scope='Four exposed source-development locations, identical 405 fit questions and 256 pair batches. Raw rank-one writes; no SAE or target dictionary. Known carry is a labelled diagnostic. Bootstrap intervals are pointwise, conditional on these locations and fixed model.')
    for path in [ART/'R55_COMBINED_RESULTS.json',ROOT/'paper/data/arithmetic_layer_function.json']:
        path.write_text(json.dumps(result,indent=2)+'\n')
    with (ROOT/'paper/data/arithmetic_layer_function.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    tex=[r'\begin{table}[t]',r'\centering\small',r'\begin{tabular}{llrrrr}',r'\toprule',
         r'& & \multicolumn{2}{c}{Two operands} & \multicolumn{2}{c}{Three operands} \\',
         r'Layer & Operator & Change & Preserve & Change & Preserve \\',r'\midrule']
    for layer in LAYERS:
        if layer!=LAYERS[0]:tex.append(r'\midrule')
        for method,label in zip(METHODS,['Initial writer','Fitted writer','Known carry']):
            values=[]
            for arity in [2,3]:
                row=next(r for r in all_rows if r['layer']==layer and r['arity']==arity and r['method']==method)
                values += [row['change'],row['preserve']]
            tex.append(f'{layer} & {label} & '+' & '.join(f'{v:.2f}' for v in values)+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',
        r'\caption{Carry interventions at four Qwen block outputs, indexed from zero. All sites receive the same 405 source-fit questions and 256 learning batches. Initial and fitted writers share a fixed carry reader; known carry uses the fitted writer with the true carry difference. Complete answers are generated for the same 512 exposed requests. Scores are percentages; changing the carry and preserving same-carry answers remain separate.}',
        r'\label{tab:carry_sites}',r'\end{table}']
    (ROOT/'paper/tables/arithmetic_layer_function.tex').write_text('\n'.join(tex)+'\n')
    print(json.dumps(dict(table=all_rows,total_driver_seconds=result['total_driver_seconds'])))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    data=json.loads((ROOT/'paper/data/arithmetic_layer_function.json').read_text())
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'mathtext.fontset':'stix',
        'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(6.7,3.85),sharex=True,sharey=True)
    styles=[('#777777','o',-.20),('#246344','s',0),('#744467','D',.20)]
    for j,arity in enumerate([2,3]):
        for i,endpoint in enumerate(['change','preserve']):
            ax=axes[i,j]
            for method,(color,marker,offset) in zip(METHODS,styles):
                rows=[next(r for r in data['table'] if r['arity']==arity and r['layer']==l and r['method']==method) for l in LAYERS]
                means=np.array([r[endpoint] for r in rows]);bounds=np.array([r[endpoint+'_interval'] for r in rows])
                ax.errorbar(means,np.arange(4)+offset,xerr=np.maximum(0,np.stack([means-bounds[:,0],bounds[:,1]-means])),
                    fmt=marker,color=color,mfc='white' if method=='raw_initial' else color,ms=4,lw=.65,capsize=1.8)
            ax.set_yticks(np.arange(4),[str(l) for l in LAYERS]);ax.set_xlim(-3,103);ax.set_ylim(3.5,-.5)
            ax.set_xticks([0,25,50,75,100]);ax.grid(axis='x',color='#dddddd',lw=.45);ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
            if j==0:ax.set_ylabel('Block output')
            ax.set_title(('Two operands' if arity==2 else 'Three operands')+' · '+('change' if endpoint=='change' else 'preserve'),loc='left')
            if i==1:ax.set_xlabel('Complete-answer success (%)')
    handles=[Line2D([0],[0],marker=m,color=c,lw=0,ms=4,mfc='white' if k==0 else c)
             for k,(c,m,_o) in enumerate(styles)]
    fig.legend(handles,['Initial writer','Fitted writer','Known carry'],loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.52,1.025))
    fig.tight_layout(rect=(0,0,1,.965),h_pad=1.4,w_pad=1.3)
    for suffix in ['pdf','svg','png']:
        fig.savefig(ROOT/f'paper/figures/arithmetic_layer_function.{suffix}',bbox_inches='tight',dpi=220)
    plt.close(fig)


if __name__=='__main__':
    import sys
    plot() if '--plot' in sys.argv else main()
