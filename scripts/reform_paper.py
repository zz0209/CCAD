"""Source-backed export and figures for the scientific reform experiments."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,json,hashlib,statistics


def export(root,paper):
    manifest=json.loads((paper/'reform_runs.json').read_text());inputs=[]
    def read(rel):
        p=root/rel;inputs.append(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        return json.loads(p.read_text())
    norm=read(manifest['norm_run']+'/query_results.json')['queries']
    groups=read(manifest['group_run']+'/query_results.json')['queries']
    p=root/manifest['group_run']/'functional.raw.jsonl'
    inputs.append(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    rows=[json.loads(x) for x in p.read_text().splitlines()];queries=[]
    for q in groups:
        rr=[r for r in rows if r['query']==q['query'] and r['stratum']=='source_active']
        methods={}
        for name in sorted({r['method'] for r in rr}):
            v=[r for r in rr if r['method']==name]
            methods[name]=dict(source_kl=statistics.mean(r['source_kl'] for r in v),
                source_effect=statistics.mean(r['source_effect_kl'] for r in v),
                relative_kl=sum(r['source_kl'] for r in v)/max(sum(r['source_effect_kl'] for r in v),1e-15))
        queries.append(dict(query=q['query'],objective=q['objective'],anchor=q['anchor'],
            source_members=q['fit']['source_members'],target_members=q['fit']['target_members'],
            source_weight_sum=q['fit']['source_weight_sum'],target_weight_sum=q['fit']['target_weight_sum'],
            discovery_error=q['metrics']['joint_group']['discovery']['actual_relative_error'],
            calibration_error=q['metrics']['joint_group']['calibration']['actual_relative_error'],methods=methods))
    aggregate={}
    for obj in ['topk','matryoshka']:
        qq=[q for q in queries if q['objective']==obj];aggregate[obj]={}
        for name in qq[0]['methods']:
            aggregate[obj][name]=dict(mean_query_relative_kl=statistics.mean(q['methods'][name]['relative_kl'] for q in qq),
                median_query_relative_kl=statistics.median(q['methods'][name]['relative_kl'] for q in qq),
                pooled_relative_kl=sum(q['methods'][name]['source_kl'] for q in qq)/sum(q['methods'][name]['source_effect'] for q in qq))
    result=dict(norm_queries=norm,norm_means={name:{metric:statistics.mean(q['summary'][name][metric] for q in norm)
        for metric in ['iia','source_kl']} for name in norm[0]['summary']},groups=queries,group_summary=aggregate,
        inputs=inputs,scope=manifest['scope'])
    out=paper/'data/reform_r22.json';out.write_text(json.dumps(result,indent=2)+'\n')
    with (paper/'data/reform_r22_group_function.csv').open('w',newline='') as f:
        fields=['query','objective','anchor','source_members','target_members','discovery_error','calibration_error','method','source_kl','source_effect','relative_kl']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for q in queries:
            for name,met in q['methods'].items():w.writerow({**{k:q[k] for k in fields[:7]},'method':name,**met})
    lines=[r'\begin{tabular}{lrr}',r'\toprule',r'Method & IIA (\%) & Source KL\\',r'\midrule']
    for name,label in [('geometric_original','Geometric'),('norm_only_matched64','Two scales, 64 updates'),('norm_only_stronger256','Two scales, 256 updates'),('functional_learned_norm','Full weights, 64 updates')]:
        v=result['norm_means'][name];lines.append(f"{label} & {100*v['iia']:.2f} & {v['source_kl']:.5f}"+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}']);(paper/'tables/reform_norm_fit.tex').write_text('\n'.join(lines)+'\n')
    return result


def plot(paper):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    data=json.loads((paper/'data/reform_r22.json').read_text())
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'mathtext.fontset':'stix',
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    names=[('target_best_atom','Best atom','#999999','s'),('target_group','Native group','#22685F','o'),
           ('target_group_rank1','Rank-one projection','#75465B','x')]
    fig,axes=plt.subplots(1,2,figsize=(6.8,3.65),sharex=True)
    for ax,(obj,label) in zip(axes,[('topk','TopK'),('matryoshka','Matryoshka')]):
        qq=[q for q in data['groups'] if q['objective']==obj]
        qq=sorted(qq,key=lambda q:q['methods']['target_group']['relative_kl'])
        for i,q in enumerate(qq):
            values=[q['methods'][n]['relative_kl'] for n,_,_,_ in names]
            ax.plot([min(values),max(values)],[i,i],color='#cccccc',lw=.8,zorder=1)
            for (n,_,color,marker),v in zip(names,values):ax.scatter(v,i,color=color,marker=marker,s=19,linewidths=.85,zorder=3)
        ax.set_yticks(range(len(qq)),[str(q['anchor']) for q in qq]);ax.invert_yaxis()
        ax.set(xlim=(-.03,1.1),xticks=[0,.25,.5,.75,1],xlabel='Relative KL to source-group ablation',ylabel='Source anchor')
        ax.set_title(label,loc='left',fontsize=10);ax.grid(axis='x',color='#ededed',lw=.5);ax.tick_params(axis='y',length=0)
    handles=[Line2D([],[],color=c,marker=m,lw=0,markersize=4,label=label) for _,label,c,m in names]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,1.01),ncol=3,frameon=False,fontsize=8.5)
    fig.subplots_adjust(top=.86,bottom=.16,left=.085,right=.99,wspace=.34)
    out=paper/'figures/reform_native_groups'
    for suffix in ['pdf','svg','png']:fig.savefig(out.with_suffix('.'+suffix),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--paper',type=Path,default=Path(__file__).resolve().parents[1]/'paper')
    ap.add_argument('--figures-only',action='store_true');ap.add_argument('--data-only',action='store_true');args=ap.parse_args()
    if not args.figures_only:export(Path(__file__).resolve().parents[1],args.paper)
    if not args.data_only:plot(args.paper)
