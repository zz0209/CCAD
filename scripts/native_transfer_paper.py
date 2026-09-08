"""Source-backed tables and figures for target-native external-task operations."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

TASKS = ['agr_gender', 'filler_gap_obj', 'npi_any_obj-relc', 'garden_npz_v-trans']
LABELS = ['Gender', 'Filler--gap', 'NPI licensing', 'NP/Z garden path']


def export(root, out, read):
    relative = Path('runs/FINAL5_R14_causalgym_finite_writer_v4_20260908')
    if not (root/relative/'metrics.summary.json').exists():
        return None
    summaries = {}
    sources = []
    for name, run_id in [('geometry', 'FINAL5_R14_causalgym_native_cpu_v2_20260908'),
                         ('source_ig', 'FINAL5_R14_causalgym_source_ig16_v3_20260908'),
                         ('finite', relative.name)]:
        base = Path('runs')/run_id
        summary = read(base/'metrics.summary.json')
        assert summary['status']=='PASS' and read(base/'contract_validation.json')['ok']
        summaries[name] = summary
        sources += [(base/'metrics.summary.json').as_posix(), (base/'metrics.raw.jsonl').as_posix()]
    raw_path = root/relative/'metrics.raw.jsonl'
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    assert len(raw)==summaries['finite']['rows']
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest()==summaries['finite']['metrics_raw_sha256']
    cfg = read(relative/'config.resolved.json')
    fits = read(relative/'fit_diagnostics.json')['rows']
    panel = read(relative/'panel.json')['rows']
    sources += [(relative/name).as_posix() for name in ['config.resolved.json','fit_diagnostics.json','panel.json','projection_diagnostics.json']]
    # Resample entire prompt-connected components. Reciprocal directions are first
    # averaged; no dictionary-seed or template-population uncertainty is implied.
    import numpy as np
    rng = np.random.default_rng(140908)
    lookup = {(r['task'],r['site'],r['method'],r['row_id']):r for r in raw if r['kind']=='intervention'}
    deltas = []
    for task in TASKS:
        for operation in ['whole','half_dose','members_0','members_1']:
            for method in ['fcc','full','raw','source_oracle']:
                left = method+'_fixed_native' if operation=='whole' else method+'_fixed_'+operation
                right = method+'_finite_native' if operation=='whole' else method+'_finite_'+operation
                groups = {}
                for row in raw:
                    if row['kind']!='intervention' or row['task']!=task or row['site']!='changed_region_end' or row['method']!=left:
                        continue
                    other = lookup[task,row['site'],right,row['row_id']]
                    assert row['reference_identity']==other['reference_identity']
                    groups.setdefault(row['component'],[]).append(other['kl_to_source']-row['kl_to_source'])
                values = np.array([statistics.fmean(v) for _,v in sorted(groups.items())])
                assert len(values)==7
                bootstrap = values[rng.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)
                deltas.append(dict(task=task,operation=operation,method=method,n_components=len(values),
                                   finite_minus_fixed=float(values.mean()),component_means=values.tolist(),
                                   descriptive_95_interval=np.quantile(bootstrap,[.025,.975]).tolist()))
    examples = []
    for task in TASKS:
        row = next(r for r in panel if r['task']==task and r['split']=='held_component_development')
        observations = [r for r in raw if r['kind']=='intervention' and r['task']==task and r['site']=='changed_region_end' and r['row_id']==row['row_id']]
        examples.append(dict(task=task,panel=row,donor=next(r for r in panel if r['row_id']==row['donor_id']),observations=observations,
                             choice='First hash-ordered retained evaluation component and side, without outcome selection'))
    data = dict(summaries=summaries,config=cfg,fits=fits,finite_differences=deltas,examples=examples,
                input_summary_paths=sources,
                interval_scope='Descriptive percentile resampling of seven observed prompt components; reciprocal directions averaged. Conditional on one dictionary pair and these task constructions, not seed or template population inference.')
    (out/'data/external_native_transfer.json').write_text(json.dumps(data,indent=2)+'\n')
    cells = [r for r in summaries['finite']['cells'] if r['kind']=='intervention']
    table_lookup = {(r['task'],r['site'],r['method']):r for r in cells}
    def value(task,site,method,key='kl_to_source_mean'):
        return table_lookup[task,site,method][key]
    lines = []
    for method,label in [('noop','No operation'),('same_read_members_native','Same reading members, native'),
                         ('fcc_readout','Compact readout'),('dense_readout','Dense-selected readout'),
                         ('full_readout','Full-code readout'),('raw_readout','Raw readout'),
                         ('one_to_one_readout','Conditional assignment'),('source_class_mean_readout','Class mean (label sign)'),
                         ('fcc_fixed_native','Compact, bounded writer'),('full_fixed_native','Full codes, bounded writer'),
                         ('raw_fixed_native','Raw, bounded writer'),('fcc_adaptive_native','Compact, adaptive writer'),
                         ('full_adaptive_native','Full codes, adaptive writer'),('raw_adaptive_native','Raw, adaptive writer'),
                         ('fcc_finite_native','Compact, finite writer'),('full_finite_native','Full codes, finite writer'),
                         ('raw_finite_native','Raw, finite writer'),('source_oracle_fixed_native','True source, bounded writer'),
                         ('source_oracle_finite_native','True source, finite writer')]:
        lines.append(label+' & '+' & '.join(f"{value(task,'changed_region_end',method):.6f}" for task in TASKS)+r' \\')
    (out/'tables/external_native_whole.tex').write_text('\n'.join(lines)+'\n')
    lines = []
    for task,label in zip(TASKS,LABELS):
        for method,ml in [('raw_donor','Raw donor'),('source_native','Selected source 16'),
                          ('source_geometric_same_budget','Geometric source 16'),('source_native_atom','First source atom'),
                          ('source_full_sae_donor','Full source SAE')]:
            vals = []
            for site in cfg['sites']:
                vals += [value(task,site,method,'donor_oriented_change_mean'),100*value(task,site,method,'donor_label_correct_mean')]
            lines.append(label+' & '+ml+' & '+' & '.join(f'{v:.3f}' for v in vals)+r' \\')
    (out/'tables/external_source_effect.tex').write_text('\n'.join(lines)+'\n')
    lines = []
    for task,label in zip(TASKS,LABELS):
        for method,ml in [('fcc','Compact'),('full','Full codes'),('raw','Raw'),('source_oracle','True source')]:
            vals = []
            for operation in ['half_dose','members_0','members_1']:
                vals += [value(task,'changed_region_end',method+'_'+variant+'_'+operation) for variant in ['readout','fixed','finite']]
            lines.append(label+' & '+ml+' & '+' & '.join(f'{v:.6f}' for v in vals)+r' \\')
    (out/'tables/external_native_components.tex').write_text('\n'.join(lines)+'\n')
    def tex(value):
        return str(value).replace('\\',r'\textbackslash{}').replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#')
    lines=[]
    for example,label in zip(examples,LABELS):
        p=example['panel'];donor=example['donor'];obs={r['method']:r for r in example['observations']}
        words=[tex(x) for x in p['spans'][1:]]
        words[p['changed_region']-1]=r'\emph{'+words[p['changed_region']-1]+'}'
        context=tex(label)+': '+''.join(words)+r'\newline '+tex(p['spans'][p['changed_region']].strip())+r' $\to$ '+tex(donor['spans'][donor['changed_region']].strip())+r'; continuation '+tex(p['label'].strip())+r' $\to$ '+tex(p['donor_label'].strip())
        values=[obs['source_native']['donor_oriented_change']]+[obs[m]['kl_to_source'] for m in ['fcc_readout','fcc_fixed_native','fcc_finite_native']]
        lines.append(context+' & '+' & '.join(f'{v:.5f}' for v in values)+r' \\[5pt]')
    (out/'tables/external_native_examples.tex').write_text('\n'.join(lines)+'\n')
    with (out/'data/external_native_cells.csv').open('w',newline='') as f:
        keys = sorted(set().union(*(r.keys() for r in cells)))
        writer = csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(cells)
    semantic_base = Path('runs/FINAL5_R14_causalgym_semantic_controls_v5_20260908')
    if (root/semantic_base/'metrics.summary.json').exists():
        semantic = read(semantic_base/'metrics.summary.json')
        assert semantic['status']=='PASS' and read(semantic_base/'contract_validation.json')['ok']
        semantic_raw = [json.loads(line) for line in (root/semantic_base/'metrics.raw.jsonl').read_text().splitlines()]
        assert len(semantic_raw)==semantic['rows']
        assert hashlib.sha256((root/semantic_base/'metrics.raw.jsonl').read_bytes()).hexdigest()==semantic['metrics_raw_sha256']
        old_lookup = {(r['kind'],r.get('task'),r.get('site'),r.get('method'),r.get('row_id')):r for r in raw}
        common = 0
        max_error = 0.
        for row in semantic_raw:
            key = (row['kind'],row.get('task'),row.get('site'),row.get('method'),row.get('row_id'))
            if key in old_lookup and row['kind']=='intervention':
                previous = old_lookup[key]
                assert row.get('reference_identity')==previous.get('reference_identity')
                if row['kl_to_source'] is not None:
                    max_error = max(max_error,abs(row['kl_to_source']-previous['kl_to_source']))
                else:
                    assert previous['kl_to_source'] is None
                    assert row['donor_oriented_change']==previous['donor_oriented_change']
                common += 1
        assert common > 1000 and max_error < 1e-10
        semantic_fits = read(semantic_base/'fit_diagnostics.json')
        data['semantic'] = dict(summary=semantic, fits=semantic_fits,
                               common_rows_replayed=common, common_kl_max_error=max_error)
        sources += [(semantic_base/name).as_posix() for name in ['metrics.summary.json','metrics.raw.jsonl','fit_diagnostics.json','config.resolved.json']]
        sources += ['runs/FINAL5_R14_semantic_contexts_v2_20260908/'+name for name in ['matches.json','config.resolved.json','metrics.summary.json']]
        s_lookup = {(r['task'],r['site'],r['method']):r for r in semantic['cells'] if r['kind']=='intervention'}
        lines = []
        methods = [('fcc','Compact'),('dense','Dense-selected'),('full','Full codes'),('raw','Raw'),
                   ('semantic_context','Context distribution OT'),('centroid_context','Context centroid'),
                   ('decoder_nearest','Decoder nearest'),('global_hungarian','Global assignment')]
        for suffix, operation in [('readout','Readout'),('copied_native','Native copy'),('fixed_native','Bounded writer')]:
            for method,label in methods:
                if (TASKS[0],'changed_region_end',method+'_'+suffix) not in s_lookup:
                    continue
                lines.append(operation+' & '+label+' & '+' & '.join(f"{s_lookup[t,'changed_region_end',method+'_'+suffix]['kl_to_source_mean']:.6f}" for t in TASKS)+r' \\')
        (out/'tables/external_semantic_controls.tex').write_text('\n'.join(lines)+'\n')
        (out/'data/external_semantic_controls.json').write_text(json.dumps(data['semantic'],indent=2)+'\n')
        (out/'data/external_native_transfer.json').write_text(json.dumps(data,indent=2)+'\n')
    return data


def plot(data,save):
    if not data:
        return
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    cells = {(r['task'],r['site'],r['method']):r for r in data['summaries']['finite']['cells'] if r['kind']=='intervention'}
    # Real operator diagram: native decoder columns, state clipping and model path.
    fig = plt.figure(figsize=(7,4.55))
    ax = fig.add_axes([.035,.69,.93,.27]);ax.axis('off');ax.set(xlim=(0,7),ylim=(0,2))
    ax.text(0,1.94,'(a) Read the source operation; write through the target decoder',va='top',fontsize=9)
    items = [(0.45,r'$\Delta z_{t,T}$','Reading members'),(2.0,r'$\widehat c=W_T^{\mathsf{T}}\Delta z_{t,T}$','Source coordinates'),
             (4.0,r'$u_J=\max(A\widehat c,-z_{t,J})$','Target state constraint'),(6.25,r'$q_t=D_{t,J}u_J$','Native physical edit')]
    for x,equation,label in items:
        ax.text(x,1.0,equation,ha='center',va='center',fontsize=10)
        ax.text(x,.43,label,ha='center',fontsize=8)
    for lo,hi in [(.90,1.13),(2.98,3.18),(5.12,5.51)]:
        ax.annotate('',xy=(hi,1.0),xytext=(lo,1.0),arrowprops=dict(arrowstyle='->',color='#666666',lw=.7))
    ax.text(.05,.02,'The update is injected at layer 3, before the final token; later Transformer layers run in full.',fontsize=8)
    # Effect magnitudes distinguish source capture from transfer fidelity.
    a = fig.add_axes([.255,.105,.715,.46])
    names = ['raw_donor','source_full_sae_donor','source_native','source_native_atom']
    values = np.array([[cells[t,'changed_region_end',m]['donor_oriented_change_mean'] for t in TASKS] for m in names])
    cmap = LinearSegmentedColormap.from_list('source_strength',['#f5f5f1','#286956'])
    a.imshow(values,cmap=cmap,vmin=0,vmax=max(4.,values.max()),aspect='auto')
    for (i,j),v in np.ndenumerate(values):
        a.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=9,color='white' if v>2.1 else '#262626')
    a.set(yticks=range(4),yticklabels=['Raw donor','Full source SAE','Selected source 16','First source atom'],
          xticks=range(4),xticklabels=['Gender','Filler–gap','NPI','NP/Z'])
    a.tick_params(length=0);a.set_title('(b) Mean donor-oriented log-odds change (nat)',loc='left',pad=12,fontsize=9)
    fig.text(.255,.025,'Seven evaluation components per task; reciprocal directions averaged; source 1 to target 2.',fontsize=7.5)
    save(fig,'external_native_operation')
    # One matrix per writing rule, identical data scale; avoid pooling weak tasks.
    fig = plt.figure(figsize=(7,5.25))
    cmap = LinearSegmentedColormap.from_list('kl_error',['#faf9f5','#b5a0bf','#65456f'])
    for index,variant in enumerate(['readout','fixed_native','finite_native']):
        a = fig.add_axes([.19,.75-index*.21,.765,.15])
        values = np.array([[cells[t,'changed_region_end',m+'_'+variant]['kl_to_source_mean'] for t in TASKS] for m in ['fcc','full','raw']])
        # Log-color floor is for display only; numbers and exported data keep values.
        a.imshow(np.log10(np.maximum(values,1e-7)),cmap=cmap,vmin=-7,vmax=-.5,aspect='auto')
        for (i,j),v in np.ndenumerate(values):
            exponent = int(np.floor(np.log10(max(v,1e-30))))
            text = r'$<10^{-6}$' if v<1e-6 else f'{v:.4f}' if v>=.001 else rf'${v/10**exponent:.1f}\times10^{{{exponent}}}$'
            a.text(j,i,text,ha='center',va='center',color='white' if v>.003 else '#262626',fontsize=8)
        a.set(yticks=range(3),yticklabels=['Compact','Full codes','Raw'],xticks=range(4),
              xticklabels=['Gender','Filler–gap','NPI','NP/Z'] if index==2 else [])
        a.tick_params(length=0)
        a.set_title(['(a) Source-aligned readout','(b) Fixed members, bounded least-squares writer',
                     '(c) Fixed members, finite-response writer'][index],loc='left',fontsize=9,pad=6)
    a = fig.add_axes([.19,.055,.765,.135])
    diff = {(r['task'],r['operation']):r for r in data['finite_differences'] if r['method']=='fcc'}
    values = np.array([[diff[t,o]['finite_minus_fixed'] for t in TASKS] for o in ['whole','half_dose','members_0','members_1']])
    colors = LinearSegmentedColormap.from_list('delta',['#286956','#faf9f5','#785481'])
    bound = max(abs(values.min()),abs(values.max()),1e-6)
    a.imshow(values,cmap=colors,vmin=-bound,vmax=bound,aspect='auto')
    for (i,j),v in np.ndenumerate(values):
        exponent=int(np.floor(np.log10(max(abs(v),1e-30))))
        label=f'{v:+.4f}' if abs(v)>=.0001 else rf'${v/10**exponent:+.1f}\times10^{{{exponent}}}$'
        a.text(j,i,label,ha='center',va='center',fontsize=7.5,
               color='white' if abs(v)>.65*bound else '#262626')
    a.set(yticks=range(4),yticklabels=['Whole','Half dose','First half','Second half'],xticks=range(4),xticklabels=[])
    a.tick_params(length=0);a.set_title('(d) Compact writer: finite minus bounded KL (negative favors finite)',loc='left',fontsize=8.5,pad=6)
    fig.text(.19,.012,'KL to the same source operation (nat). Darker purple in (a--c): larger error; common log scale.',fontsize=7.2)
    save(fig,'external_native_fidelity')
