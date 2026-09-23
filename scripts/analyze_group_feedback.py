import argparse
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

from analyze_shift_vector_reuse import (evaluation_statistics, identity, read_groups,
                                       read_json, summarize, write_json)
from analyze_shift_member_responses import group_statistics


def evaluate_original(args):
    assert not args.output.exists()
    with np.load(args.original_head) as raw:
        weight = raw['weight'].astype(float).ravel()
        bias = float(raw['bias'][0])
    targets = [read_groups(run,weight) for run in args.runs]
    reference = targets[0][0]
    for target in targets[1:]:
        np.testing.assert_array_equal(target[0]['document_sha256'],reference['document_sha256'])
        assert set(target[3])==set(targets[0][3])
    evaluation = np.flatnonzero(reference['splits']=='evaluation')
    labels,genders = reference['labels'],reference['genders']
    strata = [evaluation[(labels[evaluation]==y)&(genders[evaluation]==g)] for y in [0,1] for g in [0,1]]
    assert all(len(cell)>0 for cell in strata)
    methods = list(targets[0][3])

    def statistics(documents):
        result,details = {},{}
        for method in methods:
            normalized, summaries = [],[]
            for t,(_,baseline,source,groups) in enumerate(targets):
                v = groups[method]
                denominator = float(np.mean(source[documents]**2))
                numerator = float(np.mean(v['error'][documents]**2))
                assert denominator>0
                normalized.append(numerator/denominator)
                stats = group_statistics(v['actual']@weight,None,None,source@weight,
                                         baseline@weight+bias,labels,genders,documents)
                for condition,ids in [('joint',[0,1]),('clean',[0]),('conditional',[1])]:
                    error = v['error'][documents][:,ids]
                    truth = source[documents][:,ids]
                    stats[condition+'/pooled_rmse'] = float(np.sqrt(np.mean(error**2)))
                    stats[condition+'/pooled_source_rms'] = float(np.sqrt(np.mean(truth**2)))
                    stats[condition+'/pooled_nrmse_common_scale'] = float(np.sqrt(np.mean(error**2)/denominator))
                summaries.append(stats)
                details.update({f't{t}/{method}/{k}':value for k,value in stats.items()})
            result[method+'/primary_pooled_nrmse'] = float(np.sqrt(np.mean(normalized)))
            result.update({method+'/'+k:float(np.mean([s[k] for s in summaries])) for k in summaries[0]})
        return result,details

    points,details = statistics(evaluation)
    print({method:{key:points[method+'/'+key] for key in ['primary_pooled_nrmse',
        'joint/nrmse_common_source_scale','clean/profession_accuracy','conditional/profession_accuracy']}
        for method in methods},flush=True)
    draws,detail_draws = {k:[] for k in points},{k:[] for k in details}
    rng = np.random.default_rng(9232341)
    for i in range(args.bootstrap):
        selected = np.concatenate([rng.choice(cell,len(cell),replace=True) for cell in strata])
        current,current_details = statistics(selected)
        for k,v in current.items():
            draws[k].append(v)
        for k,v in current_details.items():
            detail_draws[k].append(v)
        if (i+1)%250==0:
            print(f'原任务共同文档 bootstrap {i+1}/{args.bootstrap}',flush=True)
    suffixes = [k[len(methods[0])+1:] for k in points if k.startswith(methods[0]+'/')]
    comparisons = {}
    for first,second in combinations(methods,2):
        comparisons[first+'_minus_'+second] = {name:dict(
            value=points[first+'/'+name]-points[second+'/'+name],
            ci95=summarize(np.array(draws[first+'/'+name])-np.array(draws[second+'/'+name]))) for name in suffixes}
    args.output.mkdir(parents=True)
    write_json(args.output/'RESULTS.json',dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        statistics={k:dict(value=v,ci95=summarize(draws[k])) for k,v in points.items()},
        per_target={k:dict(value=v,ci95=summarize(detail_draws[k])) for k,v in details.items()},
        paired_contrasts=comparisons,documents=len(evaluation),
        fixed_targets=[read_json(run/'config.resolved.json')['target_seed'] for run in args.runs],
        primary='Square root of mean fixed-target pooled squared error divided by each target source pooled energy, both backgrounds jointly',
        original_head='Fixed professor/nurse readout; original source operation preservation and classification reported separately',
        bootstrap=dict(draws=args.bootstrap,seed=9232341,unit='Document within label/gender, shared across methods, targets and backgrounds'),
        identities=dict(original_head=identity(args.original_head),analyzer=identity(Path(__file__)),
            groups=[identity(run/'response_index.json') for run in args.runs])))
    arrays = dict(document_sha256=reference['document_sha256'][evaluation],labels=labels[evaluation],genders=genders[evaluation])
    for t,(_,baseline,source,groups) in enumerate(targets):
        arrays[f't{t}__baseline'] = baseline[evaluation]
        arrays[f't{t}__source_effect'] = source[evaluation]
        for method,values in groups.items():
            arrays[f't{t}__{method}__actual'] = values['actual'][evaluation]
    np.savez_compressed(args.output/'row_arrays.npz',**arrays)
    keys = ['primary_pooled_nrmse','joint/nrmse_common_source_scale','clean/profession_accuracy','conditional/profession_accuracy']
    lines = ['# 完整组反馈的原任务使用结果','','| 方法 | pooled nRMSE | head nRMSE | clean准确率 | conditional准确率 |',
             '|---|---:|---:|---:|---:|']
    for method in methods:
        lines.append('| '+method+' | '+' | '.join(f'{points[method+"/"+key]:.6f}' for key in keys)+' |')
    (args.output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(dict(output=str(args.output),documents=len(evaluation),methods=len(methods)),flush=True)


def evaluate(args):
    assert not args.output.exists()
    panel = read_json(args.panel)
    with np.load(args.original_head) as raw:
        weight = raw['weight'].astype(float).ravel()
    targets = [read_groups(run, weight) for run in args.runs]
    reference = targets[0][0]
    for target in targets[1:]:
        np.testing.assert_array_equal(target[0]['document_sha256'], reference['document_sha256'])
        assert set(target[3])==set(targets[0][3])
    row_map = {r['document_sha256']:r for r in panel['rows']}
    rows = [row_map[str(x)] for x in reference['document_sha256']]
    assert len(rows)==len(row_map)
    professions = np.array([r['profession'] for r in rows])
    genders = np.array([r['gender'] for r in rows])
    heads = []
    for task in panel['tasks']:
        path = args.heads_directory/f'none__full__{task["name"]}__probe42.npz'
        with np.load(path) as raw:
            head = dict(task, weight=raw['weight'].astype(float).ravel(), bias=float(raw['bias'][0]), identity=identity(path))
        head['cache'] = []
        for _,baseline,source,methods in targets:
            cache = dict(source=source@head['weight'], baseline=baseline@head['weight']+head['bias'])
            for method,arrays in methods.items():
                cache[method] = {key:arrays[key]@head['weight'] for key in ['actual','error','parallel','perpendicular']}
                v = cache[method]
                np.testing.assert_allclose(v['error'], v['actual']-cache['source'], atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(v['error']**2, v['parallel']**2+v['perpendicular']**2+
                                          2*v['parallel']*v['perpendicular'], atol=1e-10, rtol=1e-10)
            head['cache'].append(cache)
        heads.append(head)
    assert len(heads)==4
    strata = [np.flatnonzero((professions==p)&(genders==g)) for p in sorted(set(professions)) for g in [0,1]]
    assert len(strata)==8 and all(len(cell) for cell in strata)
    rng = np.random.default_rng(9232341)
    points, head_points = evaluation_statistics(targets,heads,professions,genders,np.arange(len(rows)),True)
    draws = {key:[] for key in points}
    head_draws = {key:[] for key in head_points}
    for i in range(args.bootstrap):
        sampled = np.concatenate([rng.choice(cell,len(cell),replace=True) for cell in strata])
        current, current_heads = evaluation_statistics(targets,heads,professions,genders,sampled,True)
        for key,value in current.items():
            draws[key].append(value)
        for key,value in current_heads.items():
            head_draws[key].append(value)
        if (i+1)%250==0:
            print(f'共同文档 bootstrap {i+1}/{args.bootstrap}', flush=True)
    methods = list(targets[0][3])
    comparisons = {}
    metric_names = ['primary_nrmse','clean/actual_accuracy','conditional/actual_accuracy',
                    'joint/actual_accuracy','clean/actual_worst_group','conditional/actual_worst_group']
    for first,second in combinations(methods,2):
        comparison = {}
        for name in metric_names:
            a,b = first+'/'+name,second+'/'+name
            differences = [x-y if x is not None and y is not None else None for x,y in zip(draws[a],draws[b])]
            comparison[name] = dict(value=points[a]-points[b] if points[a] is not None and points[b] is not None else None,
                                    ci95=summarize(differences))
        comparisons[first+'_minus_'+second] = comparison
    args.output.mkdir(parents=True)
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        statistics={k:dict(value=v, ci95=summarize(draws[k])) for k,v in points.items()},
        per_head={k:dict(value=v, ci95=summarize(head_draws[k])) for k,v in head_points.items()},
        paired_contrasts=comparisons, documents=len(rows),
        fixed_targets=[read_json(run/'config.resolved.json')['target_seed'] for run in args.runs],
        heads=[{k:v for k,v in head.items() if k not in ['weight','bias','cache']} for head in heads],
        primary='Square root after averaging head/source-scale-normalized squared error over fixed targets and heads, two backgrounds jointly',
        weighting='Four profession/gender cells equal within each head; heads and targets equal',
        bootstrap=dict(draws=args.bootstrap, seed=9232341, unit='Document within profession/gender; common draw across every method, head, target and background'),
        evidence='Previously exposed later-head development; all predeclared prefix supports retained',
        identities=dict(panel=identity(args.panel), original_head=identity(args.original_head), analyzer=identity(Path(__file__)),
            statistics_module=identity(Path(__file__).with_name('analyze_shift_vector_reuse.py')),
            groups=[identity(run/'response_index.json') for run in args.runs]))
    write_json(args.output/'RESULTS.json',result)
    arrays = dict(document_sha256=reference['document_sha256'], professions=professions, genders=genders)
    for t,(_,baseline,source,methods_data) in enumerate(targets):
        arrays[f't{t}__baseline'] = baseline
        arrays[f't{t}__source_effect'] = source
        for method,values in methods_data.items():
            for key,value in values.items():
                arrays[f't{t}__{method}__{key}'] = value
    np.savez_compressed(args.output/'row_arrays.npz',**arrays)
    lines = ['# 完整组反馈的后来读出', '', '| 方法 | nRMSE | clean准确率 | conditional准确率 |', '|---|---:|---:|---:|']
    for method in methods:
        lines.append('| '+method+' | '+' | '.join(f'{points[method+"/"+key]:.6f}' for key in metric_names[:3])+' |')
    (args.output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(dict(output=str(args.output),documents=len(rows),methods=len(methods)),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--endpoint',choices=['later','original'],default='later')
    for name in ['panel','heads-directory']:
        parser.add_argument('--'+name,type=Path)
    for name in ['original-head','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--bootstrap',type=int,default=1000)
    parser.add_argument('--check-existing',action='store_true')
    args = parser.parse_args()
    assert args.bootstrap>0
    if args.check_existing:
        check_existing(args)
    elif args.endpoint=='original':
        evaluate_original(args)
    else:
        assert args.panel is not None and args.heads_directory is not None
        evaluate(args)


def check_existing(args):
    destination = args.output/'DIRECT_SSE_CHECK.json'
    assert not destination.exists()
    saved = read_json(args.output/'RESULTS.json')
    arrays = dict(np.load(args.output/'row_arrays.npz',allow_pickle=False))
    methods = sorted({key.split('__')[1] for key in arrays if key.endswith('__actual')})
    target_ids = sorted({key.split('__')[0] for key in arrays if key.endswith('__source_effect')})
    errors = {}
    if args.endpoint=='original':
        for method in methods:
            ratios = []
            for t in target_ids:
                source = arrays[t+'__source_effect']
                actual = arrays[t+'__'+method+'__actual']
                ratios.append(float(np.sum((actual-source)**2)/np.sum(source**2)))
            recomputed = float(np.sqrt(np.mean(ratios)))
            errors[method+'/primary_pooled_nrmse'] = abs(recomputed-saved['statistics'][method+'/primary_pooled_nrmse']['value'])
    else:
        professions = arrays['professions']
        genders = arrays['genders']
        for method in methods:
            ratios = []
            for head in saved['heads']:
                with np.load(head['identity']['path']) as raw:
                    weight = raw['weight'].astype(float).ravel()
                cells = [np.flatnonzero((professions==p)&(genders==g))
                         for p in [head['negative'],head['positive']] for g in [0,1]]
                for t in target_ids:
                    source = arrays[t+'__source_effect']@weight
                    actual = arrays[t+'__'+method+'__actual']@weight
                    mse = np.mean([np.sum((actual[cell]-source[cell])**2)/(2*len(cell)) for cell in cells])
                    scale = np.mean([np.sum(source[cell]**2)/(2*len(cell)) for cell in cells])
                    ratios.append(float(mse/scale))
            errors[method+'/primary_nrmse'] = abs(float(np.sqrt(np.mean(ratios)))-saved['statistics'][method+'/primary_nrmse']['value'])
    assert max(errors.values()) < 1e-10
    write_json(destination,dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Direct finite group minus source SSE from saved per-document arrays; no selection or model execution',
        inputs=dict(results=identity(args.output/'RESULTS.json'),arrays=identity(args.output/'row_arrays.npz')),
        analyzer=identity(Path(__file__)), errors=errors,maximum_error=max(errors.values())))
    print(dict(check=str(destination),maximum_error=max(errors.values())),flush=True)


if __name__ == '__main__':
    main()
