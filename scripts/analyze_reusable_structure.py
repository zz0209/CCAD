import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from fit_interaction_request_coordinates import decode_parameters, linear_design, source_polynomial
from analyze_interaction_structure import BITS, OPERATIONS, decompose


BOOLEAN = np.array([[(bits>>axis)&1 for axis in range(3)] for bits in BITS],dtype=float)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity(path):
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def save(path,value):
    assert not path.exists(),path
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def source_terms(requests,pooled):
    zero = np.flatnonzero(np.all(requests==0,axis=1))
    assert len(zero)==1
    indices = [int(np.flatnonzero(np.all(requests==q,axis=1))[0]) for q in BOOLEAN]
    baseline = pooled[:,zero[0]]
    terms,_,error = decompose(pooled[:,indices],baseline)
    return np.stack([terms[name] for name in OPERATIONS],axis=1),baseline,error


def fit_arrays(terms,target_singletons):
    starts = [(1-3e-4)*np.eye(3)+1e-4*np.ones((3,3)),.5*np.eye(3)+np.ones((3,3))/6,np.ones((3,3))/3]
    def residual(parameters):
        matrix,_ = decode_parameters(parameters)
        return (source_polynomial(terms,BOOLEAN[:3],matrix)-target_singletons).ravel()
    def jacobian(parameters):
        matrix,chain = decode_parameters(parameters)
        _,jac = source_polynomial(terms,BOOLEAN[:3],matrix,True)
        return jac.reshape(-1,9)@chain
    fits = []
    for i,matrix in enumerate(starts):
        theta = np.log(matrix/(3-matrix.sum(1,keepdims=True))).ravel()
        result = least_squares(residual,theta,jac=jacobian,bounds=(-12,12),max_nfev=500,
                               ftol=1e-10,xtol=1e-10,gtol=1e-10)
        b,_ = decode_parameters(result.x)
        fits.append(dict(initialization=i,initial_B=matrix.tolist(),B=b.tolist(),theta=result.x.tolist(),
            fit_sse=float(result.fun@result.fun),distance_to_identity=float(np.sum((b-np.eye(3))**2)),
            success=bool(result.success),status=int(result.status),message=result.message,nfev=int(result.nfev),
            njev=int(result.njev),optimality=float(result.optimality)))
        print(f'B 初始化 {i+1}/3，nfev {result.nfev}，SSE {fits[-1]["fit_sse"]:.8g}',flush=True)
    minimum = min(row['fit_sse'] for row in fits)
    ties = [row for row in fits if np.isclose(row['fit_sse'],minimum,rtol=1e-10,atol=1e-12)]
    selected = min(ties,key=lambda row:(row['distance_to_identity'],row['initialization']))
    design = linear_design(terms,BOOLEAN[:3]).reshape(-1,9)
    response = target_singletons.ravel()
    m,_,rank,singular = np.linalg.lstsq(design,response,rcond=None)
    diagonal = np.array([np.sum(terms[:,i]*target_singletons[:,i])/np.sum(terms[:,i]**2) for i in range(3)])
    return dict(M=m.reshape(3,3).tolist(),M_fit_sse=float(np.sum((design@m-response)**2)),M_rank=int(rank),
        M_singular_values=singular.tolist(),M_normal_residual=float(np.max(np.abs(design.T@(design@m-response)))),
        B=selected['B'],selected_initialization=selected['initialization'],initialization_fits=fits,
        diagonal=diagonal.tolist(),tie_rule='SSE isclose rtol1e-10 atol1e-12, then distance to identity, then initialization index')


def predict(terms,requests,fit):
    additive_design = linear_design(terms,requests)
    source = source_polynomial(terms,requests,np.eye(3))
    higher = source-additive_design@np.eye(3).ravel()
    additive = additive_design@np.array(fit['M']).ravel()
    return dict(M_source_higher=additive+higher,B=source_polynomial(terms,requests,np.array(fit['B'])),
        M_additive=additive,source_unit=source,
        diagonal_source_higher=additive_design@np.diag(fit['diagonal']).ravel()+higher)


def load_run(run):
    assert read(run/'status.json')['status']=='PASS'
    index = read(run/'request_index.json')
    rows = read(run/'panel.json')['rows']
    documents = [r['document_sha256'] for r in rows]
    families = {}
    for family in ['source','target']:
        records = sorted(index[family+'_blocks'],key=lambda item:tuple(item['request']))
        requests,values,logits = [],[],[]
        for item in records:
            assert item['status']=='PASS'
            with np.load(item['path'],allow_pickle=False) as raw:
                assert str(raw['status'])=='PASS'
                assert raw['document_sha256'].tolist()==documents
                np.testing.assert_array_equal(raw['request'],item['request'])
                requests.append(item['request'])
                values.append(raw['pooled512'].astype(float))
                logits.append(raw['logits'].astype(float))
        assert len(set(map(tuple,requests)))==len(requests)
        families[family] = dict(requests=np.array(requests),pooled=np.stack(values,axis=1),logits=np.stack(logits,axis=1))
    return index,rows,families


def fit_runs(runs,output):
    assert not output.exists()
    started,cpu_started = time.perf_counter(),time.process_time()
    result = dict(targets={},inputs=[],specifications=dict(anchors=['0','P','N','W'],
        B_bounds='Nonnegative with row sum at most 3; theta[-12,12]',max_nfev=500,
        ftol=1e-10,xtol=1e-10,gtol=1e-10,target_interventions=24,target_baseline=8,
        source_boolean_calls=64,loss='Unnormalized 8 documents x 3 anchors x 512 coordinates SSE'))
    saved_arrays = {}
    for run in runs:
        index,rows,families = load_run(run)
        assert len(rows)==8
        terms,baseline,error = source_terms(families['source']['requests'],families['source']['pooled'])
        target = families['target']
        order = [int(np.flatnonzero(np.all(target['requests']==q,axis=1))[0]) for q in BOOLEAN[:3]]
        effects = target['pooled'][:,order]-baseline[:,None]
        seed = str(index['identity']['target_seed'])
        assert seed not in result['targets']
        result['targets'][seed] = dict(fit_arrays(terms,effects),source_boolean_reconstruction_error=error,
            calibration_documents=[r['document_sha256'] for r in rows],run=str(run.resolve()))
        grid = np.array([[p,n,w] for p in [0,.5,1] for n in [0,.5,1] for w in [0,.5,1]])
        global_choices = {}
        for method,prediction in predict(terms,grid,result['targets'][seed]).items():
            losses = np.sum((prediction[:,None]-terms[:,:3,None])**2,axis=(0,3))
            indices = np.argmin(losses,axis=1)
            global_choices[method] = dict(indices=indices.tolist(),requests=grid[indices].tolist(),
                fit_predicted_sse=losses[np.arange(3),indices].tolist(),all_fit_predicted_sse=losses.tolist())
        result['targets'][seed]['global_choices'] = global_choices
        result['targets'][seed]['global_choice_rule'] = 'Minimize fit8 predicted pooled SSE to each true source singleton over lexicographic grid; no target grid response used'
        saved_arrays['t'+seed+'__source_terms'] = terms
        saved_arrays['t'+seed+'__target_singletons'] = effects
        result['inputs'].append(dict(index=identity(run/'request_index.json'),panel=identity(run/'panel.json')))
    result.update(written_at_utc=datetime.now(timezone.utc).isoformat(),analyzer=identity(Path(__file__)),
        numerical_module=identity(Path(__file__).with_name('fit_interaction_request_coordinates.py')),
        cost=dict(wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu_started,threads=2,gpu_sequences=0))
    output.mkdir(parents=True)
    save(output/'MODELS.json',result)
    np.savez_compressed(output/'calibration_arrays.npz',**saved_arrays)
    print(dict(output=str(output),cost=result['cost']),flush=True)


def evaluate_runs(args):
    assert not args.output.exists()
    started,cpu_started = time.perf_counter(),time.process_time()
    models = read(args.models)
    with np.load(args.head) as raw:
        weight = raw['weight'].astype(float).ravel()
        bias = float(raw['bias'][0])
    observations = {}
    saved_arrays,choices_by_target,inputs = {},{},[]
    methods = ['M_source_higher','B','M_additive','source_unit','diagonal_source_higher']
    documents = None

    def add(key,numerator,denominator=None,root=False):
        observations.setdefault(key,dict(numerator=[],denominator=[],root=root))
        observations[key]['numerator'].append(np.asarray(numerator))
        observations[key]['denominator'].append(np.ones_like(numerator) if denominator is None else np.asarray(denominator))

    for run in args.runs:
        index,rows,families = load_run(run)
        seed = str(index['identity']['target_seed'])
        fit = models['targets'][seed]
        hashes = [r['document_sha256'] for r in rows]
        if documents is None:
            documents = hashes
        assert hashes==documents and not set(hashes)&set(fit['calibration_documents'])
        source,target = families['source'],families['target']
        queries = target['requests']
        expected = np.array([[p,n,w] for p in [0,.5,1] for n in [0,.5,1] for w in [0,.5,1]])
        np.testing.assert_array_equal(queries,expected)
        np.testing.assert_array_equal(source['requests'],queries)
        terms,baseline,reconstruction = source_terms(queries,source['pooled'])
        assert reconstruction<1e-10
        zero = int(np.flatnonzero(np.all(queries==0,axis=1))[0])
        np.testing.assert_allclose(target['pooled'][:,zero],baseline,atol=2e-5,rtol=0)
        true = target['pooled']-baseline[:,None]
        source_true = source['pooled']-baseline[:,None]
        predictions = predict(terms,queries,fit)
        singletons = [int(np.flatnonzero(np.all(queries==q,axis=1))[0]) for q in BOOLEAN[:3]]
        nonzero = np.flatnonzero(np.any(queries!=0,axis=1))
        fractional = np.flatnonzero(np.any(queries==.5,axis=1))
        combinations = np.array([i for i,q in enumerate(queries) if np.all(np.isin(q,[0,1])) and q.sum()>=2])
        groups = dict(unseen23=np.array([i for i in nonzero if i not in singletons]),
                      fractional19=fractional,boolean_combinations4=combinations,all26=nonzero,singletons3=np.array(singletons))
        groups.update({'q_'+''.join(str(int(2*x)) for x in q):np.array([i]) for i,q in enumerate(queries) if i!=zero})
        assert len(groups['unseen23'])==23 and len(fractional)==19 and len(combinations)==4
        labels = np.array([r['label'] for r in rows])
        genders = np.array([r['gender'] for r in rows])
        desired = source_true[:,singletons]
        actual_distance = np.mean((true[:,None]-desired[:,:,None])**2,axis=-1)
        oracle_choice = np.argmin(actual_distance,axis=-1)
        oracle_error = np.take_along_axis(actual_distance,oracle_choice[...,None],axis=-1)[...,0]
        selected = dict(original_request=np.broadcast_to(singletons,(len(rows),3)),oracle=oracle_choice)
        for method,prediction in predictions.items():
            predicted_distance = np.mean((prediction[:,None]-desired[:,:,None])**2,axis=-1)
            selected[method] = np.argmin(predicted_distance,axis=-1)
            fixed = fit['global_choices'][method]
            np.testing.assert_array_equal(queries[fixed['indices']],fixed['requests'])
            selected[method+'_fixed'] = np.broadcast_to(fixed['indices'],(len(rows),3))
            for group,indices in groups.items():
                error = prediction[:,indices]-true[:,indices]
                scale = np.mean(true[:,indices]**2,axis=(1,2))
                prefix = method+'/prediction/'+group
                add(prefix+'/nrmse',np.mean(error**2,axis=(1,2)),scale,True)
                add(prefix+'/rmse',np.mean(error**2,axis=(1,2)),root=True)
                add(prefix+'/head_rmse',np.mean((error@weight)**2,axis=1),root=True)
                actual_logits = target['logits'][:,indices]
                predicted_logits = baseline@weight+bias
                predicted_logits = predicted_logits[:,None]+prediction[:,indices]@weight
                add(prefix+'/predicted_sign_agreement',np.mean((predicted_logits>0)==(actual_logits>0),axis=1))
        main_adjustment = linear_design(terms,queries)@(np.array(fit['M'])-np.eye(3)).ravel()
        interpolation = predictions['source_unit']-source_true
        correspondence = source_true+main_adjustment-true
        np.testing.assert_allclose(interpolation+correspondence,predictions['M_source_higher']-true,atol=1e-11,rtol=1e-11)
        for group,indices in groups.items():
            a,b = interpolation[:,indices],correspondence[:,indices]
            scale = np.mean(true[:,indices]**2,axis=(1,2))
            for name,value in [('interpolation',a*a),('correspondence',b*b),('cross',2*a*b),('total',(a+b)**2)]:
                add('mechanism/'+group+'/'+name+'_mean_square',np.mean(value,axis=(1,2)))
                add('mechanism/'+group+'/'+name+'_normalized_square',np.mean(value,axis=(1,2)),scale)
        target_choices = {}
        for method,indices in selected.items():
            measured_error = np.take_along_axis(actual_distance,indices[...,None],axis=-1)[...,0]
            logits = target['logits'][np.arange(len(rows))[:,None],indices]
            desired_scale = np.mean(desired**2,axis=(1,2))
            add(method+'/selection/pooled_nrmse',measured_error.mean(axis=1),desired_scale,True)
            add(method+'/selection/rmse',measured_error.mean(axis=1),root=True)
            add(method+'/selection/regret_mse',(measured_error-oracle_error).mean(axis=1))
            add(method+'/selection/regret_l2',(np.sqrt(512*measured_error)-np.sqrt(512*oracle_error)).mean(axis=1))
            add(method+'/selection/oracle_choice_rate',np.mean(indices==oracle_choice,axis=1))
            add(method+'/selection/accuracy',np.mean((logits>0)==labels[:,None],axis=1))
            for j,name in enumerate(['P','N','W']):
                add(method+'/selection/'+name+'/pooled_nrmse',measured_error[:,j],np.mean(desired[:,j]**2,axis=1),True)
                add(method+'/selection/'+name+'/regret_l2',np.sqrt(512*measured_error[:,j])-np.sqrt(512*oracle_error[:,j]))
                add(method+'/selection/'+name+'/accuracy',((logits[:,j]>0)==labels).astype(float))
            target_choices[method] = dict(indices=indices.tolist(),requests=queries[indices].tolist(),
                actual_mse=measured_error.tolist(),actual_logits=logits.tolist())
        add('source/desired_accuracy',np.mean((source['logits'][:,singletons]>0)==labels[:,None],axis=1))
        add('source/baseline_accuracy',((source['logits'][:,zero]>0)==labels).astype(float))
        choices_by_target[seed] = target_choices
        saved_arrays.update({f't{seed}__source_actual':source_true,f't{seed}__target_actual':true,
            f't{seed}__M_prediction':predictions['M_source_higher'],f't{seed}__B_prediction':predictions['B']})
        inputs.append(dict(run=str(run.resolve()),index=identity(run/'request_index.json'),panel=identity(run/'panel.json')))
    strata = [np.flatnonzero((labels==y)&(genders==g)) for y in [0,1] for g in [0,1]]
    assert all(len(cell)>0 for cell in strata)
    for value in observations.values():
        value['numerator'] = np.stack(value['numerator'])
        value['denominator'] = np.stack(value['denominator'])
    def statistics(indices):
        values = {}
        for key,item in observations.items():
            numerator = item['numerator'][:,indices].mean()
            denominator = item['denominator'][:,indices].mean()
            value = numerator/denominator if denominator>0 else None
            values[key] = float(np.sqrt(value) if item['root'] else value) if value is not None else None
        return values
    points = statistics(np.arange(len(rows)))
    per_target = {}
    for t,seed in enumerate(choices_by_target):
        current = {}
        for key,item in observations.items():
            numerator,denominator = item['numerator'][t].mean(),item['denominator'][t].mean()
            ratio = numerator/denominator if denominator>0 else None
            current[key] = float(np.sqrt(ratio) if item['root'] else ratio) if ratio is not None else None
        per_target[seed] = current
    print({m:{k:points[m+'/'+k] for k in ['prediction/unseen23/nrmse','selection/pooled_nrmse','selection/regret_l2','selection/accuracy']} for m in methods},flush=True)
    draws = {key:[] for key in points}
    rng = np.random.default_rng(9231546)
    for iteration in range(args.bootstrap):
        sampled = np.concatenate([rng.choice(cell,len(cell),replace=True) for cell in strata])
        for key,value in statistics(sampled).items():
            draws[key].append(value)
        if (iteration+1)%250==0:
            print(f'请求预测共同文档 bootstrap {iteration+1}/{args.bootstrap}',flush=True)
    def summarize(values):
        finite = [v for v in values if v is not None]
        return np.quantile(finite,[.025,.975]).tolist() if finite else None
    contrasts = {}
    for other in ['B','M_additive','source_unit','diagonal_source_higher','original_request','oracle']:
        difference = {}
        for key in points:
            if not key.startswith('M_source_higher/'):
                continue
            suffix = key[len('M_source_higher/'):]
            comparison = other+'/'+suffix
            if comparison not in points:
                continue
            difference[suffix] = dict(value=points[key]-points[comparison] if points[key] is not None and points[comparison] is not None else None,
                ci95=summarize([a-b if a is not None and b is not None else None for a,b in zip(draws[key],draws[comparison])]))
        contrasts['M_source_higher_minus_'+other] = difference
    for method in methods:
        difference = {}
        for key in points:
            prefix = method+'_fixed/selection/'
            if not key.startswith(prefix):
                continue
            suffix = key[len(method+'_fixed/'):]
            comparison = method+'/'+suffix
            difference[suffix] = dict(value=points[key]-points[comparison] if points[key] is not None and points[comparison] is not None else None,
                ci95=summarize([a-b if a is not None and b is not None else None for a,b in zip(draws[key],draws[comparison])]))
        contrasts[method+'_fixed_minus_per_document'] = difference
    for other in ['original_request','oracle','B_fixed','M_additive_fixed','source_unit_fixed','diagonal_source_higher_fixed']:
        difference = {}
        for key in points:
            prefix = 'M_source_higher_fixed/'
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix):]
            comparison = other+'/'+suffix
            difference[suffix] = dict(value=points[key]-points[comparison] if points[key] is not None and points[comparison] is not None else None,
                ci95=summarize([a-b if a is not None and b is not None else None for a,b in zip(draws[key],draws[comparison])]))
        contrasts['M_source_higher_fixed_minus_'+other] = difference
    args.output.mkdir(parents=True)
    save(args.output/'RESULTS.json',dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        statistics={k:dict(value=v,ci95=summarize(draws[k])) for k,v in points.items()},paired_contrasts=contrasts,
        per_target=per_target,
        documents=len(rows),fixed_targets=list(choices_by_target),requests=queries.tolist(),inputs=inputs,
        models=identity(args.models),analyzer=identity(Path(__file__)),head=identity(args.head),
        definitions=dict(prediction='RMSE divided by common actual target effect RMS; documents, requests, and fixed targets equally weighted',
            selection='For each source singleton desired effect choose lexicographically first minimum predicted pooled distance among all 27 requests',
            regret='Actual selected error minus actual full-grid minimum; both per-coordinate squared distance and Euclidean distance saved',
            mechanism='M error = source multilinear interpolation error + true source response with singleton correction minus actual target response',
            deployment='Eight source Boolean forwards per document; source fractional19 used only for interpolation diagnostics'),
        budget=dict(target_calibration_per_seed=32,source_calibration_shared=64,source_deployment_per_document=8,
            fixed_choice_source_runtime_per_document=0,fixed_choice_target_runtime_per_desired_request=1,
            direct_target_grid_per_document=27,target_unfitted_requests=23,source_diagnostic_fractional_per_document=19,
            relation_acquisition='Original native relation acquisition retained as reused cost'),
        bootstrap=dict(draws=args.bootstrap,seed=9231546,unit='Document within four label/gender cells; all requests, methods and fixed targets share draws'),
        cost=dict(cpu_wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu_started,threads=2,gpu_sequences=0)))
    save(args.output/'SELECTIONS.json',dict(documents=documents,targets=choices_by_target))
    np.savez_compressed(args.output/'responses.npz',document_sha256=np.array(documents),labels=labels,genders=genders,requests=queries,**saved_arrays)
    lines = ['# 共同请求结构的确认','','| 方法 | 23请求 nRMSE | 选择 nRMSE | Euclidean regret | 选择准确率 |','|---|---:|---:|---:|---:|']
    for method in methods:
        lines.append('| '+method+' | '+' | '.join(f'{points[method+"/"+k]:.6f}' for k in ['prediction/unseen23/nrmse','selection/pooled_nrmse','selection/regret_l2','selection/accuracy'])+' |')
    lines += ['','| 固定调用 | 选择 nRMSE | Euclidean regret | 准确率 |','|---|---:|---:|---:|']
    for method in [m+'_fixed' for m in methods]+['original_request','oracle']:
        lines.append('| '+method+' | '+' | '.join(f'{points[method+"/selection/"+k]:.6f}' for k in ['pooled_nrmse','regret_l2','accuracy'])+' |')
    (args.output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(dict(output=str(args.output)),flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command',required=True)
    fit = sub.add_parser('fit')
    fit.add_argument('--runs',type=Path,nargs='+',required=True)
    fit.add_argument('--output',type=Path,required=True)
    check = sub.add_parser('check-development')
    check.add_argument('--output',type=Path,required=True)
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--runs',type=Path,nargs='+',required=True)
    for name in ['models','head','output']:
        evaluation.add_argument('--'+name,type=Path,required=True)
    evaluation.add_argument('--bootstrap',type=int,default=1000)
    check_results = sub.add_parser('check-results')
    check_results.add_argument('--output',type=Path,required=True)
    supplement = sub.add_parser('supplementary-contrasts')
    supplement.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    threadpool_limits(limits=2)
    if args.command=='check-development':
        check_development(args.output)
    elif args.command=='fit':
        fit_runs(args.runs,args.output)
    elif args.command=='evaluate':
        evaluate_runs(args)
    elif args.command=='check-results':
        check_results_directory(args.output)
    else:
        supplementary_contrasts(args.output)


def supplementary_contrasts(output):
    path = output/'SUPPLEMENTARY_CONTRASTS.json'
    assert not path.exists()
    choices = read(output/'SELECTIONS.json')['targets']
    arrays = dict(np.load(output/'responses.npz',allow_pickle=False))
    labels,genders,q = arrays['labels'],arrays['genders'],arrays['requests']
    singletons = [np.flatnonzero(np.all(q==r,axis=1)).item() for r in np.eye(3)]
    scale = np.stack([np.mean(arrays[f't{seed}__source_actual'][:,singletons]**2,axis=-1) for seed in choices])
    methods = {}
    for method in ['M_additive_fixed','original_request']:
        mse = np.array([target[method]['actual_mse'] for target in choices.values()])
        logits = np.array([target[method]['actual_logits'] for target in choices.values()])
        oracle = np.array([target['oracle']['actual_mse'] for target in choices.values()])
        methods[method] = dict(mse=mse,accuracy=((logits>0)==labels[None,:,None]).astype(float),
            regret_l2=np.sqrt(512*mse)-np.sqrt(512*oracle),regret_mse=mse-oracle)
    def calculate(indices):
        stats = {}
        for group,requests in [('all',[0,1,2]),('P',[0]),('N',[1]),('W',[2])]:
            for method,value in methods.items():
                current = dict(pooled_nrmse=float(np.sqrt(value['mse'][:,indices][:,:,requests].mean()/scale[:,indices][:,:,requests].mean())),
                    accuracy=float(value['accuracy'][:,indices][:,:,requests].mean()),
                    regret_l2=float(value['regret_l2'][:,indices][:,:,requests].mean()),
                    regret_mse=float(value['regret_mse'][:,indices][:,:,requests].mean()))
                stats.update({method+'/'+group+'/'+k:v for k,v in current.items()})
        return stats
    point = calculate(np.arange(len(labels)))
    draws = {key:[] for key in point}
    strata = [np.flatnonzero((labels==y)&(genders==g)) for y in [0,1] for g in [0,1]]
    rng = np.random.default_rng(9231546)
    for _ in range(1000):
        indices = np.concatenate([rng.choice(c,len(c),replace=True) for c in strata])
        for key,value in calculate(indices).items():
            draws[key].append(value)
    contrast = {}
    for key in point:
        if key.startswith('M_additive_fixed/'):
            suffix = key.removeprefix('M_additive_fixed/')
            other = 'original_request/'+suffix
            contrast[suffix] = dict(value=point[key]-point[other],
                ci95=np.quantile(np.array(draws[key])-np.array(draws[other]),[.025,.975]).tolist())
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        comparison='Predeclared fixed M_additive choice minus original request; all parameters and q frozen before confirmation',
        statistics={k:dict(value=v,ci95=np.quantile(draws[k],[.025,.975]).tolist()) for k,v in point.items()},
        paired_contrasts=contrast,documents=len(labels),fixed_targets=list(choices),
        bootstrap=dict(draws=1000,seed=9231546,unit='Shared document sample within four label/gender cells'),
        inputs={name:identity(output/name) for name in ['RESULTS.json','SELECTIONS.json','responses.npz','DIRECT_CHECK.json']},
        analyzer=identity(Path(__file__)),new_fits=0,new_model_forwards=0)
    save(path,result)
    print({k:v for k,v in contrast.items() if k.startswith('all/')},flush=True)


def check_results_directory(output):
    destination = output/'DIRECT_CHECK.json'
    assert not destination.exists()
    report = read(output/'RESULTS.json')
    selections = read(output/'SELECTIONS.json')
    arrays = dict(np.load(output/'responses.npz',allow_pickle=False))
    q = arrays['requests']
    zero = np.flatnonzero(np.all(q==0,axis=1)).item()
    singleton = [np.flatnonzero(np.all(q==x,axis=1)).item() for x in np.eye(3)]
    unseen = [i for i in range(27) if i not in [zero,*singleton]]
    errors = {}
    desired_energy = []
    selection_error = {}
    nrmse_arrays = {m:[] for m in ['M_source_higher','B']}
    nrmse_scale = []
    for item in report['inputs']:
        run = Path(item['run'])
        index = read(run/'request_index.json')
        seed = str(index['identity']['target_seed'])
        ordered = sorted(index['source_blocks'],key=lambda r:tuple(r['request']))
        with np.load(ordered[zero]['path']) as raw:
            baseline = raw['pooled512'].astype(float)
        raw_targets = []
        for record in sorted(index['target_blocks'],key=lambda r:tuple(r['request'])):
            with np.load(record['path']) as raw:
                raw_targets.append(raw['pooled512'].astype(float)-baseline)
        actual = np.stack(raw_targets,axis=1)
        errors['t'+seed+'_raw_target_array'] = float(np.max(np.abs(actual-arrays['t'+seed+'__target_actual'])))
        desired = arrays['t'+seed+'__source_actual'][:,singleton]
        desired_energy.append(np.mean(desired**2,axis=(1,2)))
        nrmse_scale.append(np.mean(actual[:,unseen]**2,axis=(1,2)))
        for method,key in [('M_source_higher','M'),('B','B')]:
            predicted = arrays[f't{seed}__{key}_prediction']
            nrmse_arrays[method].append(np.mean((predicted[:,unseen]-actual[:,unseen])**2,axis=(1,2)))
            distances = np.sum((predicted[:,None]-desired[:,:,None])**2,axis=-1)
            chosen = np.argmin(distances,axis=-1)
            np.testing.assert_array_equal(chosen,selections['targets'][seed][method]['indices'])
        for method,choice in selections['targets'][seed].items():
            selected = actual[np.arange(len(actual))[:,None],np.array(choice['indices'])]
            mse = np.sum((selected-desired)**2,axis=-1)/512
            errors['t'+seed+'_'+method+'_selected_mse'] = float(np.max(np.abs(mse-choice['actual_mse'])))
            selection_error.setdefault(method,[]).append(mse.mean(axis=1))
    for method,values in nrmse_arrays.items():
        value = float(np.sqrt(np.mean(values)/np.mean(nrmse_scale)))
        errors[method+'_primary_nrmse'] = abs(value-report['statistics'][method+'/prediction/unseen23/nrmse']['value'])
    for method,values in selection_error.items():
        value = float(np.sqrt(np.mean(values)/np.mean(desired_energy)))
        errors[method+'_selection_nrmse'] = abs(value-report['statistics'][method+'/selection/pooled_nrmse']['value'])
    assert max(errors.values())<1e-10
    # 固定简单比较的配对区间只复算已保存选择，不重新拟合或选择请求。
    labels,genders = arrays['labels'],arrays['genders']
    strata = [np.flatnonzero((labels==y)&(genders==g)) for y in [0,1] for g in [0,1]]
    rng = np.random.default_rng(9231546)
    samples = [np.arange(len(labels))]+[np.concatenate([rng.choice(c,len(c),replace=True) for c in strata]) for _ in range(1000)]
    comparisons = {}
    for first,second in [('M_additive_fixed','original_request'),('M_additive','original_request')]:
        differences = []
        a,b,scale = np.array(selection_error[first]),np.array(selection_error[second]),np.array(desired_energy)
        for indices in samples:
            differences.append(float(np.sqrt(a[:,indices].mean()/scale[:,indices].mean())-
                                     np.sqrt(b[:,indices].mean()/scale[:,indices].mean())))
        comparisons[first+'_minus_'+second] = dict(selection_pooled_nrmse=differences[0],ci95=np.quantile(differences[1:],[.025,.975]).tolist())
    save(destination,dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Raw target request files independently reassembled; direct SSE and saved request indices checked without fitting',
        result=identity(output/'RESULTS.json'),analyzer=identity(Path(__file__)),errors=errors,
        maximum_error=max(errors.values()),additional_saved_selection_contrasts=comparisons))
    print(dict(maximum_error=max(errors.values()),additional_saved_selection_contrasts=comparisons),flush=True)


def check_development(output):
    assert not output.exists()
    root = Path('artifacts/scientific_reform_20260923')
    existing = read(root/'INTERACTION_REQUEST_COORDINATES_DEVELOPMENT.json')
    from fit_interaction_request_coordinates import read_existing
    checked = {}
    started = time.perf_counter()
    for seed,values in existing['targets'].items():
        data = read_existing(values['input']['run'],existing['shared_split'])
        indices = existing['actual_fit_indices']
        result = fit_arrays(data['terms'][indices],data['target_effects'][indices,:3])
        checked[seed] = dict(M_max_error=float(np.max(np.abs(np.array(result['M'])-values['M']))),
            B_max_error=float(np.max(np.abs(np.array(result['B'])-values['B']))),fit=result)
        checked[seed]['per_initialization_B_max_error'] = [float(np.max(np.abs(np.array(f['B'])-
            values['initialization_fits'][i]['B']))) for i,f in enumerate(result['initialization_fits'])]
        checked[seed]['selection_rule'] = 'Current numerical ties use nearest identity; historical development used strict minimum SSE'
        assert checked[seed]['M_max_error']<1e-10
        assert max(checked[seed]['per_initialization_B_max_error'])<1e-10
    save(output,dict(checks=checked,cpu_wall_seconds=time.perf_counter()-started,
                     reference=identity(root/'INTERACTION_REQUEST_COORDINATES_DEVELOPMENT.json')))


if __name__ == '__main__':
    main()
