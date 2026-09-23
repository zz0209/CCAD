from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import sys
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '4'

sys.path.insert(0, 'D:/CCAD_Storage/environments/f4_sparse_overlay_v1')

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from analyze_interaction_structure import BITS, OPERATIONS, decompose, digest, summary


def source_polynomial(terms, requests, matrix, derivative=False):
    coordinates = np.asarray(requests)@np.asarray(matrix).reshape(3, 3).T
    values = np.zeros((terms.shape[0], len(requests), terms.shape[2]), dtype=np.float64)
    coordinate_derivative = np.zeros((*values.shape, 3), dtype=np.float64) if derivative else None
    for index, bits in enumerate(BITS):
        axes = [axis for axis in range(3) if bits & (1 << axis)]
        product = np.prod(coordinates[:, axes], axis=1)
        values += terms[:, index, None, :]*product[None, :, None]
        if derivative:
            for axis in axes:
                others = [other for other in axes if other != axis]
                multiplier = np.prod(coordinates[:, others], axis=1) if others else np.ones(len(requests))
                coordinate_derivative[..., axis] += terms[:, index, None, :]*multiplier[None, :, None]
    if not derivative:
        return values
    jacobian = coordinate_derivative[..., :, None]*np.asarray(requests)[None, :, None, None, :]
    return values, jacobian.reshape(*values.shape, 9)


def read_existing(run, split):
    path = Path(run)/'evaluation/original.npz'
    with np.load(path) as data:
        documents = data['document_sha256'].tolist()
        assert [documents[i] for i in split['fit_indices']] == split['fit_documents']
        assert [documents[i] for i in split['check_indices']] == split['check_documents']
        methods, operations = data['method_names'].tolist(), data['operation_names'].tolist()
        order = [operations.index(name) for name in OPERATIONS]
        baseline = data['baseline_pooled512'].astype(np.float64)
        source, _, _ = decompose(data['pooled512'][:, methods.index('source'), order].astype(np.float64), baseline)
        target, effects, _ = decompose(data['pooled512'][:, methods.index('native'), order].astype(np.float64), baseline)
    terms = np.stack([source[name] for name in OPERATIONS], axis=1)
    return dict(terms=terms, target_terms=target, target_effects=effects, documents=documents,
        array_path=str(path), array_sha256=digest(path))


def decode_parameters(parameters):
    exponential = np.exp(parameters.reshape(3, 3))
    matrix = 3*exponential/(1+exponential.sum(1, keepdims=True))
    jacobian = np.zeros((9, 9))
    for row in range(3):
        jacobian[row*3:row*3+3,row*3:row*3+3] = np.diag(matrix[row])-np.outer(matrix[row],matrix[row])/3
    return matrix, jacobian


def linear_design(terms, requests, higher=False):
    design = terms[:, :3].transpose(0,2,1)[:,None,:, :,None]*requests[None,:,None,None,:]
    design = np.broadcast_to(design,(len(terms),len(requests),512,3,3)).reshape(len(terms),len(requests),512,9)
    if higher:
        pair = sum(terms[:,index,None,:]*np.prod(requests[:,[axis for axis in range(3) if bits & (1<<axis)]],axis=1)[None,:,None]
            for index,bits in enumerate(BITS) if bits.bit_count()==2)
        triple = terms[:,6,None,:]*np.prod(requests,axis=1)[None,:,None]
        design = np.concatenate([design,pair[...,None],triple[...,None]],axis=-1)
    return design


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    threadpool_limits(limits=4)
    started, cpu_started = time.perf_counter(), time.process_time()
    root = Path('artifacts/scientific_reform_20260923')
    parent_path = root/'INTERACTION_STRUCTURE_ADDITIVE_SUMMARY.json'
    parent = json.loads(parent_path.read_text())
    suffix = 'SMOKE' if args.smoke else 'DEVELOPMENT'
    output = root/f'INTERACTION_REQUEST_COORDINATES_{suffix}.json'
    assert not output.exists()
    fit = parent['shared_split']['fit_indices'][:2] if args.smoke else parent['shared_split']['fit_indices']
    check = parent['shared_split']['check_indices'][:2] if args.smoke else parent['shared_split']['check_indices']
    requests = np.array([[(bits>>axis)&1 for axis in range(3)] for bits in BITS],dtype=np.float64)
    initial_matrices = [(1-3e-4)*np.eye(3)+1e-4*np.ones((3,3)), .5*np.eye(3)+np.ones((3,3))/6, np.ones((3,3))/3]
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), scope='Real-array smoke only' if args.smoke else 'Exposed development fit8/check56; no independent confirmation',
        parent=dict(path=str(parent_path),sha256=digest(parent_path)), shared_split=parent['shared_split'],
        actual_fit_indices=fit,actual_check_indices=check,targets={},common={},
        specifications=dict(B='B[source,target]=3exp(theta)/(1+row_sum_exp(theta)); theta bounds[-12,12]',
            source_model='Exact seven-Boolean-response multilinear interpolant; nonBoolean values are a prediction model awaiting physical execution',
            source_domain='r in [0,3]^3; values above one strengthen subtraction of the same source contribution',
            anchors=['P','N','W'],loss='Unnormalized fit pooled512 SSE',initial_B=[matrix.tolist() for matrix in initial_matrices],
            max_nfev=5 if args.smoke else 500,ftol=1e-10,xtol=1e-10,gtol=1e-10,
            budgets=dict(B=len(fit)*3,M_additive=len(fit)*3,M_source_higher=len(fit)*3,M_two_scales=len(fit)*4),
            selection='Minimum fit SSE among three fixed initializations',main_check_requests=['PN','PW','NW','PNW']))
    combined = {}
    for item in parent['inputs'][:1] if args.smoke else parent['inputs']:
        run = Path(item['run'])
        data = read_existing(run,parent['shared_split'])
        assert data['array_sha256'] == item['sha256']
        seed = str(json.loads((run/'config.resolved.json').read_text())['target_seed'])
        with np.load(item['probe']) as probe:
            weight = probe['weight'].reshape(-1).astype(np.float64)
        terms, target = data['terms'], data['target_effects']
        source_all = source_polynomial(terms,requests,np.eye(3))
        source_error = max(float(np.max(np.abs(source_all[:,i]-sum(terms[:,j] for j,b in enumerate(BITS) if b & bits == b))))
            for i,bits in enumerate(BITS))
        assert source_error < 1e-11
        def residual(parameters):
            matrix,_ = decode_parameters(parameters)
            return (source_polynomial(terms[fit],requests[:3],matrix)-target[fit,:3]).ravel()
        def jacobian(parameters):
            matrix,chain = decode_parameters(parameters)
            _,jac = source_polynomial(terms[fit],requests[:3],matrix,True)
            return jac.reshape(-1,9)@chain
        initial_parameters = [np.log(matrix/(3-matrix.sum(1,keepdims=True))).ravel() for matrix in initial_matrices]
        point = initial_parameters[1]
        analytic = jacobian(point)
        numerical = np.column_stack([(residual(point+1e-5*np.eye(9)[j])-residual(point-1e-5*np.eye(9)[j]))/2e-5 for j in range(9)])
        derivative_error = float(np.max(np.abs(analytic-numerical)))
        assert derivative_error < 1e-7
        fits = []
        for start_id,parameters in enumerate(initial_parameters):
            fitted = least_squares(residual,parameters,jac=jacobian,bounds=(-12,12),max_nfev=5 if args.smoke else 500,
                ftol=1e-10,xtol=1e-10,gtol=1e-10)
            matrix,_ = decode_parameters(fitted.x)
            fits.append(dict(initialization=start_id,B=matrix.tolist(),theta=fitted.x.tolist(),fit_sse=float(np.sum(fitted.fun**2)),
                success=bool(fitted.success),status=int(fitted.status),message=fitted.message,nfev=int(fitted.nfev),njev=int(fitted.njev),optimality=float(fitted.optimality)))
            print(f'Target {seed}, initialization {start_id+1}/3, nfev {fitted.nfev}, fit SSE {fits[-1]["fit_sse"]:.8g}',flush=True)
        selected = min(fits,key=lambda row:row['fit_sse'])
        matrix = np.array(selected['B'])
        x = linear_design(terms[fit],requests[:3]).reshape(-1,9)
        m,_,rank,_ = np.linalg.lstsq(x,target[fit,:3].ravel(),rcond=None)
        all_linear = linear_design(terms[check],requests)
        additive = all_linear@m
        higher = source_all[check]-linear_design(terms[check],requests)@np.eye(3).ravel()
        anchor_indices = [0,1,2,6]
        x_extra = linear_design(terms[fit],requests[anchor_indices],True).reshape(-1,11)
        extra,_,rank_extra,_ = np.linalg.lstsq(x_extra,target[fit][:,anchor_indices].ravel(),rcond=None)
        predictions = dict(B=source_polynomial(terms[check],requests,matrix),M_additive=additive,
            M_source_higher=additive+higher,M_two_scales=linear_design(terms[check],requests,True)@extra)
        current = dict(input=item,initialization_fits=fits,selected_initialization=selected['initialization'],B=selected['B'],
            row_sums=matrix.sum(1).tolist(),M=m.reshape(3,3).tolist(),M_fit_rank=int(rank),
            extra_information=dict(M=extra[:9].reshape(3,3).tolist(),b2=float(extra[9]),b3=float(extra[10]),rank=int(rank_extra),fit_sse=float(np.sum((x_extra@extra-target[fit][:,anchor_indices].ravel())**2))),
            derivative_maximum_absolute_error=derivative_error,source_boolean_reconstruction_error=source_error,
            requests={},combined_requests={},per_document=[])
        for method,prediction in predictions.items():
            for index,name in enumerate(OPERATIONS):
                actual,src = target[check,index],source_all[check,index]
                current['requests'].setdefault(name,{})[method] = summary(prediction[:,index],actual,src,weight)
                combined.setdefault((name,method),[]).append((prediction[:,index],actual,src))
            for group,indices in [('pairs',[3,4,5]),('four_unfitted',[3,4,5,6])]:
                pred,actual,src = [array[:,indices].reshape(-1,512) for array in (prediction,target[check],source_all[check])]
                current['combined_requests'].setdefault(group,{})[method] = summary(pred,actual,src,weight)
                combined.setdefault((group,method),[]).append((pred,actual,src))
        for position,index in enumerate(check):
            row = dict(document_sha256=data['documents'][index],predictions={})
            for method,prediction in predictions.items():
                row['predictions'][method] = {name:dict(vector_rmse=float(np.sqrt(np.mean((prediction[position,j]-target[index,j])**2))),
                    head_prediction=float(prediction[position,j]@weight),head_actual=float(target[index,j]@weight),
                    head_error=float((prediction[position,j]-target[index,j])@weight)) for j,name in enumerate(OPERATIONS)}
            current['per_document'].append(row)
        result['targets'][seed] = current
    for (name,method),values in combined.items():
        pred,actual,src = [np.concatenate([row[i] for row in values]) for i in range(3)]
        result['common'].setdefault(name,{})[method] = summary(pred,actual,src,weight)
    result['cost'] = dict(driver_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu_started,threads=4,gpu_sequences=0)
    result['script_sha256'] = digest(Path(__file__))
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(output),cost=result['cost'],common={name:{method:stats['nrmse'] for method,stats in methods.items()}
        for name,methods in result['common'].items()})))


if __name__ == '__main__':
    main()
