from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import platform
import sys
import time
import traceback

import numpy as np
sys.path.append('D:/CCAD_Storage/environments/r005a_sparsify_overlay')
import psutil
import torch

sys.path.insert(0,'D:/CCAD_Storage/environments/f4_sparse_overlay_v1')
from scipy import sparse
import scipy
from threadpoolctl import threadpool_limits


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


class Progress:
    def __init__(self,config,directory):
        self.config,self.directory = config,directory
        self.wall,self.cpu = time.perf_counter(),time.process_time()
        self.started_at_utc = datetime.now(timezone.utc).isoformat()
        self.process = psutil.Process()
        self.peak = 0

    def report(self,stage,**fields):
        memory = self.process.memory_info()
        self.peak = max(self.peak,getattr(memory,'peak_wset',memory.rss))
        row = dict(stage=stage,written_at_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-self.wall,
            cpu_seconds=time.process_time()-self.cpu,peak_rss_bytes=self.peak,**fields)
        write(self.directory/'progress.json',row)
        print(json.dumps(row),flush=True)
        if row['wall_seconds']>self.config['budget_seconds']:
            raise TimeoutError('CPU driver budget exceeded; completed stages retained')
        if self.peak>self.config['maximum_ram_bytes']:
            raise MemoryError('RAM budget exceeded')
        return row


def load_codes(path):
    with np.load(path) as data:
        return sparse.csr_matrix((data['values'].astype(np.float64),(data['rows'],data['columns'])),
            shape=tuple(data['shape']))


def fit_direction(gram,cross,rms_output,rms_input,ids_input,k,progress,label):
    count = len(rms_output)
    candidates = np.full((count,k),-1,np.int32)
    beta = np.zeros((count,k),np.float64)
    coefficients = np.zeros_like(beta)
    active = np.flatnonzero(rms_input>0)
    for i in range(count):
        if rms_output[i]==0 or not len(active):
            continue
        score = cross[i,active]/(rms_output[i]*rms_input[active])
        order = np.lexsort((ids_input[active],-score))[:k]
        chosen = active[order]
        candidates[i,:len(chosen)] = chosen
        matrix = gram[np.ix_(chosen,chosen)].astype(np.float64)/np.outer(rms_input[chosen],rms_input[chosen])
        ridge = 1e-3*np.trace(matrix)/len(chosen)
        rhs = cross[i,chosen]/(rms_output[i]*rms_input[chosen])
        solution = np.linalg.solve(matrix+ridge*np.eye(len(chosen)),rhs)
        beta[i,:len(chosen)] = solution
        coefficients[i,:len(chosen)] = solution*rms_output[i]/rms_input[chosen]
        if (i+1)%512==0:
            progress.report('RIDGE',direction=label,completed_features=i+1,total_features=count)
    return candidates,beta,coefficients


def matrix_from_fit(candidates,coefficients,width):
    valid = candidates>=0
    rows = np.broadcast_to(np.arange(len(candidates))[:,None],candidates.shape)[valid]
    result = sparse.csr_matrix((coefficients[valid],(rows,candidates[valid])),shape=(len(candidates),width))
    result.eliminate_zeros()
    return result


def make_tree(c12,b12,c21,b21,ids1,ids2):
    n1,n2 = len(c12),len(c21)
    edges = {}
    for reverse,(candidates,beta) in enumerate(((c12,b12),(c21,b21))):
        for i in range(len(candidates)):
            for j,value in zip(candidates[i],beta[i]):
                if j<0:
                    continue
                pair = (int(j),i) if reverse else (i,int(j))
                item = edges.setdefault(pair,[0.,0.,False,False])
                item[reverse] = float(value*value)
                item[reverse+2] = True
    pairs = sorted(edges,key=lambda pair:(-max(edges[pair][:2]),int(ids1[pair[0]]),int(ids2[pair[1]])))
    weights = np.array([max(edges[pair][:2]) for pair in pairs])
    parents = np.arange(n1+n2)
    node = np.arange(n1+n2)
    children,merge_edge = [],[]

    def root(i):
        while parents[i]!=i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for ei,((a,b),weight) in enumerate(zip(pairs,weights)):
        if weight==0:
            continue
        a,b = root(a),root(n1+b)
        if a==b:
            continue
        if a>b:
            a,b = b,a
        children.append([node[a],node[b]])
        merge_edge.append(ei)
        parents[b] = a
        node[a] = n1+n2+len(children)-1
    roots = sorted({int(node[root(i)]) for i in range(n1+n2)})
    return dict(children=np.array(children,np.int32).reshape(-1,2),merge_edge=np.array(merge_edge,np.int32),
        edge_members=np.array(pairs,np.int32),edge_weight=weights,
        edge_directional_energy=np.array([edges[pair][:2] for pair in pairs]),
        candidate_present=np.array([edges[pair][2:] for pair in pairs],bool),roots=np.array(roots,np.int32))


def contribution_gram(z1,d1,z2,d2,progress,label,block=256):
    result = np.empty((z1.shape[1],z2.shape[1]),np.float32)
    for start in range(0,len(result),block):
        stop = min(start+block,len(result))
        code = (z1[:,start:stop].T@z2).toarray()/z1.shape[0]
        result[start:stop] = code*(d1[start:stop]@d2.T)
        progress.report('CONTRIBUTION_GRAM',matrix=label,completed_features=stop,total_features=len(result))
    return result


def cross_sum(matrix,left,right):
    if not len(left) or not len(right):
        return 0.
    return float(matrix[np.ix_(left,right)].sum(dtype=np.float64))


def tree_metrics(tree,g11,g22,g12,mean1,mean2,fixed1,fixed2,pw,random_map):
    n1,n2 = len(g11),len(g22)
    leaves = n1+n2
    total = leaves+len(tree['children'])
    # 每个有效组件保留成员与均值；失效子组件及时释放。
    active = {}
    sizes = np.zeros((total,2),np.int32)
    moments = np.zeros((total,7),np.float64)
    means = np.zeros((total,7),np.float64)
    fixed_means = np.zeros_like(means)
    center_adjustment = np.zeros_like(means)
    for i in range(leaves):
        a = np.array([i],np.int32) if i<n1 else np.empty(0,np.int32)
        b = np.array([i-n1],np.int32) if i>=n1 else np.empty(0,np.int32)
        vectors = np.zeros((8,mean1.shape[1]),np.float64)
        if len(a):
            vectors[0],vectors[2] = mean1[i],mean2[pw[i]]
            vectors[4],vectors[6] = fixed1[i],fixed2[pw[i]]
            moments[i,[0,3,4]] = [g11[i,i],g22[pw[i],pw[i]],g12[i,pw[i]]]
        else:
            j = i-n1
            vectors[1],vectors[3] = mean2[j],mean2[random_map[j]]
            vectors[5],vectors[7] = fixed2[j],fixed2[random_map[j]]
            moments[i,[1,5]] = [g22[j,j],g22[random_map[j],random_map[j]]]
        active[i] = (a,b,vectors)
        sizes[i] = [len(a),len(b)]
        means[i] = vector_moments(vectors[:4])
        fixed_means[i] = vector_moments(vectors[4:])
        center_adjustment[i] = vector_moments(vectors[:4]-vectors[4:])-means[i]
    running = moments[:leaves].sum(0)
    running_mean = means[:leaves].sum(0)
    running_fixed = fixed_means[:leaves].sum(0)
    running_adjustment = center_adjustment[:leaves].sum(0)
    coverage = np.zeros(2,np.int64)
    mixed,max_size = 0,1
    curve = [[0,leaves,0,1,0,0,*running,*running_mean,*running_fixed,*running_adjustment]]
    for step,(left,right) in enumerate(tree['children']):
        node = leaves+step
        a,b,va = active.pop(int(left))
        c,d,vb = active.pop(int(right))
        increase = np.array([2*cross_sum(g11,a,c),2*cross_sum(g22,b,d),
            cross_sum(g12,a,d)+cross_sum(g12,c,b),2*cross_sum(g22,pw[a],pw[c]),
            cross_sum(g12,a,pw[c])+cross_sum(g12,c,pw[a]),2*cross_sum(g22,random_map[b],random_map[d]),
            cross_sum(g12,a,random_map[d])+cross_sum(g12,c,random_map[b])])
        moments[node] = moments[left]+moments[right]+increase
        vectors = va+vb
        means[node] = vector_moments(vectors[:4])
        fixed_means[node] = vector_moments(vectors[4:])
        center_adjustment[node] = vector_moments(vectors[:4]-vectors[4:])-means[node]
        sizes[node] = sizes[left]+sizes[right]
        active[node] = (np.concatenate((a,c)),np.concatenate((b,d)),vectors)
        running += increase
        running_mean += means[node]-means[left]-means[right]
        running_fixed += fixed_means[node]-fixed_means[left]-fixed_means[right]
        running_adjustment += center_adjustment[node]-center_adjustment[left]-center_adjustment[right]
        for child,sign in ((left,-1),(right,-1),(node,1)):
            if np.all(sizes[child]>0):
                coverage += sign*sizes[child]
                mixed += sign
        max_size = max(max_size,int(sizes[node].sum()))
        curve.append([step+1,leaves-step-1,mixed,max_size,*coverage,*running,*running_mean,*running_fixed,*running_adjustment])
    return sizes,moments,means,fixed_means,center_adjustment,np.asarray(curve,np.float64)


def vector_moments(v):
    return np.array([v[0]@v[0],v[1]@v[1],v[0]@v[1],v[2]@v[2],v[0]@v[2],v[3]@v[3],v[0]@v[3]])


def members(tree,node,n1,n2):
    todo,a,b = [int(node)],[],[]
    while todo:
        item = todo.pop()
        if item<n1:
            a.append(item)
        elif item<n1+n2:
            b.append(item-n1)
        else:
            todo.extend(tree['children'][item-n1-n2].tolist())
    return np.array(sorted(a),np.int32),np.array(sorted(b),np.int32)


def error_statistics(moments):
    source,target,cross,pw_target,pw_cross,random_target,random_cross = map(float,moments)
    errors = dict(graph=source+target-2*cross,PW=source+pw_target-2*pw_cross,
        random=source+random_target-2*random_cross)
    return dict(source_energy=source,target_energy=target,PW_target_energy=pw_target,
        random_target_energy=random_target,error_squared=errors,
        nrmse={key:float(np.sqrt(max(value,0)/source)) if source>0 else None for key,value in errors.items()})


def decompose(z1,d1,z2,d2,relation,a,b,m1,m2):
    x = np.asarray(z1[:,a]@d1[a])
    y = np.asarray((z2@relation[a].T)@d1[a])
    t = np.asarray(z2[:,b]@(relation[:,b].T@d1))
    w = np.asarray(z2[:,b]@d2[b])
    fields = [x-y,y-t,t-w]
    actual = x-w
    residual = fields[0]+fields[1]+fields[2]-actual
    raw = np.array([[np.einsum('ij,ij->',a,b)/len(a) for b in fields] for a in fields])
    center = np.stack([value.mean(0) for value in fields])
    mean = center@center.T
    x0 = m1[a]@d1[a]
    y0 = np.asarray(relation[a]@m2)@d1[a]
    t0 = m2[b]@(relation[:,b].T@d1)
    w0 = m2[b]@d2[b]
    independent_mean = np.stack((x0-y0,y0-t0,t0-w0))
    independent_mean_gram = independent_mean@independent_mean.T
    fixed_centered = raw-independent_mean@center.T-center@independent_mean.T+independent_mean_gram
    actual_energy = float(np.einsum('ij,ij->',actual,actual)/len(actual))
    return dict(term_names=['readout','commutator','decoder'],uncentered_gram=raw.tolist(),
        empirical_mean_gram=mean.tolist(),empirical_centered_gram=(raw-mean).tolist(),
        independent_mean_gram=independent_mean_gram.tolist(),fixed_centered_gram=fixed_centered.tolist(),
        independent_mean_residual_cross_gram=(raw-independent_mean_gram-fixed_centered).tolist(),
        actual_error_squared=actual_energy,
        summed_terms_squared=float(raw.sum()),identity_max_absolute_error=float(np.abs(residual).max()),
        identity_squared_error_difference=float(raw.sum()-actual_energy))


def analyze(config,objective,directory,progress):
    completed = directory/f'{objective}_result.json'
    if completed.exists():
        return json.loads(completed.read_text())
    cache = Path(config['reference_run'])
    paths = [cache/f'natural_{split}_{objective}_seed{seed}.npz'
        for split in ('discovery','calibration','mean') for seed in (1,2)]
    checkpoint_paths = [Path(config['checkpoint_directory'])/f'{objective}_seed{seed}.pt' for seed in (1,2)]
    pw_path = Path(config['pw_run'])/f'{objective}_s1_t2_global_pw.npz'
    with np.load(pw_path) as saved:
        permutation = saved['permutation'].astype(np.int64)
    assert len(permutation)==8192 and len(np.unique(permutation))==8192
    count = config.get('feature_limit',8192)
    ids1 = np.arange(count)
    ids2 = np.sort(permutation[ids1]) if count<8192 else np.arange(8192)
    inverse = {int(value):i for i,value in enumerate(ids2)}
    pw = np.array([inverse[int(permutation[i])] for i in ids1],np.int32)
    random_map = np.random.default_rng(config['random_seed']).permutation(len(ids2))
    z1,z2,c1,c2,m1,m2 = [load_codes(path) for path in paths]
    assert z1.shape==z2.shape==(8192,8192) and c1.shape==c2.shape==(4096,8192)
    assert m1.shape==m2.shape==(2048,8192)
    m1,m2 = np.asarray(m1[:,ids1].mean(0)).ravel(),np.asarray(m2[:,ids2].mean(0)).ravel()
    z1,c1,z2,c2 = z1[:,ids1],c1[:,ids1],z2[:,ids2],c2[:,ids2]
    decoders = []
    for path,ids in zip(checkpoint_paths,(ids1,ids2)):
        state = torch.load(path,map_location='cpu',weights_only=True)
        assert objective in ('topk','matryoshka')
        decoder = state['decoder.weight'].T if objective=='topk' else state['W_dec']
        assert tuple(decoder.shape)==(8192,1024)
        decoders.append(decoder.numpy()[ids].copy())
        del decoder
        del state
    d1,d2 = decoders
    identity = {str(path):digest(path) for path in [*paths,*checkpoint_paths,pw_path]}
    fit_path = directory/f'{objective}_ridge.npz'
    if not fit_path.exists():
        progress.report('DISCOVERY_GRAMS',objective=objective)
        g11 = (z1.T@z1).toarray()/z1.shape[0]
        g22 = (z2.T@z2).toarray()/z2.shape[0]
        g12 = (z1.T@z2).toarray()/z1.shape[0]
        rms1,rms2 = np.sqrt(g11.diagonal()),np.sqrt(g22.diagonal())
        c12,b12,r12 = fit_direction(g22,g12,rms1,rms2,ids2,config['candidates'],progress,'2_to_1')
        c21,b21,r21 = fit_direction(g11,g12.T,rms2,rms1,ids1,config['candidates'],progress,'1_to_2')
        np.savez_compressed(fit_path,c12=c12,b12=b12,r12=r12,c21=c21,b21=b21,r21=r21,
            rms1=rms1,rms2=rms2,source1_ids=ids1,source2_ids=ids2)
        del g11,g22,g12
    with np.load(fit_path) as fit:
        relation12 = matrix_from_fit(fit['c12'],fit['r12'],len(ids2))
        relation21 = matrix_from_fit(fit['c21'],fit['r21'],len(ids1))
        tree = make_tree(fit['c12'],fit['b12'],fit['c21'],fit['b21'],ids1,ids2)
        zero_rms = [int((fit[key]==0).sum()) for key in ('rms1','rms2')]
    tree_path = directory/f'{objective}_tree.npz'
    np.savez_compressed(tree_path,**tree,source1_ids=ids1,source2_ids=ids2)
    progress.report('TREE_COMPLETE',objective=objective,merges=len(tree['children']),roots=len(tree['roots']))
    g11 = contribution_gram(c1,d1,c1,d1,progress,'11',config['gram_block'])
    g22 = contribution_gram(c2,d2,c2,d2,progress,'22',config['gram_block'])
    g12 = contribution_gram(c1,d1,c2,d2,progress,'12',config['gram_block'])
    mean1 = np.asarray(c1.mean(0)).ravel()[:,None]*d1
    mean2 = np.asarray(c2.mean(0)).ravel()[:,None]*d2
    sizes,moments,means,fixed_means,adjustment,curve = tree_metrics(tree,g11,g22,g12,mean1,mean2,
        m1[:,None]*d1,m2[:,None]*d2,pw,random_map)
    del g11,g22,g12
    metric_path = directory/f'{objective}_nodes.npz'
    np.savez_compressed(metric_path,sizes=sizes,moments=moments,empirical_mean_moments=means,
        empirical_centered_moments=moments-means,independent_mean_moments=fixed_means,
        fixed_centered_moments=moments+adjustment,independent_mean_residual_cross_moments=-fixed_means-adjustment,
        coverage_curve=curve,PW_local_permutation=pw,random_local_permutation=random_map)
    eligible = np.flatnonzero(np.all(sizes>0,axis=1))
    chosen = []
    for requested in config['representative_sizes']:
        if len(eligible):
            chosen.append(int(eligible[np.lexsort((eligible,np.abs(sizes[eligible].sum(1)-requested)))[0]]))
    chosen = sorted(set(chosen))
    representatives = []
    for node in chosen:
        a,b = members(tree,node,len(ids1),len(ids2))
        item = dict(node=node,source1_members=ids1[a].tolist(),source2_members=ids2[b].tolist(),
            PW_source2_members=ids2[pw[a]].tolist(),random_source2_members=ids2[random_map[b]].tolist(),
            raw=error_statistics(moments[node]),empirical_mean=error_statistics(means[node]),
            empirical_centered=error_statistics(moments[node]-means[node]),
            independent_mean=error_statistics(fixed_means[node]),fixed_centered=error_statistics(moments[node]+adjustment[node]),
            independent_mean_residual_cross_moments=(-fixed_means[node]-adjustment[node]).tolist(),
            decomposition_2_to_1=decompose(c1,d1,c2,d2,relation12,a,b,m1,m2),
            decomposition_1_to_2=decompose(c2,d2,c1,d1,relation21,b,a,m2,m1))
        item['gram_vs_direct_actual_error_squared_difference'] = (
            item['raw']['error_squared']['graph']-item['decomposition_2_to_1']['actual_error_squared'])
        representatives.append(item)
        progress.report('DECOMPOSITION',objective=objective,node=node,members=len(a)+len(b))
    result = dict(objective=objective,identity=identity,source1_ids=ids1.tolist(),source2_ids=ids2.tolist(),
        discovery_positions=8192,calibration_positions=4096,independent_mean_positions=2048,zero_rms_columns=zero_rms,
        ridge_path=str(fit_path),tree_path=str(tree_path),node_metrics_path=str(metric_path),
        node_count=len(sizes),root_count=len(tree['roots']),representatives=representatives,
        representative_selection='Nearest total member count to each configured size, mixed components only, ties by node ID; discovery tree only',
        moment_columns=['source1_energy','source2_energy','source1_source2_inner','PW_target_energy','PW_cross','random_target_energy','random_cross'],
        curve_columns=['merge_count','component_count','mixed_component_count','largest_total_members','covered_source1','covered_source2',
            'source1_energy','source2_energy','cross','PW_target_energy','PW_cross','random_target_energy','random_cross',
            'empirical_mean_source1_energy','empirical_mean_source2_energy','empirical_mean_cross','empirical_mean_PW_target_energy','empirical_mean_PW_cross','empirical_mean_random_target_energy','empirical_mean_random_cross',
            'independent_mean_source1_energy','independent_mean_source2_energy','independent_mean_cross','independent_mean_PW_target_energy','independent_mean_PW_cross','independent_mean_random_target_energy','independent_mean_random_cross',
            'fixed_center_adjustment_source1','fixed_center_adjustment_source2','fixed_center_adjustment_cross','fixed_center_adjustment_PW_target','fixed_center_adjustment_PW_cross','fixed_center_adjustment_random_target','fixed_center_adjustment_random_cross'],
        units='Mean squared vector norm over calibration positions; fixed centering uses independent 2048-position means; empirical variance is separate; nRMSE denominator is source contribution energy',
        graph_weight='Maximum of the two signed RMS ridge coefficient squares; candidate presence saved separately; not joint explained variance')
    write(completed,result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    directory = Path(config['run_storage_root'])/config['run_id']
    directory.mkdir(parents=True,exist_ok=args.resume)
    if args.resume:
        assert json.loads((directory/'config.json').read_text())==config
        assert digest(directory/'source_snapshot.py')==digest(__file__)
    else:
        write(directory/'config.json',config)
        (directory/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    progress = Progress(config,directory)
    write(directory/'status.json',dict(status='RUNNING',started_at_utc=progress.started_at_utc))
    write(directory/'environment.json',dict(python=sys.executable,python_version=platform.python_version(),
        numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,psutil=psutil.__version__,
        cpu_threads=config['cpu_threads'],gpu_used=False))
    try:
        torch.set_num_threads(config['cpu_threads'])
        with threadpool_limits(config['cpu_threads']):
            results = [analyze(config,objective,directory,progress) for objective in config['objectives']]
        cost = progress.report('COMPLETE')
        cost['started_at_utc'] = progress.started_at_utc
        cost['ended_at_utc'] = datetime.now(timezone.utc).isoformat()
        cost['generated_array_bytes'] = sum(path.stat().st_size for path in directory.glob('*.npz'))
        cost['generated_bytes_before_final_reports'] = sum(path.stat().st_size for path in directory.iterdir() if path.is_file())
        provenance = {path:digest(path) for path in config['identity_files']}
        result = dict(run_id=config['run_id'],status='PASS',smoke=config['feature_limit']<8192,
            scope=config['scope'],source_seeds=[1,2],objectives=results,code_sha256=digest(__file__),
            identity_files=provenance,cost=cost,calibration_use='Observation only; no R or node selection from calibration values')
        write(directory/'result.json',result)
        destination = Path(config['output'])
        destination.parent.mkdir(parents=True,exist_ok=True)
        assert not destination.exists()
        write(destination,result)
        generated_bytes = sum(path.stat().st_size for path in directory.iterdir() if path.is_file())+destination.stat().st_size
        if generated_bytes>config['maximum_new_bytes']:
            raise OSError('Generated output exceeds the configured storage budget')
        write(directory/'status.json',dict(status='PASS',cost=cost))
    except Exception:
        write(directory/'status.json',dict(status='FAIL',error=traceback.format_exc(),
            started_at_utc=progress.started_at_utc,ended_at_utc=datetime.now(timezone.utc).isoformat(),
            peak_rss_bytes=progress.peak,
            wall_seconds=time.perf_counter()-progress.wall,cpu_seconds=time.process_time()-progress.cpu))
        raise


if __name__=='__main__':
    main()
