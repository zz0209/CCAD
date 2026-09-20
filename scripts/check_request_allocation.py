from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import torch
from scipy.optimize import LinearConstraint, linprog, lsq_linear, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / 'src'), 'D:/CCAD_Storage/environments/r005a_dictionary_overlay',
                'D:/CCAD_Storage/references/source/dictionary_learning_60ec6bf'])
from dictionary_learning.trainers.top_k import AutoEncoderTopK
from ccad.intervention_transport import transport_delta, refine_columns


def vertex_requests(parts):
    return np.array([[(mask >> j) & 1 for j in range(parts)] for mask in range(1, 2**parts)], dtype=float)


def fit_common(decoder, source, capacity, queries, initial):
    members, parts = initial.shape
    truth = source @ queries.T
    scale = max(float(np.square(truth).sum() / len(queries)), 1e-12)
    gram, rhs = decoder.T @ decoder, decoder.T @ source
    moment = queries.T @ queries / len(queries)
    constraint = LinearConstraint(np.kron(np.eye(members), queries), -np.repeat(capacity, len(queries)), np.inf)

    def objective(flat):
        error = decoder @ flat.reshape(members, parts) @ queries.T - truth
        return float(np.square(error).sum() / len(queries) / scale)

    def gradient(flat):
        return (2 * (gram @ flat.reshape(members, parts) - rhs) @ moment / scale).ravel()

    solved = minimize(objective, initial.ravel(), jac=gradient, method='SLSQP', constraints=constraint,
                      options=dict(ftol=1e-11, maxiter=2000))
    assert solved.success, solved.message
    matrix = solved.x.reshape(members, parts)
    assert (capacity[:, None] + matrix @ queries.T).min() >= -1e-7
    independent = []
    for request in queries:
        fitted = lsq_linear(decoder, source @ request, bounds=(-capacity, np.inf), tol=1e-11,
                            max_iter=2000, lsq_solver='exact')
        assert fitted.success, fitted.message
        independent.append(fitted.x)
    adjusted = np.array(independent).T
    common_error = objective(solved.x)
    adjusted_error = float(np.square(decoder @ adjusted - truth).sum() / len(queries) / scale)
    assert adjusted_error <= common_error + 1e-6
    return dict(common_relative_mse=common_error, request_relative_mse=adjusted_error,
                reduction=common_error-adjusted_error, common_iterations=int(solved.nit),
                source_energy=scale), matrix, adjusted


def witness():
    decoder = np.array([[1., 0., 1.], [0., 1., 1.]])
    capacity = np.array([0., 0., 1.])
    source = -np.eye(2)
    queries = vertex_requests(2)
    equality = np.kron(decoder, np.eye(2))
    inequalities = -np.kron(np.eye(3), queries)
    exact = linprog(np.zeros(6), A_ub=inequalities, b_ub=np.repeat(capacity, len(queries)),
                    A_eq=equality, b_eq=source.ravel(), bounds=[(None, None)]*6, method='highs')
    assert exact.status == 2
    result, matrix, _ = fit_common(decoder, source, capacity, queries, np.zeros((3, 2)))
    assert abs(result['common_relative_mse'] - 3/16) < 1e-7
    assert result['request_relative_mse'] < 1e-10
    grid = np.array([(a, b) for a in np.linspace(0, 1, 101) for b in np.linspace(0, 1, 101)])
    update = np.column_stack([np.maximum(grid[:, 1]-grid[:, 0], 0),
                             np.maximum(grid[:, 0]-grid[:, 1], 0), -grid.max(1)])
    assert np.allclose(update @ decoder.T, -grid, atol=1e-14)
    assert (capacity+update).min() >= -1e-14
    return dict(**result, affine_exact_lp_status=int(exact.status), optimal_common=matrix.tolist(),
                grid_points=len(grid), maximum_physical_error=float(np.abs(update @ decoder.T+grid).max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--states', type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    torch.set_num_threads(2)
    config = json.loads((ROOT / 'configs/profile_confirmation_t2_20260920.json').read_text())
    profile = torch.load(config['source_metric_cache'], map_location='cpu', weights_only=True)
    natural = torch.load(config['natural_cache'], map_location='cpu', weights_only=True)
    sd = torch.load(Path(config['target_directory']) / 'resid_4_seed2.pt', map_location='cpu', weights_only=True)
    target = AutoEncoderTopK(512, sd['encoder.weight'].shape[0], int(sd['k']))
    target.load_state_dict(sd)
    target.requires_grad_(False)
    bank = np.load(ROOT / config['grammar_source_parameters'])
    source = {k: torch.tensor(bank[k]) for k in ['center', 'encoder', 'encoder_bias', 'decoder']}
    states = natural['resid_4']
    activations = torch.relu((states-source['center']) @ source['encoder'].T+source['encoder_bias'])
    eligible = torch.nonzero((activations > 0).sum(-1) >= 2).flatten()
    chosen = eligible[:args.states]
    assert len(chosen) == args.states, (len(eligible), args.states)
    result = dict(run_id='REQUEST_ALLOCATION_GEOMETRY_20260920', written_at_utc=datetime.now(timezone.utc).isoformat(),
                  status='RUNNING', mathematical_witness=witness(), natural_states=len(states),
                  eligible_states=len(eligible), selected_indices=chosen.tolist(), observations=[],
                  scope='Development geometry diagnostic on the first cached natural states with at least two active source members. No language-model response evaluation.',
                  comparison='Identical selected target members, metric and all 15 nonempty request vertices; common coefficients optimized for the same request moment.',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for index in chosen.tolist():
        h = states[index:index+1]
        zs = activations[index].numpy().astype(float)
        _, _, columns = transport_delta(h, target, source, torch.ones(4), target.encoder.weight, 8, False)
        candidates = -(target.encoder.weight @ source['decoder'].T) * activations[index][None]
        ids = (candidates.abs().sum(-1)*target.decoder.weight.norm(dim=0)).topk(8).indices
        decoder = target.decoder.weight[:, ids].numpy().astype(float)
        capacity = target.encode(h)[0, ids].numpy().astype(float)
        physical = -source['decoder'].numpy().T.astype(float) * zs[None]
        for name, metric_root in [('identity', torch.eye(512)), ('source_profile', profile['grammar']['resid_4'])]:
            initial = refine_columns(h, target, source, columns, 8, 128, metric_root=metric_root, accelerate=True)[0, ids]
            root = metric_root.numpy().astype(float)
            values, common, requested = fit_common(root.T @ decoder, root.T @ physical, capacity,
                                                   vertex_requests(4), initial.numpy().astype(float))
            result['observations'].append(dict(state=index, metric=name, source_active=int((zs>0).sum()),
                                               selected_members=ids.tolist(), **values))
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(result['observations'][-1]), flush=True)
    result.update(status='PASS', wall_seconds=time.perf_counter()-started,
                  completed_at_utc=datetime.now(timezone.utc).isoformat())
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], wall_seconds=result['wall_seconds'])), flush=True)


if __name__ == '__main__':
    main()
