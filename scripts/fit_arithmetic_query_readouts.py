"""Fit source-member predictors on the same frozen source pair states.

Raw ridge predicts member amplitudes from hidden differences. Budgeted sparse
activation prediction uses normalized target-code differences, a 512-row pool,
and a shared 64-row union across roles. Both decode into the source dictionary
and retain every source-member column for later requests. No target outputs.
"""
import os
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import sys
import platform
import traceback
import time
import hashlib
import numpy as np
from run_causalgym_multisite import MultisiteWork, ROOT, write


def ridge(x, y, fraction):
    k = x @ x.T / len(x)
    penalty = fraction * max(np.trace(k)/len(k), 1e-12)
    return x.T @ np.linalg.solve(k+penalty*np.eye(len(k)), y)/len(x)


def sparse(x, y, alpha):
    from sklearn.linear_model import MultiTaskLasso
    scale_x = np.sqrt((x*x).mean(0)).clip(1e-10)
    scale_y = np.sqrt((y*y).mean(0)).clip(1e-10)
    model = MultiTaskLasso(alpha=alpha, fit_intercept=False, max_iter=30000, tol=1e-6, selection='cyclic')
    model.fit(x/scale_x, y/scale_y)
    coef = model.coef_.T * scale_y[None, :] / scale_x[:, None]
    return coef, dict(iterations=int(model.n_iter_), dual_gap=float(model.dual_gap_))


def fit_operation(x, h, y, roles, nroles, spec):
    energy = np.sqrt((x*x).mean(0)).clip(1e-10)
    ys = np.sqrt((y*y).mean(0)).clip(1e-10)
    correlation = (x/energy).T @ (y/ys) / len(x)
    pool = np.argsort(-np.square(correlation).sum(1), kind='stable')[:spec['pool']]
    prelim = {}
    importance = np.zeros(len(pool))
    for role in range(nroles):
        take = roles == role
        if not take.any() or not y[take].any():
            continue
        coef, info = sparse(x[take][:, pool], y[take], spec['screen_alpha'])
        importance += (np.square(coef).sum(1) * np.square(energy[pool]))
        prelim[role] = info
    selected = pool[np.argsort(-importance, kind='stable')[:spec['members']]]
    raw = np.zeros((nroles, h.shape[1], y.shape[1]), dtype=np.float32)
    activation = np.zeros((nroles, len(selected), y.shape[1]), dtype=np.float32)
    details = []
    for role in range(nroles):
        take = np.where(roles == role)[0]
        if len(take) == 0 or not y[take].any():
            continue
        # Inner source-fit selection only; the target functional panel is never read.
        perm = np.random.default_rng(spec['fold_seed']+role).permutation(len(take))
        train, valid = take[perm[16:]], take[perm[:16]]
        assert len(train) and len(valid)
        trials = []
        for fraction in spec['ridge_grid']:
            coef = ridge(h[train], y[train], fraction)
            error = float(np.square(h[valid]@coef-y[valid]).sum()/max(np.square(y[valid]).sum(), 1e-12))
            trials.append((error, fraction))
        _, fraction = min(trials)
        raw[role] = ridge(h[take], y[take], fraction)
        lasso_trials = []
        for alpha in spec['lasso_grid']:
            coef, info = sparse(x[train][:, selected], y[train], alpha)
            error = float(np.square(x[valid][:, selected]@coef-y[valid]).sum()/max(np.square(y[valid]).sum(), 1e-12))
            lasso_trials.append((error, alpha, info))
        _, alpha, _ = min(lasso_trials, key=lambda v: v[0])
        activation[role], final = sparse(x[take][:, selected], y[take], alpha)
        details.append(dict(role=role, sites=len(take), ridge_fraction=fraction, ridge_inner_scores=trials,
                            lasso_alpha=alpha, lasso_inner_scores=lasso_trials, lasso_final=final,
                            source_target_constant_mse=float(np.square(y[take]).sum())))
    return raw, activation, selected, dict(roles=details, screening=prelim,
        selection='Pool/union screen uses all permitted source-fit states; inner penalty scores are development estimates, not independent performance evidence.')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', type=Path, required=True); args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    w = MultisiteWork(cfg, args.config, ['scripts/fit_arithmetic_query_readouts.py', 'scripts/run_causalgym_multisite.py', 'src/ccad/artifacts.py'])
    error = None
    try:
        import torch, sklearn, scipy
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('high')
        torch.use_deterministic_algorithms(True)
        spec = cfg['readout_fit']; parent = ROOT/spec['relation_run']; cache = ROOT/spec['source_cache_identity']
        assert json.loads(w.checked(parent/'status.json').read_text())['status'] == 'PASS'
        old = json.loads(w.checked(parent/'config.resolved.json').read_text())
        panel = json.loads(w.checked(parent/'panel.json').read_text())
        with np.load(w.checked(cache/'states.npz')) as z:
            hidden = z['hidden']
        original_count = len(hidden)
        with np.load(w.checked(parent/'source_view_states.npz')) as z:
            hidden = np.concatenate([hidden, z['hidden']], axis=0)
        assert len(hidden) == len(panel['rows'])
        tc = json.loads(w.checked(ROOT/cfg['training_run']/'config.resolved.json').read_text())
        sys.path.extend([tc['dictionary_source_dir'], tc['dictionary_overlay_dir']])
        w.checked(Path(tc['dictionary_source_dir'])/'dictionary_learning/trainers/top_k.py')
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        snaps = json.loads(w.checked(ROOT/cfg['training_run']/'checkpoints.json').read_text())['checkpoints']
        def load(seed):
            snap = next(p for p in snaps if p['seed']==seed and p['step']==cfg['checkpoint_step'])
            checkpoint = w.checked(snap['path'])
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == snap['sha256']
            ae = AutoEncoderTopK(hidden.shape[-1], tc['dict_size'], tc['k'])
            ae.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
            ae.to(cfg.get('encoder_device','cpu')); ae.eval(); ae.requires_grad_(False)
            with np.load(w.checked(cache/f'source_seed{seed}.npz')) as z:
                codes = z['codes']
            with torch.no_grad():
                witness = ae.encode(torch.tensor(hidden[:256].reshape(-1, hidden.shape[-1]),device=cfg.get('encoder_device','cpu'))).cpu().numpy().reshape(codes[:256].shape)
                encoding_error = float(np.max(np.abs(witness-codes[:256])))
                assert encoding_error < .002, encoding_error
                view = ae.encode(torch.tensor(hidden[original_count:].reshape(-1, hidden.shape[-1]),device=cfg.get('encoder_device','cpu'))).cpu().numpy().reshape(len(hidden)-original_count, hidden.shape[1], -1)
            return ae.decoder.weight.T.cpu().numpy(), np.concatenate([codes, view]), encoding_error
        for s, t in old['relation_transfer']['seed_pairs']:
            with np.load(w.checked(parent/f'relation_s{s}_t{t}.npz')) as z:
                pp, gate = z['fit_pairs'], z['source_gate']
                sources = {op: z[f'{op}_source_indices'] for op in ['unit', 'tens']}
            assert all(panel['rows'][int(i)]['split']=='fit' for i in pp.ravel())
            ii, jj = pp.T
            decoder, scode, se = load(s)
            source = scode[jj]-scode[ii]; del scode
            _, tcode, te = load(t)
            target = tcode[jj]-tcode[ii]; del tcode
            rawh = hidden[jj]-hidden[ii]
            shift = np.array([1-panel['rows'][int(i)]['template'] for i in ii])
            roles = (np.arange(rawh.shape[1])[None]+shift[:, None]).ravel()
            payload = dict(fit_pairs=pp, source_gate=gate)
            for c, op in enumerate(['unit', 'tens']):
                si = sources[op]
                y = source[..., si].reshape(-1, len(si))*gate[roles][:, si, c]
                raw, activation, selected, info = fit_operation(target.reshape(-1, target.shape[-1]).astype(float),
                    rawh.reshape(-1, rawh.shape[-1]).astype(float), y.astype(float), roles, gate.shape[0], spec)
                payload.update({f'{op}_raw': raw, f'{op}_activation': activation, f'{op}_target_indices': selected,
                                f'{op}_source_indices': si, f'{op}_decoder': decoder[si]})
                w.record(kind='readout_fit', task='arithmetic_source_fit', row_id=c,
                         component=f'source{s}_target{t}', method='raw_and_activation', mode='k64',
                         seed=s, source_seed=s, target_seed=t, operation=op,
                         source_encoding_max_error=se, target_encoding_max_error=te, **info)
                w.progress('READOUT_FIT', source_seed=s, target_seed=t, operation=op)
                if time.perf_counter()-w.wall_start > cfg['budget_seconds']:
                    raise TimeoutError('Readout fit budget exceeded')
            np.savez_compressed(w.run/f'readout_s{s}_t{t}.npz', **payload)
        w.checks.update(no_target_outputs_used=True, same_source_fit_pair_identities=True, source_columns_retained=True)
        w.environment = dict(python=sys.executable, python_version=platform.python_version(), numpy=np.__version__,
                             scipy=scipy.__version__, sklearn=sklearn.__version__, torch=torch.__version__,
                             fit_device='cpu',encoder_device=cfg.get('encoder_device','cpu'),matmul_precision='high',threads=2)
    except Exception as exc:
        error = repr(exc); (w.run/'traceback.log').write_text(traceback.format_exc())
    return w.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
