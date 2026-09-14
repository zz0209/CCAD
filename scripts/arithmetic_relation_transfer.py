"""Translate frozen source masks on the states where their operations execute.

Classical bounded least squares fits decoder contribution fields. Clean-trajectory
and source-intervention-trajectory fits use the same source pair identities and
target member allowance. The target model's outputs never enter either fit.
"""
import json
from datetime import datetime, timezone

from fit_component_correspondence import fit


def fit_relations(w, cfg, saes, codes, gates, hidden, parent, generate, budget,
                  rows=None, view_map=None):
    import numpy as np
    import torch
    from scipy.optimize import linear_sum_assignment

    rc = cfg['relation_transfer']
    cap = cfg['members'][0]
    result, metadata = {}, []
    # Fit uses float64 Grams; preserve the model's original physical replay mode.
    for s, t in rc['seed_pairs']:
        sg = gates[s, 'counterfactual_weighted', cap]
        ds, dt = saes[s].decoder.weight.T, saes[t].decoder.weight.T
        maskfile = parent / f'counterfactual_seed{s}_k{cap}.npz'
        if rc.get('source_mask_runs'):
            maskrun = w.run.parent.parent / rc['source_mask_runs'][str(s)]
            assert json.loads(w.checked(maskrun / 'status.json').read_text())['status'] == 'PASS'
            mc = json.loads(w.checked(maskrun / 'config.resolved.json').read_text())
            assert all(mc[key] == cfg[key] for key in ['training_run','checkpoint_step','model_revision','source_cache_run'])
            maskfile = maskrun / rc['source_mask_pattern'].format(seed=s, members=cap)
        with np.load(w.checked(maskfile)) as data:
            all_pairs = data['fit_pairs']
            if rc.get('source_mask_runs'):
                sg = torch.tensor(data['gates'], device=w.device)
                gates[s, 'counterfactual_weighted', cap] = sg
        old_panel = json.loads((parent / 'panel.json').read_text())
        assert all(old_panel['rows'][i]['split'] == 'fit' for pair in all_pairs for i in pair)
        if rc.get('include_equivalent_views'):
            assert rows is not None and view_map is not None
            extra = [(view_map[i], view_map[j]) for i, j in all_pairs
                     if view_map[i] != i and view_map[j] != j]
            all_pairs = np.concatenate([all_pairs, np.array(extra)], axis=0)
            assert all(rows[i]['split'] == 'fit' for pair in all_pairs for i in pair)
        order = np.random.default_rng(rc['pair_seed']).permutation(len(all_pairs))[:rc['fit_pairs']]
        pp = all_pairs[order].tolist()
        ii, jj = [i for i, j in pp], [j for i, j in pp]
        clean, path, assignment = [torch.zeros_like(sg) for _ in range(3)]
        details = []
        for c, operation in enumerate(['unit', 'tens']):
            parts = []
            contexts = rc.get('contexts', ['clean', 'path'])
            for off in (range(0, len(pp), cfg['batch_size']) if 'path' in contexts else []):
                chunk = pp[off:off + cfg['batch_size']]
                recipient, donor = [i for i, j in chunk], [j for i, j in chunk]

                def patch(current, step):
                    zs = saes[s].encode(current.reshape(-1, current.shape[-1])).reshape(*current.shape[:-1], -1)
                    return ((codes[s][donor, :step + 1] - zs) * sg[:, c]) @ ds

                _, _, hpath, _ = generate(recipient, patch=patch)
                assert float((hpath[:, 0] - hidden[recipient, 0]).abs().max()) < cfg['hidden_atol']
                parts.append(hpath)
                budget()
            state_sets = [('clean', hidden[ii], clean)] if 'clean' in contexts else []
            if 'path' in contexts:
                state_sets.append(('path', torch.cat(parts), path))
            for name, states, output in state_sets:
                with torch.no_grad():
                    flat = states.reshape(-1, states.shape[-1])
                    zs = saes[s].encode(flat).reshape(*states.shape[:-1], -1)
                    zt = saes[t].encode(flat).reshape(*states.shape[:-1], -1)
                    y = (((codes[s][jj] - zs) * sg[:, c]) @ ds).reshape(-1, ds.shape[-1]).double()
                    x = (codes[t][jj] - zt).reshape(-1, zt.shape[-1]).double()
                    d = dt.double()
                    energy = x.square().mean(0) * d.square().sum(1)
                    b = (x * (y @ d.T)).mean(0)
                    score = b.clamp_min(0).square() / energy.clamp_min(1e-12)
                    pool = torch.argsort(score, descending=True, stable=True)[:rc['pool']]
                    xp, dp = x[:, pool], d[pool]
                    gram = (xp.T @ xp / len(x)) * (dp @ dp.T)
                    coef, _ = fit(gram, b[pool, None], steps=rc['steps'],
                                  ridge_fraction=rc['ridge_fraction'], capacity=False)
                    allowed = torch.zeros_like(coef, dtype=torch.bool)
                    keep = torch.argsort(coef[:, 0].square() * energy[pool], descending=True, stable=True)[:cap]
                    allowed[keep, 0] = True
                    coef, info = fit(gram, b[pool, None], steps=rc['steps'],
                                     ridge_fraction=rc['ridge_fraction'], allowed=allowed, capacity=False)
                    output[pool, c] = coef[:, 0].float()
                    pred = (xp * coef[:, 0]) @ dp
                    mse = float((pred - y).square().sum() / y.square().sum().clamp_min(1e-12))
                    details.append(dict(operation=operation, context=name, relative_field_mse=mse,
                                        fit_examples=len(x), active=int((output[:, c] > 0).sum()), solver=info))
                budget()
            selected = torch.where(sg[:, c] > 0)[0]
            with torch.no_grad():
                source_d = ds[selected] / ds[selected].norm(dim=1, keepdim=True).clamp_min(1e-12)
                target_d = dt / dt.norm(dim=1, keepdim=True).clamp_min(1e-12)
                cosine = (source_d @ target_d.T).cpu().numpy()
            row, col = linear_sum_assignment(-cosine)
            assignment[torch.tensor(col, device=w.device), c] = sg[selected[torch.tensor(row, device=w.device)], c]

        payload = dict(assignment=assignment.cpu().numpy(), source_gate=sg.cpu().numpy(),
                       fit_pairs=np.array(pp), selected_pair_indices=order)
        for name, g in [('clean', clean), ('path', path)]:
            if name in contexts:
                payload[name] = g.cpu().numpy()
        np.savez_compressed(w.run / f'relation_s{s}_t{t}.npz', **payload)
        for name, g in [('clean', clean), ('path', path), ('assignment', assignment)]:
            if name != 'assignment' and name not in contexts:
                continue
            assert bool(((g >= 0) & (g <= 1)).all()) and bool(((g > 0).sum(0) <= cap).all())
            result[t, f'transfer_s{s}_{name}', cap] = g
        item = dict(source_seed=s, target_seed=t, fit_pairs=len(pp), fit_contexts=details,
                    source_operation='counterfactual_weighted', target_output_labels_used=0,
                    target_output_gradients_used=0, candidate_allowance=cap,
                    contexts=contexts, source_mask=str(maskfile),
                    equivalent_pairs=sum(bool(rows[i].get('source_view')) for i, j in pp) if rows is not None else 0,
                    assignment='Optimal signed-cosine assignment of the selected source members to distinct target members; not full-dictionary PW-MCC.')
        metadata.append(item)
        w.progress('ARITHMETIC_RELATION_FIT', source_seed=s, target_seed=t, details=details)
    (w.run / 'RELATION_FREEZE.json').write_text(json.dumps(dict(
        written_at_utc=datetime.now(timezone.utc).isoformat(), source_parent=str(parent), fits=metadata,
        scope='Source masks were trained on old fit labels. Relation fits reuse those source pairs and states, with no target output supervision. All evaluation is on previously exposed development pairs. Shared seed cycles are dependent.'), indent=2) + '\n')
    w.checks['relation_budget_and_bounds'] = True
    return result, metadata
