"""Learn bounded native carry components from the designated source fit split."""
import copy
import json
import numpy as np
import torch

from arithmetic_counterfactual_fit import fit_gates
from run_causalgym_multisite import ROOT, write


def fit_source(w, cfg, model, module, tok, ae, stored, budget, current=None):
    parent = ROOT / cfg['source_function_refit']['fit_cache_run']
    assert json.loads(w.checked(parent / 'status.json').read_text())['status'] == 'PASS'
    previous = json.loads(w.checked(parent / 'config.resolved.json').read_text())
    assert all(previous[k] == cfg[k] for k in ['training_run', 'checkpoint_step', 'model_revision', 'max_new_tokens'])
    rows = json.loads(w.checked(parent / 'panel.json').read_text())['rows']
    assert not any('c' in row for row in rows)
    with np.load(w.checked(parent / 'source_seed1.npz')) as data:
        codes = torch.tensor(data['codes'], device=w.device)
    mixed = cfg['source_function_refit'].get('mixed_source')
    raw_mixed = None
    if mixed:
        assert current is not None
        current_rows,current_codes,current_hidden=current
        old_indices=mixed['original_fit_indices']
        new_indices=[i for i,r in enumerate(current_rows) if r['split']=='fit']
        assert len(old_indices)==277 and len(new_indices)==128
        assert all(rows[i]['split']=='fit' for i in old_indices)
        with np.load(w.checked(parent/'states.npz')) as data:
            old_hidden=torch.tensor(data['hidden'][old_indices],device=w.device)
        hidden=torch.cat([old_hidden,current_hidden[new_indices]])
        codes=torch.cat([codes[old_indices],current_codes[new_indices]])
        rows=[rows[i] for i in old_indices]+[current_rows[i] for i in new_indices]
        assert len(rows)==405 and all(r['split']=='fit' for r in rows)
        # The same answer/arity-conditioned linear carry slope supplies a raw control.
        slopes=[];strata=[]
        for arity,total in sorted({(3 if 'c' in r else 2,r['total']) for r in rows}):
            ii=[i for i,r in enumerate(rows) if (3 if 'c' in r else 2)==arity and r['total']==total]
            c=torch.tensor([int(rows[i]['carry']) for i in ii],device=w.device,dtype=torch.float64)
            c=c-c.mean()
            if not bool(c.square().sum()>0):continue
            slopes.append((c[:,None]*hidden[ii,0].double()).sum(0)/c.square().sum())
            strata.append([arity,total,len(ii)])
        raw_mixed=torch.stack(slopes).mean(0).float();raw_mixed=raw_mixed/raw_mixed.norm()
        write(w.run/'MIXED_RAW_REFERENCE.json',dict(strata=strata,definition='Mean within-answer/arity hidden-state carry slopes on the identical405source questions. Unit direction used for donor projection replacement.'))
        write(w.run/'SOURCE_FIT_PANEL.json',dict(rows=rows,original_indices=old_indices,new_indices=new_indices))
    tokenrows = [tok.encode(row['prompt'], add_special_tokens=False) for row in rows]
    if cfg['source_function_refit'].get('read_write'):
        from arithmetic_carry_readwrite import fit_readwrite
        payload=fit_readwrite(w,cfg,model,module,tok,ae,rows,tokenrows,codes,hidden,stored,budget)
        payload['raw_mixed_direction']=raw_mixed.cpu().numpy()
        np.savez_compressed(w.run/'rule_members_seed1.npz',**payload)
        write(w.run/'SOURCE_REFIT.json',dict(fit_rows=405,source_seed=1,
            objective='Separate fixed source carry readout from signed writer; compare continuing diagonal fit and raw read/write.',
            read_write=cfg['source_function_refit']['read_write'],source_fit_only=True))
        return payload
    initial = torch.zeros((codes.shape[-1], 2), device=w.device)
    initial[torch.tensor(stored['field_fitted_binary_64'], device=w.device), 0] = 1
    payload = dict(stored)
    variants = cfg['source_function_refit'].get('arity_variants', [None])
    summaries = []
    for variant in variants:
        local = copy.deepcopy(cfg)
        if variant is not None:
            assert mixed and variant in ['scalar', 'members']
            local['counterfactual_fit']['arity_participation'] = variant
        fitted = fit_gates(w, local, model, module, tok, ae, rows, tokenrows, codes,
            initial, 1, 64, max(map(len, tokenrows)) + cfg['max_new_tokens'], budget,
            tag='carry_function' + (f'_arity_{variant}' if variant else ''), initial_scale=.5,
            selection_batches=cfg['source_function_refit']['selection_batches'])
        assert int((fitted[..., 1] != 0).sum()) == 0
        active = fitted[..., 0] > 0
        indices = torch.where(active.any(0) if variant else active)[0]
        if variant:
            name = cfg['source_function_refit'].get('method_prefix', 'arity') + f'_{variant}_64'
            payload[name] = indices.cpu().numpy()
            payload[name+'_weights'] = fitted[:, indices, 0].cpu().numpy()
            if local['counterfactual_fit'].get('nonnegative_update'):
                payload[name+'_nonnegative_update'] = np.array(True)
                payload[name+'_gain_bound'] = np.array(local['counterfactual_fit']['gain_bound'])
        else:
            prefix='mixed_ce' if mixed else 'function_ce'
            for method in [prefix+'_binary_64', prefix+'_weighted_64']:
                payload[method] = indices.cpu().numpy()
            payload[prefix+'_weighted_64_weights'] = fitted[indices, 0].cpu().numpy()
        summaries.append(dict(arity_participation=variant, members=len(indices),
                              updates=cfg['counterfactual_fit']['steps']))
    if raw_mixed is not None:payload['raw_mixed_direction']=raw_mixed.cpu().numpy()
    np.savez_compressed(w.run / 'rule_members_seed1.npz', **payload)
    write(w.run / 'SOURCE_REFIT.json', dict(
        fit_cache_run=cfg['source_function_refit']['fit_cache_run'],
        fit_rows=sum(r['split'] == 'fit' and not r.get('source_view') for r in rows),
        source_seed=1, variants=summaries, total_backward_batches=sum(v['updates'] for v in summaries),
        objective='Bounded carry gates; full counterfactual answer CE, balanced change/preserve, tens role only. Optional arity participation uses observed operand count, never carry or answers, at execution.',
        mixed_source=bool(mixed),evaluation=cfg['scope']))
    w.checks['source_function_refit_designated_fit_only' if mixed else 'source_function_refit_two_operand_fit_only'] = True
    return payload
