"""Learn one native carry component from the original two-operand fit split."""
import json
import numpy as np
import torch

from arithmetic_counterfactual_fit import fit_gates
from run_causalgym_multisite import ROOT, write


def fit_source(w, cfg, model, module, tok, ae, stored, budget):
    parent = ROOT / cfg['source_function_refit']['fit_cache_run']
    assert json.loads(w.checked(parent / 'status.json').read_text())['status'] == 'PASS'
    previous = json.loads(w.checked(parent / 'config.resolved.json').read_text())
    assert all(previous[k] == cfg[k] for k in ['training_run', 'checkpoint_step', 'model_revision', 'max_new_tokens'])
    rows = json.loads(w.checked(parent / 'panel.json').read_text())['rows']
    assert not any('c' in row for row in rows)
    tokenrows = [tok.encode(row['prompt'], add_special_tokens=False) for row in rows]
    with np.load(w.checked(parent / 'source_seed1.npz')) as data:
        codes = torch.tensor(data['codes'], device=w.device)
    initial = torch.zeros((codes.shape[-1], 2), device=w.device)
    initial[torch.tensor(stored['field_fitted_binary_64'], device=w.device), 0] = 1
    fitted = fit_gates(w, cfg, model, module, tok, ae, rows, tokenrows, codes,
        initial, 1, 64, max(map(len, tokenrows)) + cfg['max_new_tokens'], budget,
        tag='carry_function', initial_scale=.5,
        selection_batches=cfg['source_function_refit']['selection_batches'])
    assert int((fitted[:, 1] != 0).sum()) == 0
    indices = torch.where(fitted[:, 0] > 0)[0]
    payload = dict(stored)
    for method in ['function_ce_binary_64', 'function_ce_weighted_64']:
        payload[method] = indices.cpu().numpy()
    payload['function_ce_weighted_64_weights'] = fitted[indices, 0].cpu().numpy()
    np.savez_compressed(w.run / 'rule_members_seed1.npz', **payload)
    write(w.run / 'SOURCE_REFIT.json', dict(
        fit_cache_run=cfg['source_function_refit']['fit_cache_run'],
        fit_rows=sum(r['split'] == 'fit' and not r.get('source_view') for r in rows),
        source_seed=1, members=len(indices), updates=cfg['counterfactual_fit']['steps'],
        objective='One shared carry gate; full counterfactual answer CE, balanced change/preserve, tens role only.',
        evaluation='Exposed R50 triple panel: development after frozen-rule outcomes; no triple fitting or checkpoint selection.'))
    w.checks['source_function_refit_two_operand_fit_only'] = True
    return payload
