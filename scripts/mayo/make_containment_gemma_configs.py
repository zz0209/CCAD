import json
from pathlib import Path

configs = Path(__file__).resolve().parent / 'configs'
M = '/mnt/nvme3-data/harvardhds/ccad-m364569'
# CONTAINMENT experiment 1 on the two formal Gemma Scope targets: the task-free natural-text family (many-to-many),
# its one-to-one control with gains up to four, part supervision, and part-supervised one-to-one gains up to four,
# with the E4c candidates, initialization, learning rate and three training seeds. Every map is also scored on
# held-out generic text.
for key in ['w65k', 'l0low16k']:
    cfg = json.loads((configs / f'e4c_gemma_final_{key}.json').read_text())
    cfg.update(run_id=f'CSM_C1_GEMMA2B_natural_{key}_v1_20261011', run_root=f'{M}/runs/cross_seed_models',
               fit_families=['natural', 'natural_matched', 'parts', 'parts_matched'], skip_use=True,
               full_model=True, matched_gain_max=4., natural_lm_texts=f'{M}/data/cross_seed/natural_fineweb_slice.json',
               natural_lm_len=128, natural_lm_batch=4, natural_steps=512, natural_eval_seed=1,
               purpose=('CONTAINMENT experiment 1: a correspondence trained only on generic natural text with random feature '
                        'subsets, against part supervision and one-to-one controls with gains up to four, for the published '
                        'Gemma SHIFT explanation.'))
    (configs / f'c1_gemma_natural_{key}.json').write_text(json.dumps(cfg, indent=1))
    smoke = dict(cfg, run_id=cfg['run_id']+'_SMOKE', natural_bios=64, fit_bios=32, fit_steps=2, natural_steps=2,
                 eval_bios=16, limit_use=64, fit_seeds=[0])
    (configs / f'c1_gemma_natural_{key}_smoke.json').write_text(json.dumps(smoke, indent=1))
    print(cfg['run_id'])
