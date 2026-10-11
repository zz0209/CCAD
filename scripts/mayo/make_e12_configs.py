import json
from pathlib import Path

configs = Path(__file__).resolve().parent / 'configs'
# E12 adds the one-to-one baseline with trained gains (Hungarian matches whose deletion fractions are trained with
# part supervision) to the five E10 grid targets, with the E10 recipe, test split and training seeds unchanged.
for key in ['w16k_lower', 'w16k_higher', 'w65k_lowest', 'w16k_highest', 'w65k_highest']:
    cfg = json.loads((configs / f'e10_gemma_grid_{key}.json').read_text())
    cfg.update(run_id=f'NAACL_E12_GEMMA2B_matched_{key}_v1_20261010', fit_families=['parts_matched'],
               purpose=('One-to-one correspondence with gains trained by part supervision on the Hungarian matches, '
                        'for the E10 grid targets, to compare with the learned many-to-many correspondence.'))
    (configs / f'e12_gemma_matched_{key}.json').write_text(json.dumps(cfg, indent=1))
    print(cfg['run_id'])
    # Version 2 lets each trained gain reach four, so the one-to-one control can rescale a weak match.
    cfg.update(run_id=f'NAACL_E12_GEMMA2B_matched_{key}_v2_20261011', matched_gain_max=4.,
               purpose=('One-to-one correspondence with gains up to four trained by part supervision on the Hungarian '
                        'matches, for the E10 grid targets, to compare with the learned many-to-many correspondence.'))
    (configs / f'e12_gemma_matched_{key}_v2.json').write_text(json.dumps(cfg, indent=1))
    print(cfg['run_id'])
