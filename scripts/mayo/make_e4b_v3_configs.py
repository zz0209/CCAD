import json
from pathlib import Path

CONFIGS = Path(__file__).resolve().parent / 'configs'
FRESH = '/mnt/nvme3-data/harvardhds/ccad-m364569/runs/naacl_sprint_20261010/qwen17_fresh_panel_rg09/confirmation_panel.json'


def main():
    for direction in ['s1_t2', 's2_t1']:
        cfg = json.loads((CONFIGS / f'e4b_qwen17_{direction}.json').read_text())
        cfg['evaluation_panel'] = FRESH
        cfg['evaluation_split'] = 'rg09_new_sentence_confirmation'
        # v3 failed before evaluation because it kept the RG08 split label; v4 is the corrected confirmation.
        cfg['run_id'] = cfg['run_id'].replace('_v2_', '_v4_')
        cfg['run_id_template'] = cfg['run_id_template'].replace('_v2_', '_v4_')
        cfg['scope'] = ('Source groups use only source gradients on source_selection pairs. Fitting uses the original 64 fit '
                        'pairs per grammar. Confirmation uses 384 further BLiMP pairs that no earlier Qwen3 run evaluated, '
                        'after the dictionary learning rate was chosen on development runs. Matched update budget across arms.')
        cfg['budget'] = 'Three 512-update arms plus evaluation on 384 further pairs, at most 7200 driver seconds on one RTX 6000 Ada.'
        (CONFIGS / f'e4b_qwen17_{direction}_v3.json').write_text(json.dumps(cfg, indent=2) + '\n')


if __name__ == '__main__':
    main()
