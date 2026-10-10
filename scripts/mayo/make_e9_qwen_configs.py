import json
from pathlib import Path

CONFIGS = Path(__file__).resolve().parent / 'configs'
MAPS = ('/mnt/nvme3-data/harvardhds/ccad-m364569/runs/naacl_sprint_20261010/qwen17_matching_maps_l14_v2_s{source_seed}/'
        'hungarian_decoder_s{source_seed}_t{target_seed}.npz')


def main():
    for direction in ['s1_t2', 's2_t1']:
        cfg = json.loads((CONFIGS / f'e4b_qwen17_{direction}_v3.json').read_text())
        cfg.pop('matching_maps')
        cfg.update(
            run_id=cfg['run_id'].replace('E4B_QWEN17', 'E9_QWEN17_FIXED').replace('_v4_', '_v1_'),
            run_id_template=cfg['run_id_template'].replace('E4B_QWEN17', 'E9_QWEN17_FIXED').replace('_v4_', '_v1_'),
            variants=['membership_whole', 'membership'], membership_candidates=16, membership_init=MAPS,
            membership_lr=0.02,
            dev_choice='membership_lr=0.02 chosen on the GPT2Medium development split, NAACL_E9_GPT2_FIXED_DEV_*',
            purpose=('E9 fixed correspondence for the Qwen3-1.7B grammatical explanation between Qwen-Scope dictionaries, '
                     'run with the target SAE alone and the target dictionary frozen.'),
            budget='Two 512-update membership arms plus evaluation on 384 further pairs, at most 7200 driver seconds.')
        (CONFIGS / f'e9_qwen17_fixed_{direction}.json').write_text(json.dumps(cfg, indent=2) + '\n')


if __name__ == '__main__':
    main()
