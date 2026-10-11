import json
from pathlib import Path

configs = Path(__file__).resolve().parent / 'configs'
M = '/mnt/nvme3-data/harvardhds/ccad-m364569'
SEEDS = [0, 1, 2]
# Word lists for the word-class readers, fixed before any reader result.
FEMALE = ['woman', 'women', 'female', 'girl', 'girls', 'lady', 'ladies', 'mother', 'mothers', 'mom', 'wife', 'daughter',
          'sister', 'aunt', 'niece', 'grandmother', 'bride', 'girlfriend', 'actress', 'waitress', 'hostess', 'mrs', 'ms',
          'miss', 'madam', 'feminine', 'maternal', 'sorority', 'queen', 'princess']
MALE = ['man', 'men', 'male', 'boy', 'boys', 'gentleman', 'gentlemen', 'father', 'fathers', 'dad', 'husband', 'son',
        'brother', 'uncle', 'nephew', 'grandfather', 'groom', 'boyfriend', 'actor', 'waiter', 'host', 'mr', 'sir',
        'mister', 'masculine', 'paternal', 'fraternity', 'king', 'prince', 'guy']
# READER_COVERAGE EXP-1 on the two formal Gemma Scope targets: every saved map of CONTAINMENT experiment 1 and the
# label-free E8a families, read on development biographies by the original classifier, the E8b later classifiers, a
# gender probe, the next-token distribution, word-class mass and layer-22 states, and on held-out generic text.
def main():
    for key in ['w65k', 'l0low16k']:
        cfg = json.loads((configs / f'c1_gemma_natural_{key}.json').read_text())
        e8b = json.loads((configs / f'e8b_gemma_taskuse_{key}.json').read_text())
        c1 = f'{M}/runs/cross_seed_models/{cfg["run_id"]}'
        e8a = f'{M}/runs/naacl_sprint_20261010/NAACL_E8A_GEMMA2B_labelfree_{key}_v1_20261010'
        cfg.update(
            run_id=f'CSM_EXP1_GEMMA2B_readers_{key}_v1_20261011', run_root=f'{M}/runs/cross_seed_models', probe_run=c1,
            map_runs=[dict(run=c1, prefix='', methods=['hungarian_decoder', 'hungarian_activation', 'top2_decoder', 'native'] +
                           [f'{f}_hungarian_s{s}' for s in SEEDS for f in ['natural_sup', 'natural_matched', 'part_sup', 'matched_part_sup']]),
                      dict(run=e8a, prefix='e8a_', methods=[f'{f}_hungarian_s{s}' for s in SEEDS for f in ['complete_sup', 'random_sup', 'singleton_sup']])],
            eval_split='dev', eval_bios=256, batch=4, later_pairs=e8b['later_pairs'], later_epochs=e8b['later_epochs'],
            random_requests=16, random_seed=20261011, female_words=FEMALE, male_words=MALE, natural_docs=128,
            purpose=('READER_COVERAGE EXP-1: held-out readers for every saved Gemma SHIFT correspondence on development '
                     'biographies and generic text.'))
        for k in ['fit_families', 'fit_seeds', 'fit_inits', 'skip_use']:
            cfg.pop(k)
        (configs / f'exp1_gemma_readers_{key}.json').write_text(json.dumps(cfg, indent=1))
        smoke = dict(cfg, run_id=cfg['run_id'] + '_SMOKE', eval_bios=8, natural_docs=4, random_requests=2,
                     later_pairs=cfg['later_pairs'][:1], map_runs=[dict(cfg['map_runs'][0], methods=['hungarian_decoder', 'natural_matched_hungarian_s0'])])
        (configs / f'exp1_gemma_readers_{key}_smoke.json').write_text(json.dumps(smoke, indent=1))
        print(cfg['run_id'])


if __name__ == '__main__':
    main()
