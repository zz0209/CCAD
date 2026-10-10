import argparse
import copy
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'configs'
B = '/mnt/nvme3-data/harvardhds/ccad-m364569'
BASE = dict(
    run_root=f'{B}/runs/naacl_sprint_20261010',
    model_dir=f'{B}/assets/gemma-2-2b',
    model_revision='c5ebcd40d208330abc697524c919956e692655cf',
    dataset_dir=f'{B}/assets/bias_in_bios/052f01de644dba841176e0449528b41f27d94a61',
    features=f'{B}/assets/GEMMA_SHIFT43.json',
    l0_rule=f'{B}/assets/GEMMA_SCOPE_L0_RULE.json',
    res_root=f'{B}/assets/gemma-scope-2b-pt-res',
    att_root=f'{B}/assets/gemma-scope-2b-pt-att',
    source_width='16k', expected_features=43, probe_layer=22,
    seed=0, data_seed=42, probe_seed=42, probe_lr=0.01, probe_batch=32,
    natural_bios=3000, natural_batch=16, candidates_cos=24, candidates_corr=8,
    fit_bios=1024, fit_sequences=24, fit_steps=400, fit_lr=0.02, fit_l1=1e-5, fit_families=['complete', 'parts'],
    subset_inclusion='half',
    eval_batch=64, use_batch=64,
    layer_bands=dict(early=[0, 9], middle=[10, 15], late=[16, 22]), member_subsets=8, member_seed=0,
    active_weight=0.05,
    statistics_unit='Balanced biographies stratified by profession and gender',
    information_budget=('All target-only executors use the same target dictionaries. Label-free maps use 3000 natural '
                        'biographies of other professions. Response-trained maps also see source deletion responses of the '
                        'original classifier on 1024 ambiguous training biographies, with equal steps and sequences for '
                        'complete and part supervision.'))
TARGETS = dict(w65k=dict(target_width='65k', target_choice='sfc_rule',
                         target_label='Gemma Scope 65k at the SFC L0 rule, a wider dictionary of similar sparsity'),
               l0low16k=dict(target_width='16k', target_choice='lowest_l0',
                             target_label='Gemma Scope 16k at the lowest released L0, a sparser dictionary of equal width'))


def write(name, **kw):
    cfg = copy.deepcopy(BASE)
    cfg.update(kw)
    (OUT / f'{name}.json').write_text(json.dumps(cfg, indent=1) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final-init', choices=['native', 'hungarian'])
    parser.add_argument('--final-families', nargs='+', default=['complete', 'parts', 'subsets', 'parts_matched'])
    args = parser.parse_args()
    # smoke v3 ran out of GPU memory beside a development run; v4 repeats it on the other GPU.
    write('e4c_gemma_smoke', run_id='NAACL_E4C_GEMMA2B_smoke_v4_20261010', purpose='Smoke test on small subsets.',
          **TARGETS['l0low16k'], eval_split='dev', skip_use=False, fit_inits=['hungarian'], fit_seeds=[0],
          fit_families=['complete', 'parts', 'subsets', 'parts_matched'], subset_inclusion='uniform',
          fit_steps=3, natural_bios=64, fit_bios=64, eval_bios=64, limit_use=256, member_subsets=2)
    for key, target in TARGETS.items():
        write(f'e4c_gemma_dev_{key}', run_id=f'NAACL_E4C_GEMMA2B_dev_{key}_v1_20261010',
              purpose='Development choice of initialization and training length on the dev split; no use stage.',
              **target, eval_split='dev', skip_use=True, fit_inits=['native', 'hungarian'], fit_seeds=[0], eval_bios=600)
        write(f'e4c_gemma_dev2_{key}', run_id=f'NAACL_E4C_GEMMA2B_dev2_{key}_v1_20261010',
              purpose=('Development test of a third supervision family that adds random feature halves to the parts and '
                       'full deletion, after dev v1 showed weak transfer to untrained feature subsets; no use stage.'),
              **target, eval_split='dev', skip_use=True, fit_inits=['hungarian'], fit_seeds=[0],
              fit_families=['complete', 'parts', 'subsets'], eval_bios=600)
        write(f'e4c_gemma_dev3_{key}', run_id=f'NAACL_E4C_GEMMA2B_dev3_{key}_v1_20261010',
              purpose=('Development test of random subsets with a per-row inclusion probability drawn uniformly, after '
                       'dev2 showed that half-size subsets leave small requests unconstrained under saturating responses.'),
              **target, eval_split='dev', skip_use=True, fit_inits=['hungarian'], fit_seeds=[0], fit_families=['subsets'],
              subset_inclusion='uniform', eval_bios=600)
        learned = [f'{f}_hungarian_s{s}' for f in ['complete_sup', 'part_sup'] for s in (0, 1, 2)]
        write(f'e4c_gemma_partuse_{key}', run_id=f'NAACL_E4C_GEMMA2B_partuse_{key}_v1_20261010',
              purpose=('Part-level SHIFT: delete each annotated part alone through each method, retrain the classifier and '
                       'compare the decisions each part supports, using the saved correspondences of the final run.'),
              **target, eval_split='test', skip_use=False, fit_inits=['hungarian'], fit_seeds=[0, 1, 2],
              source_run=f'{B}/runs/naacl_sprint_20261010/NAACL_E4C_GEMMA2B_final_{key}_v2_20261010',
              maps=['hungarian_decoder', 'hungarian_activation', 'native'] + learned,
              methods=['source', 'readout', 'hungarian_decoder', 'hungarian_activation', 'native'] + learned,
              parts=['pronouns', 'female_descriptions'])
        if args.final_init:
            # final v1 of the low-L0 target was stopped before any result to change the subset family, hence v2.
            write(f'e4c_gemma_final_{key}', run_id=f'NAACL_E4C_GEMMA2B_final_{key}_v2_20261010',
                  purpose=('Confirmation on the balanced test split with the initialization chosen on the dev split, '
                           'three training seeds and the SHIFT use stage.'),
                  **target, eval_split='test', skip_use=False, fit_inits=[args.final_init], fit_seeds=[0, 1, 2],
                  fit_families=args.final_families, subset_inclusion='uniform', eval_bios=100000,
                  dev_choice=(f'fit_init={args.final_init} from NAACL_E4C_GEMMA2B_dev_*_v1_20261010; families '
                              f'{args.final_families} with uniform subset inclusion after NAACL_E4C_GEMMA2B_dev2_*_v1 and '
                              'dev3_*_v1'))


if __name__ == '__main__':
    main()
