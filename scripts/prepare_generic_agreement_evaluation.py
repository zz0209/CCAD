from pathlib import Path
import json


ROOT=Path(__file__).resolve().parents[1]
BULK=Path('D:/CCAD_Storage/runs/final_science_20260921_round05')


def main():
    template=json.loads((ROOT/'configs/fs05_cross_explanation_semantic_dev.json').read_text())
    for objective in ['local','downstream','distribution']:
        origin=BULK/f'FS05_GENERIC_{objective.upper()}_DEV_20260921'
        c=dict(template)
        c.update(run_id=f'FS05_GENERIC_{objective.upper()}_AGREEMENT_DEV_20260921',
                 purpose='Evaluate a natural-action-trained target on the excluded subject-number explanation without new fitting.',
                 scope='The checkpoint uses ordinary-text actions excluding both evaluation explanations. All 48 agreement development contexts and 21 requests retained.',
                 task_adapted_reference={'grammar':str(origin/'generic_program_512/resid_4_seed3.pt')},
                 transfer_training_run=str(origin),request_design=objective)
        path=ROOT/f'configs/fs05_generic_{objective}_agreement_dev.json'
        assert not path.exists()
        path.write_text(json.dumps(c,indent=2)+'\n')
    print('Three existing-driver evaluation configurations prepared')


if __name__=='__main__':
    main()
