from pathlib import Path
import json


ROOT=Path(__file__).resolve().parents[1]
BULK=Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round02')


def save(path,value):
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    base=json.loads((ROOT/'configs/rg02_finite_embedding_development_t2.json').read_text())
    for stage in ['smoke','development']:
        c=base.copy()
        c.update(run_id=f'RG02_FINITE_SUPPORT_{stage.upper()}_T2_20260921',arms=['sparse'],
             baseline_methods=['program'],sparse_lr=.003,
             reference_program_directory='D:/CCAD_Storage/training_curves/SCIENCE04_shift_t2_v1_20260919/tangent_mixed',
             purpose='Let finite response optimization select the same20target members and their common contribution columns.',
             scope='Matched source context and query stream; source zero columns, code capacity, frozen SAE and20member allowance preserved.',
             checkpoints=[64,256])
        if stage=='smoke':
            c.update(steps=4,checkpoints=[4],fit_documents=16,calibration_documents=4,
                     evaluation_per_cell=1,evaluation_queries=['full','pronouns','names'],log_every=1,budget_seconds=450)
        save(ROOT/f'configs/rg02_finite_support_{stage}_t2.json',c)
    c=base.copy()
    fitted=BULK/'RG02_FINITE_EMBEDDING_DEVELOPMENT_T2_20260921'
    c.update(run_id='RG02_FINITE_FIELD_TRANSFER_T3_20260921',target_seed=3,seeds=[3],arms=[],steps=0,checkpoints=[],
             baseline_methods=['tangent','geometric','program','readout'],
             reference_program_directory='D:/CCAD_Storage/training_curves/SCIENCE04_shift_t3_v1_20260919/tangent_mixed',
             transfer_fields={name:str(fitted/f'{arm}_field.pt') for name,arm in [('transfer_gain','gain'),('transfer_finite','finite')]},
             purpose='Test whether finite fields learned in target2 improve another dictionary without target3response fitting.',
             scope='Same fixed source explanation and exposed development contexts; target3has no fitting in this unit.')
    save(ROOT/'configs/rg02_finite_field_transfer_t3.json',c)


if __name__=='__main__':
    main()
