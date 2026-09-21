from pathlib import Path
import json


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/reuse_generalization_20260921_round02'
BULK=Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round02')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    BULK.mkdir(parents=True,exist_ok=True)
    base=json.loads((ROOT/'configs/fs05_human_semantic_dev_v2.json').read_text())
    base.update(run_parent='REUSE_GENERALIZATION_02',run_storage_root=str(BULK),
        generator_script='scripts/fit_finite_embedding_transport.py',
        purpose='Compare finite functional training with geometric coefficients on the same frozen target members.',
        scope='Complete55-member source program; embedding coefficients vary by token identity; all other execution uses the same frozen tangent rule.',
        source_context_membership='D:/CCAD_Storage/runs/science_upgrade_20260919/SCIENCE04_shift_t2_v1_20260919/program_context_membership.json',
        budget='Two256-update arms, fixed20embeddingmembers and unchanged natural dictionaries; paired contexts and requests.',
        budget_seconds=1800,baseline_methods=['tangent','geometric','source_embedding','readout'],arms=['gain','finite'],
        fit_documents=256,calibration_documents=16,batch_sequences=2,eval_batch_size=2,steps=256,
        gain_lr=.01,coefficient_lr=.003,inverse_steps=128,source_response_weight=.9,log_every=16,
        checkpoints=[64,256],training_seed=92102,evaluation_per_cell=4)
    for stage in ['smoke','development']:
        c=base.copy()
        c['run_id']=f'RG02_FINITE_EMBEDDING_{stage.upper()}_T2_20260921'
        if stage=='smoke':
            c.update(steps=4,checkpoints=[4],fit_documents=16,calibration_documents=4,
                     evaluation_per_cell=1,evaluation_queries=['full','pronouns','names'],log_every=1,budget_seconds=450)
        path=ROOT/f'configs/rg02_finite_embedding_{stage}_t2.json'
        if path.exists():raise FileExistsError(path)
        path.write_text(json.dumps(c,indent=2)+'\n')
        print(path)


if __name__=='__main__':
    main()
