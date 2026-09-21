from pathlib import Path
import json


ROOT=Path(__file__).resolve().parents[1]
BULK=Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round02')


def main():
    c=json.loads((ROOT/'configs/rg02_finite_field_transfer_t3.json').read_text())
    c.update(run_id='RG02_SPARSE_FIELD_TRANSFER_T3_20260921',baseline_methods=[],
        transfer_fields={'transfer_sparse':str(BULK/'RG02_FINITE_SUPPORT_DEVELOPMENT_T2_20260921/sparse_field.pt')},
        purpose='Transfer the response-fitted sparse field to target3 with no target3 response fitting.')
    path=ROOT/'configs/rg02_sparse_field_transfer_t3.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(c,indent=2)+'\n')


if __name__=='__main__':
    main()
