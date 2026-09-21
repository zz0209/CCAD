from pathlib import Path
import json


ROOT=Path(__file__).resolve().parents[1]


def main():
    base=json.loads((ROOT/'configs/rg02_finite_support_development_t2.json').read_text())
    for stage in ['smoke','development']:
        c=base.copy()
        c.update(run_id=f'RG02_FIXED_DIRECT_{stage.upper()}_T2_20260921',arms=['fixed_direct'],baseline_methods=[],
                 purpose='Compare fixed and selectable supports with identical coefficient coordinates, optimizer and source information.')
        if stage=='smoke':
            c.update(steps=4,checkpoints=[4],fit_documents=16,calibration_documents=4,
                     evaluation_per_cell=1,evaluation_queries=['full','pronouns','names'],log_every=1,budget_seconds=450)
        path=ROOT/f'configs/rg02_fixed_direct_{stage}_t2.json'
        if path.exists():raise FileExistsError(path)
        path.write_text(json.dumps(c,indent=2)+'\n')


if __name__=='__main__':
    main()
