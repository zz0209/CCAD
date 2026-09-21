from pathlib import Path
import argparse
import json


ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--regrow',action='store_true')
    args=parser.parse_args()
    c=json.loads((ROOT/'configs/rg02_finite_support_development_t2.json').read_text())
    if args.regrow:
        for stage in ['smoke','development']:
            v=c.copy()
            v.update(run_id=f'RG02_REGROW_{stage.upper()}_T2_20260921',arms=['regrow'],baseline_methods=[],
                purpose='Test finite-gradient membership replacement at the same20member budget and256updates.')
            if stage=='smoke':
                v.update(steps=32,checkpoints=[32],fit_documents=16,calibration_documents=4,
                         evaluation_per_cell=1,evaluation_queries=['full','pronouns','names'],log_every=16,budget_seconds=450)
            path=ROOT/f'configs/rg02_regrow_{stage}_t2.json'
            if path.exists():raise FileExistsError(path)
            path.write_text(json.dumps(v,indent=2)+'\n')
        return
    c.update(run_id='RG02_FINITE_STATE_DEVELOPMENT_T2_20260921',arms=['fixed_direct','sparse'],
        source_response_weight=0.,baseline_methods=[],
        purpose='Test whether preserving the full finite representation response improves later readouts compared with original-head weighting.',
        scope='Same source teacher contexts, requests, target dictionary, coefficient coordinates and256updates. Later classifiers remain excluded.')
    path=ROOT/'configs/rg02_finite_state_development_t2.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(c,indent=2)+'\n')


if __name__=='__main__':
    main()
