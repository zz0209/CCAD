"""Apply a complete-contribution map to one input, or a reciprocal contrast.

This NumPy-only consumer returns source-aligned residual vectors. It does not
run the SAE/model or perform native target-feature deletion.
"""
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from ccad.predictive_operation import PredictiveOperation


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--operation',type=Path,required=True);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--consumer',choices=['contribution','removal','contrast'],default='removal');ap.add_argument('--selected',action='store_true');ap.add_argument('--components',action='store_true');args=ap.parse_args()
    op=PredictiveOperation.load(args.operation)
    if op.metadata.get('input_kind')!='absolute_target_code':raise ValueError('A complete-contribution operation is required')
    with np.load(args.input,allow_pickle=False) as data:
        key='selected_codes' if args.selected else 'codes'
        z=data[key]
        if args.selected and 'target_members' in data and not np.array_equal(data['target_members'],op.target_members):raise ValueError('Selected member identity/order mismatch')
        if args.consumer=='contrast':
            donor=data['selected_donor_codes' if args.selected else 'donor_codes']
            if donor.shape!=z.shape:raise ValueError('Recipient and donor code shapes differ')
            z=donor-z
        elif args.consumer=='removal':z=-z
        selected=op._array(z,len(op.target_members)) if args.selected else op._array(z,op.latent_width)[...,op.target_members]
        result=dict(delta=op.predict_selected(selected),target_members=op.target_members,metadata_json=np.array(json.dumps(op.metadata)),consumer=np.array(args.consumer))
        if 'row_ids' in data:result['row_ids']=data['row_ids']
        if args.components:result['component_deltas']=op.component_deltas(selected)
    with args.output.open('xb') as f:np.savez_compressed(f,**result)
    print(json.dumps(dict(output=str(args.output.resolve()),shape=list(result['delta'].shape),consumer=args.consumer,donor_required=args.consumer=='contrast',output_kind='source-aligned residual vector, not native target ablation')))


if __name__=='__main__':main()
