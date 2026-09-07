"""Apply a saved FCC operation to full or selected target-code differences.

Examples use the selected_difference array in the packaged example NPZ. This
command writes residual updates only; it does not run or modify a language model.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from ccad.predictive_operation import PredictiveOperation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selected',action='store_true',help='Input columns follow saved target_members')
    parser.add_argument('--components',action='store_true',help='Also save each member vector; use small batches')
    args=parser.parse_args()
    operation=PredictiveOperation.load(args.operation)
    with np.load(args.input,allow_pickle=False) as data:
        delta=data['selected_difference' if args.selected else 'code_difference']
        if args.selected and 'target_members' in data and not np.array_equal(data['target_members'],operation.target_members):
            raise ValueError('Selected input member IDs or order differ from the operation')
        selected=delta if args.selected else operation._array(delta,operation.latent_width)[...,operation.target_members]
        prediction=operation.predict_selected(selected)
        result=dict(delta=prediction,target_members=operation.target_members,metadata_json=np.array(json.dumps(operation.metadata)))
        if 'row_ids' in data: result['row_ids']=data['row_ids']
        if args.components: result['component_deltas']=operation.component_deltas(selected)
    with args.output.open('xb') as handle: np.savez_compressed(handle,**result)
    print(json.dumps(dict(output=str(args.output.resolve()),shape=list(prediction.shape),factor=operation.metadata.get('factor'),
                         source_seed=operation.metadata.get('source_seed'),target_seed=operation.metadata.get('target_seed'),
                         output_kind='source-aligned residual update; not native feature ablation')))


if __name__=='__main__': main()
