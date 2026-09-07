"""Apply a compact paired completion using two saved PredictiveOperation maps.

Both selected recipient and reciprocal-donor codes are required. This command
fits nothing and returns physical residual updates without running a model.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from ccad.predictive_operation import PredictiveOperation


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--odd-operation',type=Path,required=True)
    ap.add_argument('--common-operation',type=Path,required=True)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    odd=PredictiveOperation.load(args.odd_operation);common=PredictiveOperation.load(args.common_operation)
    if odd.metadata.get('paired_part')!='odd' or common.metadata.get('paired_part')!='common':
        raise ValueError('Odd and common map roles must match their arguments')
    if not np.array_equal(odd.target_members,common.target_members):raise ValueError('The two compact maps must use the same target members and order')
    if odd.metadata.get('paired_completion_id')!=common.metadata.get('paired_completion_id') or not odd.metadata.get('paired_completion_id'):
        raise ValueError('The two maps must belong to the same paired completion')
    with np.load(args.input,allow_pickle=False) as inp:
        if not np.array_equal(inp['target_members'],odd.target_members):raise ValueError('Input member IDs/order differ from the maps')
        recipient=PredictiveOperation._array(inp['selected_recipient'],len(odd.target_members))
        donor=PredictiveOperation._array(inp['selected_donor'],len(odd.target_members))
        if recipient.shape!=donor.shape:raise ValueError('Recipient and donor shapes differ')
        odd_part=odd.predict_selected((recipient-donor)/2)
        common_part=common.predict_selected((recipient+donor)/2)
        result=dict(removal_delta=-(odd_part+common_part),contrast_delta=odd.predict_selected(donor-recipient),
            odd_contribution=odd_part,common_contribution=common_part,target_members=odd.target_members,
            metadata_json=np.array(json.dumps(dict(paired_completion_id=odd.metadata['paired_completion_id'],
                output_kind='Donor-conditioned source-aligned residual update; not native or semantic deletion',requires_reciprocal_donor=True))))
        if 'row_ids' in inp:result['row_ids']=inp['row_ids']
    with args.output.open('xb') as f:np.savez_compressed(f,**result)
    print(json.dumps(dict(output=str(args.output.resolve()),shape=list(result['removal_delta'].shape),requires_reciprocal_donor=True)))


if __name__=='__main__':main()
