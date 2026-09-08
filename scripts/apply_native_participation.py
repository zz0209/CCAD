"""Apply a saved native participation map to aligned target SAE codes.

NumPy only; writes an actual target-decoder hook update. This does not run the
language model, rerun a TopK encoder, choose source components or fit a map.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ccad.native_participation import participant_delta,feasibility


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map',type=Path,required=True)
    parser.add_argument('--states',type=Path,required=True)
    parser.add_argument('--controls',required=True,help='Comma-separated source-component strengths in [0,1]')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Keep prior outputs; choose a new output path')
    mapping=np.load(args.map,allow_pickle=False);states=np.load(args.states,allow_pickle=False)
    g,decoder=mapping['participation'],mapping['target_decoder']
    base,donor=states['base_codes'],states['aligned_donor_codes']
    controls=np.asarray([float(v) for v in args.controls.split(',')],dtype=base.dtype)
    if base.shape!=donor.shape or base.ndim<1 or g.ndim!=2 or decoder.ndim!=2:
        raise ValueError('Mismatched target-code and map dimensions')
    if base.shape[-1]!=g.shape[0] or decoder.shape[0]!=g.shape[0] or controls.shape!=(g.shape[1],):
        raise ValueError('Map, decoder, target states and controls must share their stated axes')
    if not all(np.isfinite(v).all() for v in [base,donor,g,decoder,controls]):raise ValueError('Nonfinite input')
    if min(base.min(),donor.min(),controls.min())<0 or controls.max()>1:raise ValueError('Codes must be nonnegative; controls must be in [0,1]')
    info=feasibility(g)
    if not info['nonnegative'] or not info['row_sum_at_most_one']:raise ValueError('Infeasible saved participation')
    delta,final=participant_delta(base,donor,decoder,g,controls)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,hook_delta=delta,final_selected_codes=final,controls=controls,target_members=mapping['target_members'])
    print(json.dumps(dict(output=str(args.output.resolve()),hook_delta_shape=list(delta.shape),
        minimum_final_selected_code=float(final.min()),target_members=len(mapping['target_members']),
        scope='Aligned positions and selected members supplied by caller; preserve original residual and all unedited positions when applying hook_delta.')))


if __name__=='__main__':main()
