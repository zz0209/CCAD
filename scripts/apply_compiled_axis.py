"""Apply one saved native operator to target SAE code pairs, without a GPU.

Input NPZ: base_codes, donor_codes; optional decoder [members,hidden].
The user must obtain codes with the exact target SAE identified in the run.
The output is a native update and hook-space delta, not generated text.
"""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ccad.compiled_axis import CompiledAxis

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--operator',required=True,type=Path);p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    operator=CompiledAxis.load(a.operator)
    with np.load(a.input,allow_pickle=False) as data:
        dec=data['decoder'] if 'decoder' in data else None
        result=operator.apply(data['base_codes'],data['donor_codes'],decoder=dec)
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**result)
    receipt=dict(operator=str(a.operator.resolve()),operator_sha256=hashlib.sha256(a.operator.read_bytes()).hexdigest(),input=str(a.input.resolve()),input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),output=str(a.output.resolve()),rows=len(result['coordinate']),max_changed_members=int(result['changed_members'].max(initial=0)),minimum_final_selected=float(result['final_selected_codes'].min(initial=0)),decoder_recomputed=dec is not None,cached_delta_max_abs_difference=float(np.max(np.abs(result['hidden_delta']-result['cached_hidden_delta']),initial=0)),scope='NumPy application of an existing operator to target-only actual SAE codes. No text encoding, source supervision, fitting, LM execution or semantic validation.')
    a.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
