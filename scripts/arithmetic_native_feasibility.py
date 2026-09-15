"""Small saved-state development check of legal code realization, without an LM."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time
import numpy as np
import torch
from scipy.optimize import lsq_linear
ROOT=Path(__file__).resolve().parents[1]
def main():
    start=datetime.now(timezone.utc);timer=time.perf_counter();torch.set_num_threads(2)
    run=ROOT/'runs/REFORM_R43_arithmetic_part_dose_pilot_v1_20260915'
    panel=json.loads((run/'panel.json').read_text());pairs=panel['pairs'];rows=panel['rows']
    z=np.load(run/'evaluation_seed2.npz')['codes'];rel=np.load(ROOT/'runs/REFORM_R38_qwen_member_fields_five_v1_20260914/relation_s1_t2.npz')
    pred=np.load(ROOT/'runs/REFORM_R39_qwen_query_readouts_v4_20260914/readout_s1_t2.npz')
    p=Path('D:/CCAD_Storage/training_curves/REFORM_R32_qwen_l23_topk_five_seed_16m_v1_20260914/step_16384/topk_seed2.pt')
    weights=torch.load(p,map_location='cpu',weights_only=True);decoder=weights['decoder.weight'].numpy().T
    out=[]
    for op in ['unit','tens']:
        ix=pred[op+'_target_indices'];d=decoder[ix].astype(float)
        for j in [0,1,16,17]:
            pair=pairs[j];a,b=pair['recipient'],pair['donor'];role=1-rows[a]['template']
            dx=z[b,0,ix]-z[a,0,ix];y=(dx@pred[op+'_activation'][role])@pred[op+'_decoder']
            old=(dx*rel['clean'][role,ix,0 if op=='unit' else 1])@d
            result=lsq_linear(d.T,y,bounds=(-z[a,0,ix].astype(float),np.full(len(ix),np.inf)),method='bvls',tol=1e-8)
            den=max(float(y@y),1e-12)
            out.append(dict(operation=op,pair=j,role=role,old_relative_error=float(np.square(old-y).sum()/den),
                legal_relative_error=float(np.square(d.T@result.x-y).sum()/den),min_edited_code=float(np.min(z[a,0,ix]+result.x)),solver_status=int(result.status)))
    end=datetime.now(timezone.utc)
    data=dict(started_at_utc=start.isoformat(),ended_at_utc=end.isoformat(),wall_seconds=time.perf_counter()-timer,
        rows=out,scope='Eight exposed first-position saved-state full requests on source1/target2. Fixed64target bank. Classical bounded least squares versus old interpolation; no LM outputs, generation, or performance claim.',
        checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'artifacts/correspondence_reform_20260913/R43_NATIVE_FEASIBILITY.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
if __name__=='__main__':main()
