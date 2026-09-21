from pathlib import Path
from datetime import datetime, timezone
import json
import sys

import numpy as np
import torch
from scipy.optimize import lsq_linear

from ccad.intervention_transport import transport_delta, refine_columns
from ccad.request_realization import refine_request


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'artifacts/reuse_generalization_20260921_round03/NUMERICAL_CHECK.json'
    if out.exists():raise FileExistsError(out)
    c=json.loads((root/'configs/rg03_request_realization_development.json').read_text())
    sys.path.extend([c['dictionary_source_dir'],c['dictionary_overlay_dir']])
    from dictionary_learning.trainers.top_k import AutoEncoderTopK
    torch.set_num_threads(2)
    bank=np.load(c['grammar_source_parameters'])
    sp={k:torch.tensor(bank[k]) for k in ['encoder','encoder_bias','decoder','center']}
    sd=torch.load(Path(c['target_directory'])/'resid_4_seed2.pt',map_location='cpu',weights_only=True)
    target=AutoEncoderTopK(512,len(sd['encoder.weight']),int(sd['k']))
    target.load_state_dict(sd);target.requires_grad_(False)
    states=torch.load(c['natural_cache'],map_location='cpu',weights_only=True)['resid_4']
    zs=torch.relu((states-sp['center'])@sp['encoder'].T+sp['encoder_bias'])
    indices=((zs>0).sum(-1)>=2).nonzero().flatten()[:8]
    states=states[indices];zs=zs[indices]
    candidate=-(target.encoder.weight@sp['decoder'].T)[None]*zs[:,None,:]
    ids=(candidate.abs().sum(-1)*target.decoder.weight.norm(dim=0)).topk(8,dim=-1).indices
    capacity=target.encode(states).gather(1,ids)
    q=torch.ones(4)
    _,_,common=transport_delta(states,target,sp,q,target.encoder.weight,8,False)
    common=refine_columns(states,target,sp,common,8,128,accelerate=True)
    records=[]
    for name,q in [('full',torch.ones(4)),('half',torch.full((4,),.5)),('part',torch.tensor([1.,0.,1.,0.])),('zero',torch.zeros(4))]:
        for bounded in [True,False]:
            delta,detail=refine_request(states,target,sp,q,common,8,128,bounded)
            if name=='zero':assert torch.equal(delta,torch.zeros_like(delta))
            for j,index in enumerate(indices.tolist()):
                decoder=target.decoder.weight[:,ids[j]].numpy().astype(float)
                desired=(-(zs[j]*q)@sp['decoder']).numpy().astype(float)
                bounds=(-capacity[j].numpy().astype(float),np.inf) if bounded else (-np.inf,np.inf)
                reference=lsq_linear(decoder,desired,bounds=bounds,lsq_solver='exact',tol=1e-11,max_iter=1000)
                assert reference.success
                mse=float(np.square(decoder@reference.x-desired).sum())
                energy=max(float(np.square(desired).sum()),1e-12)
                value=float(detail['request_error'][j])
                assert value>=mse-2e-5*max(1.,energy)
                assert value<=float(detail['common_error'][j])+1e-6
                if bounded:assert float(detail['minimum_code'][j])>=-1e-6
                records.append(dict(state=index,request=name,bounded=bounded,relative_optimality_gap=(value-mse)/energy,
                    common_error=float(detail['common_error'][j]),request_error=value,scipy_error=mse))
    out.write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        actual_cached_states=indices.tolist(),rows=records,checks='Independent SciPy optimum, feasible native codes, nonincreasing objective and zero request.',
        maximum_relative_optimality_gap=max(r['relative_optimality_gap'] for r in records)),indent=2)+'\n')
    print(json.dumps(dict(rows=len(records),maximum_relative_optimality_gap=max(r['relative_optimality_gap'] for r in records))))


if __name__=='__main__':
    main()
