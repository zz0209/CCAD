from pathlib import Path
from datetime import datetime,timezone
import json
import sys
import numpy as np
import torch
from ccad.intervention_transport import transport_delta,transport_field


ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(2)
    c=json.loads((ROOT/'configs/rg01_state_parts_smoke.json').read_text())
    sys.path.extend([c['dictionary_source_dir'],c['dictionary_overlay_dir']])
    from dictionary_learning.trainers.top_k import AutoEncoderTopK
    state=torch.load(Path(c['target_directory'])/'resid_4_seed3.pt',map_location='cpu',weights_only=True)
    target=AutoEncoderTopK(512,len(state['encoder.weight']),int(state['k']))
    target.load_state_dict(state)
    target.requires_grad_(False)
    h=torch.load(c['natural_cache'],map_location='cpu',weights_only=True)['resid_4'][-32:]
    bank=np.load(ROOT/c['grammar_source_parameters'])
    source={k:torch.tensor(bank[k]) for k in ['center','encoder','encoder_bias','decoder']}
    zs=torch.relu((h-source['center'])@source['encoder'].T+source['encoder_bias'])
    field=-zs.unsqueeze(-2)*source['decoder'].T
    maximum=0.; minimum=0.; changed=0
    with torch.no_grad():
        for bits in range(16):
            q=torch.tensor([(bits>>i)&1 for i in range(4)],dtype=torch.float32)
            original,_,_=transport_delta(h,target,source,q,target.encoder.weight,8,True)
            actual,dz,columns=transport_field(h,target,field,q,8,True)
            maximum=max(maximum,float((actual-original).abs().max()))
            minimum=min(minimum,float((target.encode(h)+dz).min()))
            changed=max(changed,int((dz!=0).sum(-1).max()))
            assert torch.allclose(actual,original,atol=2e-5,rtol=1e-5)
            assert minimum>=-2e-5 and changed<=8
    old=Path('D:/CCAD_Storage/runs/final_science_20260921_round05/FS05_GENERIC_LOCAL_AUDIT_20260921/responses.npz')
    current=Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round01/RG01_LOCAL_20260921/responses.npz')
    a=np.load(old); b=np.load(current)
    assert all(np.array_equal(a[k],b[k]) for k in a.files)
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),actual_states=32,requests=16,
                active_states=int((zs.sum(-1)>0).sum()),maximum_difference=maximum,minimum_code=minimum,
                maximum_members=changed,unchanged_512_update_predictions=True)
    assert result['active_states']>0
    dest=ROOT/'artifacts/reuse_generalization_20260921_round01/FIELD_INTERFACE_CHECK.json'
    assert not dest.exists()
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
