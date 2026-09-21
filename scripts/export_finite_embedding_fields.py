from pathlib import Path
import argparse
import json

import numpy as np
import torch
from safetensors import safe_open

from ccad.intervention_transport import project_capacity


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    run=args.run
    c=json.loads((run/'config.resolved.json').read_text())
    assert json.loads((run/'status.json').read_text())['status']=='PASS'
    initial=torch.load(run/'initial_execution.pt',map_location='cpu',weights_only=True)
    tokens=initial['active_tokens']
    bank=np.load(c['source_parameters'])
    with safe_open(Path(c['model_local_dir'])/'model.safetensors',framework='pt',device='cpu') as f:
        embedding=f.get_tensor('gpt_neox.embed_in.weight')[tokens]
    scale=torch.relu((embedding-torch.from_numpy(bank['embed__center']))@torch.from_numpy(bank['embed__encoder']).T
                     +torch.from_numpy(bank['embed__encoder_bias']))
    sd=torch.load(Path(c['target_directory'])/f'embed_seed{c["target_seed"]}.pt',map_location='cpu',weights_only=True)
    d=sd['decoder.weight'].T[initial['support']]
    for arm in c['arms']:
        checkpoint=torch.load(run/f'{arm}_{c["steps"]}.pt',map_location='cpu',weights_only=True)
        base=initial['coefficients'][:-1]
        proposal=base*checkpoint['gain'][:-1,None,:] if arm=='gain' else base+checkpoint['residual'][:-1]*scale[:,None,:]
        if arm in ['sparse','fixed_direct','regrow']:
            columns=checkpoint['coefficients'][:-1]
            field=sd['decoder.weight'][None]@columns
        else:
            columns=project_capacity(proposal,initial['capacity'][:-1])*(scale[:,None,:]>0)
            field=d.transpose(-1,-2)@columns
        path=run/f'{arm}_field.pt'
        if path.exists():raise FileExistsError(path)
        torch.save(dict(active_tokens=tokens,field=field),path)
        print(path)


if __name__=='__main__':
    main()
