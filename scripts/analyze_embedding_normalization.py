from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import torch
from safetensors import safe_open


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/reuse_generalization_20260921_round02'
BULK=Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round02')


def main():
    torch.set_num_threads(2)
    output=OUT/'NORMALIZATION_DIAGNOSTIC.json'
    if output.exists():raise FileExistsError(output)
    run=BULK/'RG02_FINITE_SUPPORT_DEVELOPMENT_T2_20260921'
    c=json.loads((run/'config.resolved.json').read_text())
    initial=torch.load(run/'initial_execution.pt',map_location='cpu',weights_only=True)
    tokens=initial['active_tokens']
    with safe_open(Path(c['model_local_dir'])/'model.safetensors',framework='pt',device='cpu') as f:
        x=f.get_tensor('gpt_neox.embed_in.weight')[tokens].float()
        weight=f.get_tensor('gpt_neox.layers.0.input_layernorm.weight').float()
        bias=f.get_tensor('gpt_neox.layers.0.input_layernorm.bias').float()
    cfg=json.loads((Path(c['model_local_dir'])/'config.json').read_text())
    eps=cfg['layer_norm_eps']
    bank=np.load(c['source_parameters'])
    zs=torch.relu((x-torch.from_numpy(bank['embed__center']))@torch.from_numpy(bank['embed__encoder']).T
                  +torch.from_numpy(bank['embed__encoder_bias']))
    source=-torch.from_numpy(bank['embed__decoder']).T[None]*zs[:,None,:]
    td=torch.load(Path(c['target_directory'])/'embed_seed2.pt',map_location='cpu',weights_only=True)['decoder.weight']
    geometric=td.T[initial['support']].transpose(-1,-2)@initial['coefficients'][:-1]
    sparse=td[None]@torch.load(run/'sparse_256.pt',map_location='cpu',weights_only=True)['coefficients'][:-1]
    fit=json.loads((run/'membership.json').read_text())['fit']
    counts=torch.bincount(torch.tensor([t for row in fit for t in row['tokens']]),minlength=max(tokens)+1)[tokens].double()
    counts=counts/counts.sum()
    rng=np.random.default_rng(921026)
    q=torch.tensor(np.concatenate([np.eye(source.shape[-1]),rng.uniform(0,1,(64,source.shape[-1]))]),dtype=x.dtype)
    def norm(value):return torch.nn.functional.layer_norm(value,(512,),weight,bias,eps)
    def jacobian(value):
        centered=x-x.mean(-1,keepdim=True)
        variance=centered.square().mean(-1,keepdim=True)+eps
        projected=value-value.mean(-1,keepdim=True)
        return weight/variance.sqrt()*(projected-centered*(centered*value).mean(-1,keepdim=True)/variance)
    clean=norm(x)
    effects={name:torch.einsum('ndp,qp->qnd',field,q) for name,field in [('source',source),('geometric',geometric),('sparse',sparse)]}
    source_norm=norm(x+effects['source'])-clean
    def total(value):return float((value.double().square().mean((0,2))*counts).sum())
    values={}
    for name,delta in effects.items():
        actual=norm(x+delta)-clean
        values[name]=dict(raw_error=total(delta-effects['source'])/max(total(effects['source']),1e-15),
            normalized_error=total(actual-source_norm)/max(total(source_norm),1e-15),
            jacobian_relative_error=total(actual-jacobian(delta))/max(total(actual),1e-15))
    small=effects['source'][0]*1e-3
    finite=(norm(x+small)-norm(x-small))/2
    discrepancy=float((finite-jacobian(small)).norm()/finite.norm())
    output.write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        evidence='Source fit token frequencies,74predefined member requests and frozen development fields. No confirmation output used.',
        active_source_tokens=len(tokens),observed_source_tokens=int((counts>0).sum()),
        source_fit_token_occurrences=sum(t in set(tokens.tolist()) for row in fit for t in row['tokens']),
        layer_norm_eps=eps,small_step_central_difference_relative_error=discrepancy,metrics=values),indent=2)+'\n')
    print(output.read_text())


if __name__=='__main__':
    main()
