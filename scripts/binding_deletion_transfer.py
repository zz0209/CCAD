"""Reuse frozen correspondence for source deletions absent from fitting."""
import json
from pathlib import Path


def deletion_transfers(w,cfg,rows,zs,sae,target,zt,seed,target_seed):
    import numpy as np
    import torch
    from run_causalgym_multisite import ROOT,write
    spec=cfg['deletion_transfer'];parent=ROOT/spec['run']
    assert json.loads(w.checked(parent/'status.json').read_text())['status']=='PASS'
    prior=json.loads(w.checked(parent/'config.resolved.json').read_text())
    for key in ['model_revision','training_runs','checkpoint_step','layers','semantic_queries']:
        assert cfg[key]==prior[key]
    assert json.loads(w.checked(parent/'panel.json').read_text())['rows']==rows
    with np.load(w.checked(parent/f'binding_relation_s{seed}_t{target_seed}.npz')) as z:
        si=torch.tensor(z['source_indices'],device=w.device);ti=torch.tensor(z['target_indices'],device=w.device)
        q=torch.tensor(z['source_query'],device=w.device)
    with np.load(w.checked(parent/f'selected_writer_s{seed}_t{target_seed}.npz')) as z:
        assert np.array_equal(z['source_indices'],si.cpu().numpy()) and np.array_equal(z['target_indices'],ti.cpu().numpy())
        matrices={name:torch.tensor(z['native_'+name],device=w.device) for name in ['gain','mix']}
    with np.load(w.checked(parent/f'selected_writer_candidates_s{seed}_t{target_seed}.npz')) as z:
        matrices['geometry']=torch.tensor(z['initial'],device=w.device)
    ds=sae.decoder.weight.T[si];dt=target.decoder.weight.T[ti]
    amplitude=-zs[...,si]*q
    outputs={'source_delete':amplitude@ds};diagnostics=[]
    source_current=zs[...,si]
    assert float((source_current+amplitude).min())>=-1e-6
    for name,matrix in matrices.items():
        proposed=amplitude@matrix
        for mode in ['deletion','general']:
            coeff=torch.maximum(proposed,-zt[...,ti])
            if mode=='deletion':coeff=coeff.clamp_max(0)
            keep=torch.argsort(coeff.abs()*dt.norm(dim=1),dim=-1,descending=True,stable=True)[...,:spec['members']]
            coeff=torch.zeros_like(coeff).scatter(-1,keep,coeff.gather(-1,keep))
            outputs[mode+'_'+name]=(coeff@dt).detach()
            diagnostics.append(dict(method=mode+'_'+name,min_edited_code=float((zt[...,ti]+coeff).min()),
                                    max_changed_members=int((coeff!=0).sum(-1).max()),
                                    positive_coefficients=int((coeff>0).sum()),
                                    relative_source_field_error=float(((coeff@dt)-outputs['source_delete']).square().sum()/outputs['source_delete'].square().sum())))
    if spec.get('project_current_codes'):
        from adaptive_native_execution import realize
        old_precision=torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision('highest')
        field,coeff,indices,info=realize(outputs['source_delete'],-zt[...,ti],dt,
                                       members=spec['members'],**spec['project_current_codes'])
        outputs['projected_deletion']=field.detach()
        info.update(method='projected_deletion',constraints='-z <= c <= 0; current target activations define the bounds',
                    information='Exact source deletion field; no output distributions or gradients')
        diagnostics.append(info)
        np.savez_compressed(w.run/f'deletion_projection_s{seed}_t{target_seed}.npz',
                            coefficients=coeff.cpu().numpy(),indices=ti[indices].cpu().numpy())
        torch.set_float32_matmul_precision(old_precision)
    write(w.run/f'DELETION_TRANSFER_s{seed}.json',dict(parent=parent.as_posix(),records=diagnostics,
          source_request='Delete members selected for a frozen country contrast; query depends on both recipient and donor country',
          amplitude_information='Exact source code supplied equally to every matrix; no target answer labels or new response fitting',
          target_operations={'deletion':'Only attenuate currently active target codes; -z <= c <= 0',
                             'general':'Nonnegative edited codes; c >= -z, allowing increases'},
          maximum_changed_target_members=spec['members'],new_fits=0,new_output_gradients=0))
    return outputs
