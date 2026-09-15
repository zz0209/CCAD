"""Reuse existing contextual native projection for a frozen carry request."""
import time
import numpy as np
import torch
from ccad.native_operation import adaptive_writable_support, batched_project_native


def native_rule_write(ae, hidden, codes, desired, decoder, decoder_numpy, operator, cfg):
    torch.cuda.synchronize();start=time.perf_counter()
    if operator=='reencode':
        edited=ae.encode(hidden+desired);increments=edited-codes
        realized=increments@decoder
        details=[dict(members=torch.where(row!=0)[0].cpu().tolist(),
                      increments=row[row!=0].cpu().tolist(),minimum_final_code=float(edited[i].min()))
                 for i,row in enumerate(increments)]
    else:
        budget=int(operator.split('_')[1])
        selected=adaptive_writable_support(desired.cpu().numpy(),codes.cpu().numpy(),decoder_numpy,
                    budget,np.ones(len(decoder),dtype=bool),device=str(hidden.device))
        index=torch.tensor(selected,device=hidden.device);local_z=codes.gather(1,index);local_d=decoder[index]
        increments,realized,diagnostic=batched_project_native(desired,local_z,local_d,
                    max_steps=cfg['native_projection_steps'],tolerance=cfg['native_projection_tolerance'])
        assert diagnostic['minimum_final_state']>=-1e-6
        details=[dict(members=selected[i].tolist(),increments=increments[i].cpu().tolist(),
                      minimum_final_code=float((local_z[i]+increments[i]).min()),
                      relative_projected_gradient=diagnostic['relative_projected_gradient'][i],
                      steps=diagnostic['steps']) for i in range(len(hidden))]
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    for i,item in enumerate(details):
        item.update(desired_norm=float(desired[i].norm()),realized_norm=float(realized[i].norm()),
                    relative_field_error=float((realized[i]-desired[i]).norm()/desired[i].norm().clamp_min(1e-8)),
                    batch_solve_seconds=elapsed,batch_size=len(hidden),
                    timing_scope='Same batch time repeated on each member row; divide by batch_size when summing.')
    return realized,details
