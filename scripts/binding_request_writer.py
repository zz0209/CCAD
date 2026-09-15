"""Learn source-query resolved code writers on single binding requests."""
import json
import time


def fit_binding_writers(w,cfg,rows,hidden,zs,sae,target,zt,seed,target_seed,forward,layer):
    if cfg['request_writer_fit'].get('select_on_source_fit'):
        from binding_selected_writer import fit_selected_writers
        return fit_selected_writers(w,cfg,rows,hidden,zs,sae,target,zt,seed,target_seed,forward,layer)
    import numpy as np
    import torch
    from run_causalgym_multisite import write
    spec=cfg['request_writer_fit'];started=time.perf_counter()
    previous_precision=torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(spec.get('fit_precision',previous_precision))
    with np.load(w.checked(w.run/f'binding_relation_s{seed}_t{target_seed}.npz')) as z:
        si=torch.tensor(z['source_indices'],device=w.device)
        ti=torch.tensor(z['target_indices'],device=w.device)
        q=torch.tensor(z['source_query'],device=w.device)
        readers={k:torch.tensor(z[k+'_readout'],device=w.device) for k in ['raw','code']}
    ds=sae.decoder.weight.T[si].detach();dt=target.decoder.weight.T[ti].detach()
    paired=[r['paired_row'] for r in rows]
    y=zs[paired][...,si]-zs[...,si]
    predictions={'native':(zt[paired][...,ti]-zt[...,ti])@readers['code'],
                 'raw':(hidden[paired]-hidden)@readers['raw']}
    gram=dt@dt.T
    initial=torch.linalg.solve(gram+spec['decoder_ridge']*torch.eye(len(ti),device=w.device),dt@ds.T).T
    matrices={'native':torch.nn.Parameter(initial),'raw':torch.nn.Parameter(torch.eye(len(si),device=w.device))}
    opts={k:torch.optim.Adam([v],lr=spec['lr']) for k,v in matrices.items()}
    fits=[r['row_id'] for r in rows if r['split']=='fit' and not r['donor']]
    assert all(rows[i]['split']=='fit' for i in fits)
    rng=np.random.default_rng(spec['seed']+seed)
    outputs={};payload=dict(source_indices=si.cpu().numpy(),target_indices=ti.cpu().numpy())
    calls=dict(teacher_forward=0,native_forward=0,raw_forward=0,native_backward=0,raw_backward=0)
    diagnostics=[]

    def field(kind,ii,support=None):
        coeff=(predictions[kind][ii]*q[ii])@matrices[kind]
        if kind=='native':
            coeff=torch.maximum(coeff,-zt[ii][...,ti])
            # Fixed bank; at most64 changes, chosen by decoded contribution norm.
            selected=(torch.argsort(coeff.abs()*dt.norm(dim=1),descending=True,stable=True)[...,:cfg['binding_transfer']['query_members']]
                      if support is None else support)
            coeff=torch.zeros_like(coeff).scatter(-1,selected,coeff.gather(-1,selected))
            return coeff@dt,coeff,selected
        return coeff@ds,coeff,None

    def save(update):
        with torch.no_grad():
            ii=list(range(len(rows)))
            for kind,matrix in matrices.items():
                delta,coeff,_=field(kind,ii)
                outputs[f'learned_{kind}_u{update}_country']=delta.detach()
                payload[f'{kind}_{update}']=matrix.detach().cpu().numpy().copy()
                if kind=='native':
                    diagnostics.append(dict(update=update,min_edited_code=float((zt[...,ti]+coeff).min()),
                                            max_changed_members=int((coeff!=0).sum(-1).max())))
            if spec.get('fit_probe_rows'):
                selected=[fits[int(i)] for i in np.linspace(0,len(fits)-1,spec['fit_probe_rows'],dtype=int)]
                losses={k:[] for k in matrices}
                for off in range(0,len(selected),cfg['batch_size']):
                    jj=selected[off:off+cfg['batch_size']]
                    for site in [0,1]:
                        active=(torch.arange(2,device=w.device)==site)[None,:,None]
                        teacher=forward(jj,layer,(y[jj]*q[jj])@ds*active);prob=teacher.exp();calls['teacher_forward']+=1
                        for kind in matrices:
                            dd,_,_=field(kind,jj);student=forward(jj,layer,dd*active)
                            calls[kind+'_forward']+=1
                            losses[kind].extend((prob*(teacher-student)).sum(-1).cpu().tolist())
                w.record(kind='fit_probe',task='binding',row_id=update,component=str(seed),method='matched_native_raw',
                         mode=f'layer{layer}',seed=seed,target_seed=target_seed,operation='singleton',
                         losses={k:float(np.mean(v)) for k,v in losses.items()},fit_rows=selected)
        np.savez_compressed(w.run/f'binding_request_writer_s{seed}_t{target_seed}.npz',**payload)

    save(0)
    for step in range(spec['updates']):
        ii=rng.choice(fits,size=cfg['batch_size'],replace=False).tolist()
        sites=torch.tensor(rng.integers(0,2,len(ii)),device=w.device)
        active=(torch.arange(2,device=w.device)[None]==sites[:,None])[...,None]
        with torch.no_grad():
            teacher=forward(ii,layer,(y[ii]*q[ii])@ds*active)
            prob=teacher.exp()
            calls['teacher_forward']+=1
        losses={}
        for kind,matrix in matrices.items():
            opts[kind].zero_grad(set_to_none=True)
            delta,_,support=field(kind,ii)
            student=forward(ii,layer,delta*active,grad=True)
            calls[kind+'_forward']+=1
            loss=(prob*(teacher-student)).sum(-1).mean()
            assert torch.isfinite(loss)
            loss.backward();calls[kind+'_backward']+=1
            if step==0:
                # Autograd differentiates coefficients on the selected support.
                # Check that smooth branch, and also report the unconstrained
                # finite difference and any change in the discrete top64 set.
                coordinate=np.unravel_index(int(matrix.grad.abs().argmax()),matrix.shape)
                analytic=float(matrix.grad[coordinate]);old=float(matrix[coordinate].detach());vals=[];free=[];changed=[]
                with torch.no_grad():
                    for sign in [-1,1]:
                        matrix[coordinate]=old+sign*spec['gradient_epsilon']
                        dd,_,_=field(kind,ii,support);ss=forward(ii,layer,dd*active)
                        vals.append(float((prob*(teacher-ss)).sum(-1).mean()))
                        calls[kind+'_forward']+=1
                        if kind=='native':
                            dd,_,perturbed=field(kind,ii);ss=forward(ii,layer,dd*active)
                            free.append(float((prob*(teacher-ss)).sum(-1).mean()))
                            changed.append(int((perturbed.sort(-1).values!=support.sort(-1).values).sum()))
                            calls[kind+'_forward']+=1
                    matrix[coordinate]=old
                fd=(vals[1]-vals[0])/(2*spec['gradient_epsilon'])
                w.record(kind='gradient_check',task='binding',row_id=0,component=str(seed),method=kind,
                         mode=f'layer{layer}',seed=seed,target_seed=target_seed,analytic=analytic,finite_difference=fd,
                         scope='Fixed selected support; derivatives through coefficients and code clipping',
                         unfixed_difference=None if not free else (free[1]-free[0])/(2*spec['gradient_epsilon']),
                         changed_support_coordinates=changed)
                assert abs(analytic-fd)<.015+.08*abs(analytic),(kind,analytic,fd)
            torch.nn.utils.clip_grad_norm_([matrix],spec['gradient_clip']);opts[kind].step()
            losses[kind]=float(loss.detach())
        w.record(kind='request_fit',task='binding',row_id=step,component=str(seed),method='matched_native_raw',
                 mode=f'layer{layer}',seed=seed,target_seed=target_seed,operation='singleton',
                 fit_rows=ii,sites=sites.cpu().tolist(),losses=losses)
        if step+1 in spec['save_updates']:save(step+1)
        if (step+1)%16==0:w.progress('BINDING_REQUEST_FIT',seed=seed,updates=step+1,**losses,**calls)
    info=dict(source_seed=seed,target_seed=target_seed,seconds=time.perf_counter()-started,calls=calls,
              fit_precision=torch.get_float32_matmul_precision(),
              diagnostics=diagnostics,fit_rows=len(fits),training_operations=['first','second'],
              evaluation_unfitted_operation='both',target_answer_labels=0,
              supervision='Actual source-country component response; matched target and raw output learning',
              inference='Stored member-to-coefficient matrix, code projection, top64; no output fitting')
    write(w.run/f'BINDING_REQUEST_FIT_s{seed}.json',info)
    torch.set_float32_matmul_precision(previous_precision)
    return outputs
