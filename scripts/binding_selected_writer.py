"""Compare learned mixing and source gains with fit-only model selection."""
import json
import time


def fit_selected_writers(w,cfg,rows,hidden,zs,sae,target,zt,seed,target_seed,forward,layer):
    import numpy as np
    import torch
    from run_causalgym_multisite import write
    spec=cfg['request_writer_fit'];selection=spec['select_on_source_fit']
    started=time.perf_counter();old_precision=torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(spec['fit_precision'])
    with np.load(w.checked(w.run/f'binding_relation_s{seed}_t{target_seed}.npz')) as z:
        si=torch.tensor(z['source_indices'],device=w.device);ti=torch.tensor(z['target_indices'],device=w.device)
        q=torch.tensor(z['source_query'],device=w.device)
        raw_reader=torch.tensor(z['raw_readout'],device=w.device);code_reader=torch.tensor(z['code_readout'],device=w.device)
    ds=sae.decoder.weight.T[si].detach();dt=target.decoder.weight.T[ti].detach()
    paired=[r['paired_row'] for r in rows]
    source=zs[paired][...,si]-zs[...,si]
    predicted={'native':(zt[paired][...,ti]-zt[...,ti])@code_reader,'raw':(hidden[paired]-hidden)@raw_reader}
    initial=torch.linalg.solve(dt@dt.T+spec['decoder_ridge']*torch.eye(len(ti),device=w.device),dt@ds.T).T.detach()
    fits=[r['row_id'] for r in rows if r['split']=='fit' and not r['donor']]
    contexts=sorted({rows[i]['component'] for i in fits})
    rng=np.random.default_rng(spec['seed']+seed)
    shuffled=np.array(contexts)[rng.permutation(len(contexts))]
    val_contexts=set(shuffled[:selection['validation_contexts']].tolist())
    training=[i for i in fits if rows[i]['component'] not in val_contexts]
    validation=[i for i in fits if rows[i]['component'] in val_contexts]
    assert set(rows[i]['component'] for i in training).isdisjoint(rows[i]['component'] for i in validation)
    assert all(rows[i]['split']=='fit' for i in training+validation)
    assert training and validation
    families=['native_mix','native_gain','raw_mix'];definitions={};parameters={}
    for family in families:
        for index,lr in enumerate(selection['learning_rates']):
            name=f'{family}_lr{index}';definitions[name]=dict(family=family,lr=lr)
            start=(torch.ones(len(si),device=w.device) if family=='native_gain' else
                   initial.clone() if family=='native_mix' else torch.eye(len(si),device=w.device))
            parameters[name]=torch.nn.Parameter(start)
    opts={k:torch.optim.Adam([p],lr=definitions[k]['lr']) for k,p in parameters.items()}
    calls={'teacher_forward':0,**{k:{'train_forward':0,'backward':0,'validation_forward':0,'gradient_forward':0} for k in parameters}}
    best={family:None for family in families};history=[]
    payload=dict(source_indices=si.cpu().numpy(),target_indices=ti.cpu().numpy(),
                 training_rows=np.array(training),validation_rows=np.array(validation),initial=initial.cpu().numpy())

    def matrix(name,param=None):
        p=parameters[name] if param is None else param
        return p[:,None]*initial if definitions[name]['family']=='native_gain' else p

    def field(name,ii,param=None,support=None):
        native=definitions[name]['family'].startswith('native')
        coeff=(predicted['native' if native else 'raw'][ii]*q[ii])@matrix(name,param)
        if native:
            coeff=torch.maximum(coeff,-zt[ii][...,ti])
            selected=(torch.argsort(coeff.abs()*dt.norm(dim=1),descending=True,stable=True)[...,:cfg['binding_transfer']['query_members']]
                      if support is None else support)
            coeff=torch.zeros_like(coeff).scatter(-1,selected,coeff.gather(-1,selected))
            return coeff@dt,coeff,selected
        return coeff@ds,coeff,None

    # All validation teachers use singleton requests on source-fit contexts.
    # Cache their distributions once; no development rows enter selection.
    val_batches=[]
    with torch.no_grad():
        for off in range(0,len(validation),cfg['batch_size']):
            ii=validation[off:off+cfg['batch_size']]
            for site in [0,1]:
                active=(torch.arange(2,device=w.device)==site)[None,:,None]
                teacher=forward(ii,layer,(source[ii]*q[ii])@ds*active)
                calls['teacher_forward']+=1
                val_batches.append((ii,active,teacher,teacher.exp()))

    def validate(update):
        with torch.no_grad():
            for name,p in parameters.items():
                losses=[]
                for ii,active,teacher,prob in val_batches:
                    dd,_,_=field(name,ii);student=forward(ii,layer,dd*active)
                    calls[name]['validation_forward']+=1
                    losses.extend((prob*(teacher-student)).sum(-1).cpu().tolist())
                value=float(np.mean(losses));assert np.isfinite(value)
                record=dict(candidate=name,update=update,loss=value,**definitions[name]);history.append(record)
                family=definitions[name]['family']
                if best[family] is None or value<best[family]['loss']:
                    best[family]=dict(record,parameter=p.detach().clone())
                payload[f'{name}_u{update}']=p.detach().cpu().numpy().copy()
                w.record(kind='writer_validation',task='binding',row_id=update,component=str(seed),
                         mode=f'layer{layer}',method=name,seed=seed,target_seed=target_seed,
                         operation='singleton',source_fit_validation_kl=value,validation_rows=len(validation))
        np.savez_compressed(w.run/f'selected_writer_candidates_s{seed}_t{target_seed}.npz',**payload)
        w.progress('WRITER_SELECTION',seed=seed,update=update,
                   best={f:{k:v for k,v in r.items() if k!='parameter'} for f,r in best.items()})

    validate(0)
    for step in range(spec['updates']):
        ii=rng.choice(training,size=cfg['batch_size'],replace=False).tolist()
        sites=torch.tensor(rng.integers(0,2,len(ii)),device=w.device)
        active=(torch.arange(2,device=w.device)[None]==sites[:,None])[...,None]
        with torch.no_grad():
            teacher=forward(ii,layer,(source[ii]*q[ii])@ds*active);prob=teacher.exp();calls['teacher_forward']+=1
        losses={}
        for name,p in parameters.items():
            opts[name].zero_grad(set_to_none=True);dd,_,support=field(name,ii)
            student=forward(ii,layer,dd*active,grad=True);calls[name]['train_forward']+=1
            loss=(prob*(teacher-student)).sum(-1).mean();assert torch.isfinite(loss)
            loss.backward();calls[name]['backward']+=1
            if step==0 and name=='native_gain_lr0':
                coordinate=int(p.grad.abs().argmax());analytic=float(p.grad[coordinate]);old=float(p[coordinate].detach());values=[]
                with torch.no_grad():
                    for sign in [-1,1]:
                        p[coordinate]=old+sign*spec['gradient_epsilon']
                        delta,_,_=field(name,ii,support=support);student2=forward(ii,layer,delta*active)
                        calls[name]['gradient_forward']+=1
                        values.append(float((prob*(teacher-student2)).sum(-1).mean()))
                    p[coordinate]=old
                fd=(values[1]-values[0])/(2*spec['gradient_epsilon'])
                w.record(kind='gradient_check',task='binding',row_id=0,component=str(seed),mode=f'layer{layer}',
                         method=name,seed=seed,target_seed=target_seed,operation='singleton',analytic=analytic,finite_difference=fd)
                assert abs(fd-analytic)<.015+.08*abs(analytic),(fd,analytic)
            torch.nn.utils.clip_grad_norm_([p],spec['gradient_clip']);opts[name].step();losses[name]=float(loss.detach())
        w.record(kind='request_fit',task='binding',row_id=step,component=str(seed),method='matched_search',
                 mode=f'layer{layer}',seed=seed,target_seed=target_seed,operation='singleton',
                 fit_rows=ii,sites=sites.cpu().tolist(),losses=losses)
        if step+1 in spec['save_updates']:validate(step+1)
        elif (step+1)%32==0:w.progress('MATCHED_WRITER_FIT',seed=seed,updates=step+1,losses=losses)
    # Materialize development fields only after all choices are fixed.
    outputs={};selected_payload={};diagnostics=[]
    with torch.no_grad():
        for family,chosen in best.items():
            name=chosen['candidate'];delta,coeff,_=field(name,list(range(len(rows))),chosen['parameter'])
            outputs['selected_'+family]=delta.detach()
            selected_payload[family]=matrix(name,chosen['parameter']).cpu().numpy()
            if family.startswith('native'):
                diagnostics.append(dict(family=family,min_edited_code=float((zt[...,ti]+coeff).min()),
                                        max_changed_members=int((coeff!=0).sum(-1).max())))
            chosen.pop('parameter')
    np.savez_compressed(w.run/f'selected_writer_s{seed}_t{target_seed}.npz',**selected_payload,
                        source_indices=si.cpu().numpy(),target_indices=ti.cpu().numpy())
    write(w.run/f'WRITER_SELECTION_s{seed}.json',dict(best=best,history=history,diagnostics=diagnostics,calls=calls,
          seconds=time.perf_counter()-started,training_rows=training,validation_rows=validation,
          validation_contexts=sorted(val_contexts),training_operations=['first','second'],unfitted_operation='both',
          target_answer_labels=0,development_selection=False,
          gain_definition='One unrestricted real gain per source member multiplying its entire initial target-code row',
          source_reader_fit='Fixed readers use the original full source-fit bank; writer split controls new output fitting only'))
    torch.set_float32_matmul_precision(old_precision)
    return outputs
