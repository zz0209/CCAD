"""Transfer an effective SAE-derived source axis into actual target code edits.

The source axis is a DAS-style supervised operation on reconstructed differences.
Matching/regression and nonnegative decoder writing are separate classical steps.
Every candidate uses target-only inputs at deployment. Source training outputs
may fit a candidate; held source edits never replace executed target updates.
This does not assert novelty of DAS, ridge, distillation or constrained pursuit.
"""
from pathlib import Path
import time,json
import numpy as np

def evaluate_query(*,cfg,run,query,task,objective,source_seed,target_seed,api,rows,region,batches,bh,dh,bm,rawm,gbase,dzs,dzt,Ds,Dt,zt0,ae_t,Zs,Zt,means,natural_cal,Zsc,Ztc,raw_das_delta,checked,write,record,log,context_indices=None,context_regularization=None):
    import torch
    from .finite_native_group import fit_das,fit_native
    from .native_operation import adaptive_writable_support,batched_project_native
    from .native_group_selection import support_matching_score,group_semantic_ot,source_path_prediction
    from .selection_budget import budget_select
    start=time.perf_counter();nfit=cfg['train_rows_per_task'];ncal=cfg['calibration_rows'];N=len(rows);cal=slice(nfit,nfit+ncal);held=slice(nfit+ncal,N);ev=slice(nfit,N);device=Ds.device
    source_decoded=dzs@Ds
    U,source_fit=fit_das(api,rows[:nfit],region,source_decoded[:nfit],**cfg['axis_source'])
    assert U.shape[1]==1
    beta_s=(Ds@U).squeeze(1);q=(source_decoded@U)@U.T
    np.savez_compressed(run/(query+'_axis_source.npz'),basis=U.cpu().numpy(),signed_source_coefficients=beta_s.cpu().numpy());write(run/(query+'_axis_source_fit.json'),source_fit)
    sm=[];gs=[];sl=[]
    for off,batch,bp in batches:
        out=api.forward(batch,positions=bp,delta=q[off:off+len(batch.pairs)],gradient=True);sm.append(out['margin']);gs.append(out['gradient']);sl.append(out['log_probs'])
    sm=torch.cat(sm);gs=torch.cat(gs);sl=torch.cat(sl)
    xs=Zs-means[0];xt=Zt-means[1];yn=xs@beta_s;source_importance=xs.square().mean(0)*beta_s.square();members=torch.argsort(source_importance,descending=True,stable=True)[:cfg['axis_read_budget']]
    cos=Ds[members]@Dt.T;Sc=Zs[:,members]-Zs[:,members].mean(0);Tc=Zt-Zt.mean(0)
    cov=Sc.T@Tc/len(Sc);sv=Sc.square().mean(0);tv=Tc.square().mean(0);corr=cov/(sv.sqrt()[:,None]*tv.sqrt()[None,:]).clamp_min(1e-12)
    betas={'shared_axis':(Dt@U).squeeze(1)};fits={}
    target_ids=cos.argmax(1);gain=cov[torch.arange(len(members),device=device),target_ids]/tv[target_ids].clamp_min(1e-10)
    coeff=torch.zeros(Dt.shape[0],device=device);coeff.scatter_add_(0,target_ids,beta_s[members]*gain);betas['cosine_axis']=coeff
    if cfg.get('global_pw_run'):
        p=checked(Path(cfg['global_pw_run'])/f'{objective}_s{source_seed}_t{target_seed}_global_pw.npz');mapping=np.load(p);permutation=torch.as_tensor(mapping['permutation'],device=device);gain=torch.as_tensor(mapping['gain'],dtype=Ds.dtype,device=device)
        coeff=torch.zeros(Dt.shape[0],device=device);coeff[permutation]=beta_s*gain;betas['global_pw_axis']=coeff
    # Standard scale-aware ridge on the same capped reader support. Selection
    # of that support uses natural covariance only, never held task effects.
    score=(xt.T@yn/len(xt)).square()/xt.square().mean(0).clamp_min(1e-10)
    support=torch.argsort(score,descending=True,stable=True)[:cfg['axis_read_budget']]
    scale=xt[:,support].square().mean(0).sqrt().clamp_min(1e-4)
    for name,X,y in [('natural_ridge_axis',xt,yn),('task_ridge_axis',dzt[:nfit],source_decoded[:nfit]@U[:,0])]:
        A=X[:,support]/scale;G=A.T@A/len(A);ridge=cfg['axis_ridge'];v=torch.linalg.solve(G+ridge*torch.eye(len(G),device=device),A.T@y/len(A))/scale
        coeff=torch.zeros(Dt.shape[0],device=device);coeff[support]=v;betas[name]=coeff;fits[name]=dict(ridge=ridge,support=support.cpu().tolist(),scale=scale.cpu().tolist(),information='paired natural discovery' if name.startswith('natural') else 'original train source-axis scalar and target code differences')
    if cfg.get('axis_anchored_task_correction'):
        # Preserve the complete shared-axis reader outside the observed task
        # support. A from-scratch sparse regressor can otherwise erase useful
        # signal when natural-reference support does not cover a new task.
        X=dzt[:nfit];y=source_decoded[:nfit]@U[:,0];base=betas['shared_axis'];alpha=X@base
        gain=(alpha@y)/(alpha@alpha).clamp_min(1e-10)
        betas['calibrated_shared_axis']=base*gain
        fits['calibrated_shared_axis']=dict(gain=float(gain),information='Source train axis scalar and target train codes; scalar zero-intercept least squares, no held outcomes.')
        residual=y-alpha;variance=X.square().mean(0);covariance=X.T@residual/len(X)
        importance=covariance.square()/variance.clamp_min(1e-10)
        selected=torch.argsort(importance,descending=True,stable=True)[:cfg['axis_read_budget']]
        local_scale=variance[selected].sqrt().clamp_min(1e-4);A=X[:,selected]/local_scale
        ridge=cfg['axis_anchored_task_correction']['ridge'];G=A.T@A/len(A)
        correction=torch.linalg.solve(G+ridge*torch.eye(len(G),device=device),A.T@residual/len(A))/local_scale
        coeff=base.clone();coeff[selected]+=correction;betas['residual_task_ridge_axis']=coeff
        fits['residual_task_ridge_axis']=dict(ridge=ridge,support=selected.cpu().tolist(),scale=local_scale.cpu().tolist(),base_train_mse=float(residual.square().mean()),corrected_train_mse=float((X@coeff-y).square().mean()),information='Source train reconstruction-discrepancy scalar and target train codes. Task-selected correction support256, complete shared reader retained; no held outcomes.',scope='Classical residual ridge repair, not a new solver or a compact full reader.')
    if context_indices:
        def ot_progress(row):
            with (run/(query+'_semantic_matching_progress.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
            if row['status']=='TRANSPORT_BATCH_COMPLETE':log('SEMANTIC_TRANSPORT_BATCH',query=query,**row)
        matches=context_indices[1].match(context_indices[0],members.cpu().tolist(),candidate_count=50,regularization=context_regularization,progress=ot_progress)
        coeff=torch.zeros(Dt.shape[0],device=device)
        for j,r in enumerate(matches):
            if r['status']=='MATCHED':
                t=r['target_member'];gain=cov[j,t]/tv[t].clamp_min(1e-10);coeff[t]+=beta_s[r['source_member']]*gain
        betas['semantic_ot_axis']=coeff;fits['semantic_ot_axis']=dict(matches=matches,regularization=context_regularization,scope='Original feature-level matching recipe; positive top64 contexts, centroid50, exact same entropic objective. Retrieved atom uses signed scalar OLS calibration for the source-axis readout, a declared downstream adapter.')
    # Shared-space direction control with the same per-row predicted magnitude.
    gen=torch.Generator(device=device).manual_seed(1103);wrong=torch.randn(U.shape,device=device,generator=gen);wrong-=U@(U.T@wrong);wrong/=wrong.norm().clamp_min(1e-10)
    write(run/(query+'_axis_reader_fits.json'),fits)
    np.savez_compressed(run/(query+'_axis_readers.npz'),names=np.array(list(betas)),coefficients=torch.stack(list(betas.values())).cpu().numpy(),reader_support=support.cpu().numpy(),source_members=members.cpu().numpy())
    candidates={};diag={};reader_for={};written_counts={};support_paths={};raw_signed_distill=None
    localz=zt0[ev];decoder_np=Dt.cpu().numpy();target_np=localz.cpu().numpy();budget=cfg['axis_writer_budget'];count=N-nfit
    def native(name,desired,reader,random_support=False):
        t=time.perf_counter();parts=[];supports=[];updates=[];records=[]
        for off in range(0,count,cfg['axis_writer_batch']):
            v=desired[off:off+cfg['axis_writer_batch']];z=localz[off:off+len(v)]
            if random_support:
                rng=np.random.default_rng(913+off);chosen=[]
                for row in z.cpu().numpy():
                    active=np.flatnonzero(row>0);assert len(active)<=budget
                    available=np.flatnonzero(row==0);chosen.append(np.r_[active,rng.choice(available,budget-len(active),replace=False)])
                selection=np.array(chosen)
            else:selection=adaptive_writable_support(v.cpu().numpy(),z.cpu().numpy(),decoder_np,budget,np.ones(len(Dt),bool),device=str(device))
            ix=torch.as_tensor(selection,device=device)
            with torch.no_grad():u,realized,d=batched_project_native(v,z.gather(1,ix),Dt[ix],max_steps=cfg['axis_writer_steps'],tolerance=cfg['axis_writer_tolerance'])
            assert d['minimum_final_state']>=-1e-7
            parts.append(realized);supports.append(selection);updates.append(u.cpu().numpy());records.append(d)
        candidates[name]=torch.cat(parts);reader_for[name]=reader;diag[name]=dict(class_name='native nonnegative code write',budget=budget,wall_seconds=time.perf_counter()-t,random_support=random_support,solver=records)
        written_counts[name]=np.count_nonzero(np.concatenate(updates),axis=1)
        diag[name]['actual_changed_counts']=written_counts[name].tolist()
        support_paths[name]=np.concatenate(supports)
        np.savez_compressed(run/(query+'_'+name+'_write.npz'),row_ids=np.array([r['row_id'] for r in rows[nfit:]]),members=support_paths[name],code_increment=np.concatenate(updates))
    for name,beta in betas.items():
        desired=(dzt[ev]@beta)[:,None]*U.T;native('native_'+name,desired,name)
    shared=(dzt[ev]@betas['shared_axis'])[:,None]*U.T
    native('native_random_support',shared,'shared_axis',True)
    native('native_wrong_axis',(dzt[ev]@betas['shared_axis'])[:,None]*wrong.T,'wrong_axis')
    if cfg.get('axis_compile_nonnegative'):
        # Compile two fixed nonnegative code updates once. Positive and negative
        # axis coefficients choose different fixed supports, but each individual
        # deployment still changes at most B coordinates. All increments are
        # additions, so they are feasible for every nonnegative recipient code.
        timer=time.perf_counter();unit=torch.cat([U.T,-U.T]);zero=torch.zeros((2,len(Dt)),device=device)
        paths=adaptive_writable_support(unit.cpu().numpy(),zero.cpu().numpy(),decoder_np,budget,np.ones(len(Dt),bool),device=str(device))
        ix=torch.as_tensor(paths,device=device)
        with torch.no_grad():unit_u,unit_realized,unit_diag=batched_project_native(unit,zero.gather(1,ix),Dt[ix],max_steps=cfg['axis_writer_steps'],tolerance=cfg['axis_writer_tolerance'])
        assert bool((unit_u>=0).all());compile_seconds=time.perf_counter()-timer
        alpha=dzt[ev]@betas['shared_axis'];sign=(alpha<0).long();magnitude=alpha.abs()
        name='compiled_shared_axis';candidates[name]=magnitude[:,None]*unit_realized[sign];reader_for[name]='shared_axis'
        written_counts[name]=((magnitude[:,None]*unit_u[sign])!=0).sum(1).cpu().numpy()
        diag[name]=dict(class_name='native fixed nonnegative axis compilation',compile_seconds=compile_seconds,unit_relative_error=(unit_realized-unit).norm(dim=1).cpu().tolist(),budget=budget,distinct_union_members=len(np.unique(paths)),solver=unit_diag,scope='Two source-axis signs compiled on target decoder only; every deployment uses one fixed nonnegative support with at most256 members. Compilation uses no target task labels and no held source truth. Equal write count is not equal setup/runtime cost.')
        diag[name].update(unit_code_l1=unit_u.sum(1).cpu().tolist(),unit_active_members=(unit_u>1e-7).sum(1).cpu().tolist(),unit_cancellation_ratio=(unit_u.sum(1)/unit_realized.norm(dim=1).clamp_min(1e-12)).cpu().tolist(),opposite_edits_residual_norm=float(unit_realized.sum(0).norm()))
        # Audit the saved native update against the cached direction and time
        # the target-code-to-vector kernel separately from encoding/setup/LM.
        native_replay=torch.einsum('nb,nbd->nd',magnitude[:,None]*unit_u[sign],Dt[ix[sign]])
        diag[name]['saved_native_replay_max_abs']=float((native_replay-candidates[name]).abs().max())
        def apply_compiled():
            coefficient=dzt[ev]@betas['shared_axis'];which=(coefficient<0).long()
            return coefficient.abs()[:,None]*unit_realized[which]
        for _ in range(5):apply_compiled()
        torch.cuda.synchronize();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(50):apply_compiled()
        end.record();end.synchronize()
        diag[name]['application_kernel']=dict(rows=count,repeats=50,milliseconds_per_batch=begin.elapsed_time(end)/50,milliseconds_per_row=begin.elapsed_time(end)/50/count,scope='Warm CUDA target-code scalar read plus sign-specific cached direction scaling, excluding SAE encoding, host IO, compilation and language-model execution. Batched throughput, not single-request latency.')
        np.savez_compressed(run/(query+'_compiled_axis_operator.npz'),source_axis=U.cpu().numpy(),target_reader=betas['shared_axis'].cpu().numpy(),positive_negative_members=paths,positive_negative_code_weights=unit_u.cpu().numpy(),realized_unit_directions=unit_realized.cpu().numpy())
        np.savez_compressed(run/(query+'_'+name+'_write.npz'),row_ids=np.array([r['row_id'] for r in rows[nfit:]]),members=paths[sign.cpu().numpy()],code_increment=(magnitude[:,None]*unit_u[sign]).cpu().numpy())
        if cfg.get('axis_compile_functional'):
            from .finite_native_group import fit_compiled
            alpha_train=dzt[:nfit]@betas['shared_axis']
            fitted,fit=fit_compiled(api,rows[:nfit],region,alpha_train,unit_u,unit,sl[:nfit],decoder=Dt[ix],**cfg['axis_compile_functional'])
            realized=(fitted.unsqueeze(1)@Dt[ix]).squeeze(1)
            name='compiled_functional_axis';candidates[name]=magnitude[:,None]*realized[sign];reader_for[name]='shared_axis'
            written_counts[name]=((magnitude[:,None]*fitted[sign])!=0).sum(1).cpu().numpy()
            diag[name]=dict(class_name='native fixed nonnegative output-fitted compilation',fit=fit,unit_relative_error=(realized-unit).norm(dim=1).cpu().tolist(),unit_code_l1=fitted.sum(1).cpu().tolist(),unit_cancellation_ratio=(fitted.sum(1)/realized.norm(dim=1).clamp_min(1e-12)).cpu().tolist(),budget=budget,distinct_union_members=len(np.unique(paths)),opposite_edits_residual_norm=float(realized.sum(0).norm()),scope='Same geometric supports and target-only shared scalar, with fixed code weights fitted on source training output distributions. Two sign-specific realized directions can span rank two; the unrestricted same-information two-direction fit is a separate reference.')
            replay=torch.einsum('nb,nbd->nd',magnitude[:,None]*fitted[sign],Dt[ix[sign]])
            diag[name]['saved_native_replay_max_abs']=float((replay-candidates[name]).abs().max())
            # Timing is descriptive and cannot change fitting or selection.
            def apply_fitted():
                coefficient=dzt[ev]@betas['shared_axis'];which=(coefficient<0).long()
                return coefficient.abs()[:,None]*realized[which]
            for _ in range(5):apply_fitted()
            torch.cuda.synchronize();begin.record()
            for _ in range(50):apply_fitted()
            end.record();end.synchronize()
            diag[name]['application_kernel']=dict(rows=count,repeats=50,milliseconds_per_batch=begin.elapsed_time(end)/50,milliseconds_per_row=begin.elapsed_time(end)/50/count,scope='Warm CUDA target-code scalar read plus sign-specific fitted direction scaling; excludes encoding, compilation, fitting, IO and LM. Batched throughput, not single-request latency.')
            np.savez_compressed(run/(query+'_'+name+'_operator.npz'),source_axis=U.cpu().numpy(),target_reader=betas['shared_axis'].cpu().numpy(),positive_negative_members=paths,positive_negative_code_weights=fitted.cpu().numpy(),realized_unit_directions=realized.cpu().numpy())
            np.savez_compressed(run/(query+'_'+name+'_write.npz'),row_ids=np.array([r['row_id'] for r in rows[nfit:]]),members=paths[sign.cpu().numpy()],code_increment=(magnitude[:,None]*fitted[sign]).cpu().numpy())
            control,control_fit=fit_compiled(api,rows[:nfit],region,alpha_train,unit,unit,sl[:nfit],**cfg['axis_compile_functional'])
            raw_signed_distill=magnitude[:,None]*control[sign]
            write(run/(query+'_raw_signed_distill_fit.json'),control_fit)
            np.savez_compressed(run/(query+'_raw_signed_distill.npz'),target_reader=betas['shared_axis'].cpu().numpy(),positive_negative_directions=control.cpu().numpy())
    torch.cuda.synchronize();encoder_timer=time.perf_counter()
    with torch.no_grad():new=ae_t.encode(bh[ev]+shared);increment=new-zt0[ev];candidates['reencode_shared_axis']=increment@Dt
    torch.cuda.synchronize();encoder_seconds=time.perf_counter()-encoder_timer
    assert bool((new>=0).all());assert int((increment!=0).sum(1).max())<=budget
    reader_for['reencode_shared_axis']='shared_axis';diag['reencode_shared_axis']=dict(class_name='native actual encoder write',max_changed_members=int((increment!=0).sum(1).max()),minimum_final_state=float(new.min()),wall_seconds=encoder_seconds,timing_scope='Warm batched edited-state encoding, code subtraction and complete decoder multiplication; excludes LM and serialization.')
    written_counts['reencode_shared_axis']=(increment!=0).sum(1).cpu().numpy()
    nz=increment.nonzero();np.savez_compressed(run/(query+'_reencode_shared_axis_write.npz'),rows=nz[:,0].cpu().numpy(),members=nz[:,1].cpu().numpy(),code_increment=increment[nz[:,0],nz[:,1]].cpu().numpy(),shape=np.array(increment.shape))
    if cfg.get('axis_match_reencode_count'):
        # Refit the same greedy shared-axis path at each input's actual encoder
        # change count. Zero decoder padding permits a batched variable-length
        # refit; padded coordinates have exactly zero increment and zero state.
        timer=time.perf_counter();counts=(increment!=0).sum(1);parts=[];updates=[];records=[]
        paths=support_paths['native_shared_axis'];valid_all=[]
        for off in range(0,count,cfg['axis_writer_batch']):
            v=shared[off:off+cfg['axis_writer_batch']];ix=torch.as_tensor(paths[off:off+len(v)],device=device)
            valid=torch.arange(budget,device=device)[None,:]<counts[off:off+len(v),None]
            dec=Dt[ix]*valid[:,:,None];z=localz[off:off+len(v)].gather(1,ix)*valid
            with torch.no_grad():u,realized,d=batched_project_native(v,z,dec,max_steps=cfg['axis_writer_steps'],tolerance=cfg['axis_writer_tolerance'])
            assert bool((u[~valid]==0).all());assert d['minimum_final_state']>=-1e-7
            parts.append(realized);updates.append(u.cpu().numpy());records.append(d);valid_all.append(valid.cpu().numpy())
        name='native_reencode_count';candidates[name]=torch.cat(parts);reader_for[name]='shared_axis'
        written_counts[name]=np.count_nonzero(np.concatenate(updates),axis=1)
        diag[name]=dict(class_name='native nonnegative code write at actual re-encoding change count',wall_seconds=time.perf_counter()-timer,counts=counts.cpu().tolist(),solver=records,scope='Prefixes of the already computed256-member shared-axis pursuit path, followed by joint bounded refit. Same requested shared-axis vector and exact per-input write allowance as the encoder. Full-path search and encoder costs remain, so this is not equal computation.')
        np.savez_compressed(run/(query+'_'+name+'_write.npz'),row_ids=np.array([r['row_id'] for r in rows[nfit:]]),members=paths,valid_members=np.concatenate(valid_all),code_increment=np.concatenate(updates),member_counts=counts.cpu().numpy())
    if cfg.get('axis_fixed_mask'):
        w,fit=fit_native(api,rows[:nfit],region,dzt[:nfit],Dt,torch.zeros(Dt.shape[0],device=device),budget=budget,reference=sl[:nfit],**cfg['axis_fixed_mask'])
        candidates['fixed_native_mask']=(dzt[ev]*w)@Dt;reader_for['fixed_native_mask']='fixed_mask';diag['fixed_native_mask']=dict(class_name='native fixed donor mixture',fit=fit,members=int((w>1e-7).sum()),mass=float(w.sum()));np.savez_compressed(run/(query+'_fixed_native_mask.npz'),weights=w.cpu().numpy())
        written_counts['fixed_native_mask']=((dzt[ev]*w)!=0).sum(1).cpu().numpy()
    candidates['no_op']=torch.zeros_like(shared);reader_for['no_op']='no_op';diag['no_op']=dict(class_name='native zero update')
    written_counts['no_op']=np.zeros(count,dtype=int)
    for name,counts_written in written_counts.items():
        diag[name]['actual_changed_counts']=counts_written.tolist()
        assert int(counts_written.max())<=budget
    family=sorted(candidates);selectors=['cosine','pw_mcc','semantic_ot_group','natural_mse','task_mse','task_cosine','base_linear','anchored_base_linear','source_endpoint','source_path','finite_budget','finite_margin'];scores={s:{} for s in selectors};predictions={};stats={}
    scn=(Zsc-means[0])@beta_s;tn=Ztc-means[1];source_activity=(Zsc[:,members]*beta_s[members].abs()).sum(1).cpu().numpy();natural_np=natural_cal.cpu().numpy()
    for name,qt in candidates.items():
        error=qt-q[ev];p=dict(base_linear=bm[ev]+(gbase[ev]*qt).sum(1),anchored_base_linear=sm[ev]+(gbase[ev]*error).sum(1),source_endpoint=sm[ev]+(gs[ev]*error).sum(1),source_path=source_path_prediction(bm[ev],sm[ev],gbase[ev],gs[ev],q[ev],qt));predictions[name]=p
        for key,val in p.items():scores[key][name]=float(torch.sigmoid(val[:ncal]).mean())
        scores['task_mse'][name]=-float(error[:ncal].square().sum(1).mean());scores['task_cosine'][name]=float(torch.nn.functional.cosine_similarity(qt[:ncal],q[cal],dim=1).mean())
        reader=reader_for[name];beta=betas.get(reader,torch.zeros(Dt.shape[0],device=device));weight=beta.abs()
        if reader=='fixed_mask':weight=w;beta=w*(Dt@U).squeeze(1)
        keep=torch.argsort(weight,descending=True,stable=True)[:cfg['axis_read_budget']];sparse_weight=torch.zeros_like(weight);sparse_weight[keep]=weight[keep]
        scores['cosine'][name]=support_matching_score(cos,sparse_weight);scores['pw_mcc'][name]=support_matching_score(corr.abs(),sparse_weight)
        nm=float(((tn@beta)-scn).square().mean());scores['natural_mse'][name]=-nm
        ot,od=group_semantic_ot(source_activity,(Ztc@sparse_weight).cpu().numpy(),natural_np);scores['semantic_ot_group'][name]=-ot
        if reader=='wrong_axis':scores['natural_mse'][name]=-float(((tn@betas['shared_axis']).square()+scn.square()).mean())
        stats[name]=dict(members=float(written_counts[name].mean()),member_allowance=budget,member_scope='Mean actual nonzero code increments; the declared support allowance is separate.',l1=None,relative_realization_l2=float(error.square().sum()/q[ev].square().sum().clamp_min(1e-12)),natural_mse=nm,mean_delta_norm=float(qt.norm(dim=1).mean()),semantic_ot=od,reader=reader,writer_class=diag[name]['class_name'])
    actual_cal={};per=max(1,min(ncal,cfg['finite_budget_total_sequences']//len(family)))
    for name,qt in candidates.items():
        values=[]
        for off in range(0,ncal,cfg['batch_size']):
            batch=api.batch(rows[nfit+off:nfit+min(off+cfg['batch_size'],ncal)]);bp,_=api.positions(batch,region);out=api.forward(batch,positions=bp,delta=qt[off:off+len(batch.pairs)]);values+=out['margin'].cpu().tolist()
        actual_cal[name]=np.array(values);scores['finite_budget'][name]=float(np.mean(actual_cal[name][:per]>0));scores['finite_margin'][name]=float(np.mean(1/(1+np.exp(-np.clip(actual_cal[name][:per],-80,80)))))
    chosen={s:max(family,key=lambda n:(scores[s][n],-family.index(n))) for s in selectors}
    budget_choices=budget_select(family,np.stack([actual_cal[n] for n in family]),np.stack([predictions[n]['source_endpoint'][:ncal].cpu().numpy() for n in family]),scores,cfg.get('screening_budgets',[24,48,96]),writer_forward_ratio=cfg.get('selection_writer_forward_ratio',0.),source_rows=cfg.get('selection_source_rows'))
    from datetime import datetime,timezone
    choice=dict(query=query,task=task,objective=objective,source_seed=source_seed,target_seed=target_seed,region=region,source_members=members.cpu().tolist(),selected=chosen,scores=scores,candidate_stats=stats,budget_choices=budget_choices,calibration_rows=ncal,finite_budget_sequences_per_candidate=per,finite_budget_total_sequences=per*len(family),written_before_held_candidate_execution_utc=datetime.now(timezone.utc).isoformat(),operation_class='Source rank1 DAS-style on source SAE reconstructed differences; all deployment candidates are actual target nonnegative-code writes. Reader coefficients may be signed; target native mask and target bounded projection are different classes.')
    write(run/(query+'_selection_before_held.json'),choice);write(run/(query+'_axis_writer_diagnostics.json'),diag)
    actual={};reference={'source':q[ev],'raw':(dh-bh)[ev],'shared_axis_reader':shared,'full_target':(dzt@Dt)[ev]}
    if raw_das_delta is not None:reference['raw_das']=raw_das_delta[ev]
    if raw_signed_distill is not None:reference['raw_signed_distill']=raw_signed_distill
    for name,qt in {**candidates,**reference}.items():
        values=[]
        for off in range(ncal,count,cfg['batch_size']):
            current=rows[nfit+off:nfit+off+cfg['batch_size']];batch=api.batch(current);bp,_=api.positions(batch,region);out=api.forward(batch,positions=bp,delta=qt[off:off+len(current)]);kl=(sl[nfit+off:nfit+off+len(current)].exp()*(sl[nfit+off:nfit+off+len(current)]-out['log_probs'])).sum(1)
            for j,r in enumerate(current):
                i=nfit+off+j;values.append(float(out['margin'][j]));record(dict(query=query,task=task,objective=objective,source_seed=source_seed,target_seed=target_seed,method=name,row_id=r['row_id'],official_index=r['original_index'],official_split=r['official_split'],split=r['split'],pair_key=r['pair_key'],base_correct=bool(bm[i]<0),base_margin=float(bm[i]),source_margin=float(sm[i]),source_iia=bool(sm[i]>0),margin=float(out['margin'][j]),iia=bool(out['margin'][j]>0),log_odds_ratio=float(out['margin'][j]-bm[i]),source_kl=float(kl[j]),donor_ce=-float(out['log_probs'][j,batch.src_labels[j]]),p_src=float(out['log_probs'][j,batch.src_labels[j]]),p_base=float(out['log_probs'][j,batch.base_labels[j]]),source_delta_norm=float(q[i].norm()),target_delta_norm=float(qt[off+j].norm()),predictions={k:float(v[off+j]) for k,v in predictions[name].items()} if name in predictions else {},members=int(written_counts[name][off+j]) if name in written_counts else None))
        actual[name]=dict(iia=float(np.mean(np.array(values)>0)),margin=float(np.mean(values)))
    choice.update(held_summary=actual,selector_held_iia={s:actual[n]['iia'] for s,n in chosen.items()},source_held_iia=actual['source']['iia'],oracle_native_held_iia=max(actual[n]['iia'] for n in family),wall_seconds=time.perf_counter()-start)
    np.savez_compressed(run/(query+'_calibration.npz'),names=np.array(family),actual_margin=np.stack([actual_cal[n] for n in family]),source_margin=sm[cal].cpu().numpy(),predicted_endpoint=np.stack([predictions[n]['source_endpoint'][:ncal].cpu().numpy() for n in family]),predicted_base=np.stack([predictions[n]['base_linear'][:ncal].cpu().numpy() for n in family]),base_margin=bm[cal].cpu().numpy())
    log('AXIS_QUERY_COMPLETE',query=query,source_iia=choice['source_held_iia'],raw_das_iia=actual.get('raw_das',{}).get('iia'),native_oracle=choice['oracle_native_held_iia'],selector_iia=choice['selector_held_iia'],query_seconds=choice['wall_seconds'])
    return choice
