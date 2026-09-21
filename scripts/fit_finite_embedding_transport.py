from pathlib import Path
import argparse
import json
import sys
import time
import traceback

import numpy as np
import torch
import transformers

from ccad.intervention_transport import transport_delta, refine_columns, project_capacity
from run_causalgym_multisite import MultisiteWork, write
from run_shift_explanation import source_groups
from train_shift_dictionaries import site_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    files = ['scripts/fit_finite_embedding_transport.py', 'src/ccad/intervention_transport.py',
             'scripts/run_causalgym_multisite.py', 'scripts/run_r011s1_raw_hook_asset.py',
             'scripts/run_shift_explanation.py', 'scripts/train_shift_dictionaries.py', 'src/ccad/artifacts.py']
    w = MultisiteWork(c, args.config, files)
    handles = []
    try:
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')
        torch.use_deterministic_algorithms(True)
        torch.cuda.set_device(c['device'])
        torch.cuda.reset_peak_memory_stats()
        w.torch, w.device = torch, torch.device(c['device'])
        w.environment = dict(python=sys.executable, torch=torch.__version__, numpy=np.__version__,
                             transformers=transformers.__version__, threads=2)
        sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        w.checked(Path(c['dictionary_source_dir'])/'dictionary_learning/trainers/top_k.py', 'Pinned TopK', 'MIT')
        source = json.loads(w.checked(c['source_manifest']).read_text())
        groups, _ = source_groups(w.checked(c['notebook']), source['members'])
        sites = list(source['members'])
        bank = np.load(w.checked(c['source_parameters']))
        sp = {s: {k: torch.tensor(bank[s+'__'+k], device=w.device)
                  for k in ['center', 'encoder', 'encoder_bias', 'decoder']} for s in sites}
        initial, targets = {}, {}
        for s in sites:
            path = w.checked(Path(c['target_directory'])/f'{s}_seed{c["target_seed"]}.pt')
            sd = torch.load(path, map_location='cpu', weights_only=True)
            target = AutoEncoderTopK(512, len(sd['encoder.weight']), int(sd['k'])).to(w.device)
            target.load_state_dict(sd)
            target.requires_grad_(False)
            initial[s], targets[s] = sd, target
        for name in ['config.json', 'tokenizer.json', 'model.safetensors']:
            w.checked(Path(c['model_local_dir'])/name)
        model = transformers.AutoModelForCausalLM.from_pretrained(c['model_local_dir'], local_files_only=True,
                dtype=torch.float32, attn_implementation='eager').to(w.device).eval()
        model.requires_grad_(False)
        model.config.use_cache = False
        probe = np.load(w.checked(Path(c['frozen_source_run'])/'probe.npz'))
        pw = torch.tensor(probe['weight'], device=w.device)
        pb = torch.tensor(probe['bias'], device=w.device)
        panel = json.loads(w.checked(Path(c['frozen_source_run'])/'panel.json').read_text())['rows']
        membership = json.loads(w.checked(c['source_context_membership']).read_text())
        fit_set, cal_set = set(membership['fit_documents']), set(membership['calibration_documents'])
        fit = sorted([r for r in panel if r['document_sha256'] in fit_set], key=lambda r: r['document_sha256'])[:c['fit_documents']]
        calibration = sorted([r for r in panel if r['document_sha256'] in cal_set], key=lambda r: r['document_sha256'])[:c['calibration_documents']]
        evaluate_rows = json.loads(w.checked(c['evaluation_panel']).read_text())['rows']
        if c.get('evaluation_per_cell'):
            evaluate_rows = [r for profession in sorted({r['profession'] for r in evaluate_rows}) for gender in [0, 1]
                             for r in sorted([r for r in evaluate_rows if r['profession']==profession and r['gender']==gender],
                                             key=lambda r:r['document_sha256'])[:c['evaluation_per_cell']]]
        assert len(fit)==c['fit_documents'] and len(calibration)==c['calibration_documents'] and evaluate_rows
        assert not ({r['document_sha256'] for r in evaluate_rows} & (fit_set|cal_set))
        requests = json.loads(w.checked(c['request_panel']).read_text())
        query_names = c.get('evaluation_queries', requests['queries'])
        write(w.run/'membership.json', dict(fit=fit, calibration=calibration, evaluation=evaluate_rows,
              queries=query_names, members=source['members'], source_fit_documents=len(fit), labels_used=False))
        q = {s: torch.ones(len(source['members'][s]), device=w.device) for s in sites}
        group_members = {s:torch.tensor([[member in groups[g].get(s, []) for g in groups]
                         for member in source['members'][s]],device=w.device,dtype=torch.float32) for s in sites}
        for s in sites:
            assert bool((group_members[s].sum(-1)==1).all())
        mode, arm = 'clean', 'finite'
        ids, mask, pooled = None, None, None
        p = len(q['embed'])
        allowance = 2*p
        embedding = model.gpt_neox.embed_in.weight.detach()
        zs_all = torch.relu((embedding-sp['embed']['center'])@sp['embed']['encoder'].T+sp['embed']['encoder_bias'])
        active_tokens = (zs_all.sum(-1)>0).nonzero().flatten()
        token_map = torch.full((len(embedding),), len(active_tokens), device=w.device, dtype=torch.long)
        token_map[active_tokens] = torch.arange(len(active_tokens), device=w.device)
        supports, coefficients, capacities = [], [], []
        with torch.no_grad():
            for offset in range(0,len(active_tokens),128):
                x = embedding[active_tokens[offset:offset+128]]
                target = targets['embed']
                zs = torch.relu((x-sp['embed']['center'])@sp['embed']['encoder'].T+sp['embed']['encoder_bias'])
                candidate = -(target.encoder.weight@sp['embed']['decoder'].T)[None]*zs[:,None,:]
                support = (candidate.abs().sum(-1)*target.decoder.weight.norm(dim=0)).topk(allowance,dim=-1).indices
                _,_,cols = transport_delta(x,target,sp['embed'],q['embed'],target.encoder.weight,allowance,False)
                cols = refine_columns(x,target,sp['embed'],cols,allowance,c['inverse_steps'],accelerate=True)
                supports.append(support)
                coefficients.append(cols.gather(1,support[...,None].expand(-1,-1,p)))
                capacities.append(target.encode(x).gather(1,support))
                w.progress('VOCABULARY_EXECUTION', completed=min(offset+128,len(active_tokens)), total=len(active_tokens))
        support = torch.cat(supports)
        base = torch.cat(coefficients)
        cap = torch.cat(capacities)
        scale = zs_all[active_tokens]
        decoder = targets['embed'].decoder.weight.T[support].detach()
        # 空白索引对应源贡献为零的词元，始终保持零更新。
        base = torch.cat([base,torch.zeros((1,allowance,p),device=w.device)])
        cap = torch.cat([cap,torch.zeros((1,allowance),device=w.device)])
        scale = torch.cat([scale,torch.zeros((1,p),device=w.device)])
        decoder = torch.cat([decoder,torch.zeros((1,allowance,512),device=w.device)])
        residual = torch.nn.Parameter(torch.zeros_like(base))
        gain = torch.nn.Parameter(torch.ones((len(base),p),device=w.device))
        full_base=torch.zeros((len(base),targets['embed'].decoder.weight.shape[1],p),device=w.device)
        full_base[:-1].scatter_(1,support[...,None].expand(-1,-1,p),base[:-1])
        full_capacity=targets['embed'].encode(embedding[active_tokens]).detach()
        full_capacity=torch.cat([full_capacity,torch.zeros_like(full_capacity[-1:])])
        sparse_coefficients=torch.nn.Parameter(full_base.clone())
        initial_support_mask=torch.zeros(full_base.shape[:2],device=w.device)
        initial_support_mask[:-1].scatter_(1,support,1.)
        active_support_mask=initial_support_mask.clone()
        state = dict(active_tokens=active_tokens.cpu(), support=support.cpu(), coefficients=base.cpu(), capacity=cap.cpu())
        torch.save(state,w.run/'initial_execution.pt')
        del supports,coefficients,capacities,zs_all
        transferred = {}
        frozen_coefficients={}
        for name,path in c.get('frozen_coefficients',{}).items():
            checkpoint=torch.load(w.checked(path),map_location=w.device,weights_only=True)
            columns=checkpoint['coefficients']
            assert columns.shape==full_base.shape
            assert bool(((-columns).clamp_min(0).sum(-1)<=full_capacity+1e-5).all())
            assert bool((columns*(scale[:,None,:]==0)==0).all())
            assert int((columns.abs().sum(-1)>0).sum(-1).max())<=allowance
            frozen_coefficients[name]=columns
        for name,path in c.get('transfer_fields',{}).items():
            data=torch.load(w.checked(path),map_location=w.device,weights_only=True)
            assert torch.equal(data['active_tokens'],active_tokens)
            field=data['field']
            assert field.shape==(len(active_tokens),512,p)
            with torch.no_grad():
                d=decoder[:-1]
                gram=d@d.transpose(-1,-2)
                rhs=d@field
                lipschitz=gram.abs().sum(-1).amax(-1).clamp_min(1e-8)
                value=base[:-1].clone();current=value;momentum=1.
                for _ in range(c['inverse_steps']):
                    updated=project_capacity(current-(gram@current-rhs)/lipschitz[:,None,None],cap[:-1])
                    next_momentum=(1+(1+4*momentum**2)**.5)/2
                    current=updated+(momentum-1)/next_momentum*(updated-value)
                    value=updated;momentum=next_momentum
                value=value*(scale[:-1,None,:]>0)
                transferred[name]=torch.cat([value,torch.zeros_like(base[-1:])])

        def current_columns():
            proposal = base*gain[:,None,:] if arm=='gain' else base+residual*scale[:,None,:]
            return project_capacity(proposal,cap)*(scale[:,None,:]>0)

        def hook(site):
            def apply(module,inputs,output):
                nonlocal pooled
                x = output[0] if isinstance(output,tuple) else output
                if mode!='clean':
                    ss = sp[site]
                    if mode=='source' or (mode=='source_embedding' and site=='embed'):
                        zs = torch.relu((x-ss['center'])@ss['encoder'].T+ss['encoder_bias'])
                        x = x-(zs*q[site])@ss['decoder']
                    elif mode=='readout':
                        target=targets[site]
                        rec=target.decode(target.encode(x))
                        zs=torch.relu((rec-ss['center'])@ss['encoder'].T+ss['encoder_bias'])
                        x=x-(zs*q[site])@ss['decoder']
                    elif site=='embed' and (mode in ['geometric','learned'] or mode in transferred or mode in frozen_coefficients):
                        ix=token_map[ids]
                        if (mode=='learned' and arm in ['sparse','fixed_direct','regrow']) or mode in frozen_coefficients:
                            unique,inverse=ix.flatten().unique(return_inverse=True)
                            coefficients=frozen_coefficients[mode] if mode in frozen_coefficients else sparse_coefficients
                            delta=((coefficients[unique]@q[site])@targets[site].decoder.weight.T)[inverse].reshape_as(x)
                        else:
                            columns=(transferred[mode] if mode in transferred else base if mode=='geometric' else current_columns())[ix]
                            delta=torch.einsum('btk,btkd->btd',columns@q[site],decoder[ix])
                        x=x+delta
                    else:
                        target=targets[site]
                        delta,_,_=transport_delta(x,target,ss,q[site],target.encoder.weight,2*len(q[site]),True)
                        x=x+delta
                if site=='resid_4':
                    pooled=(x*mask[...,None]).sum(1)/mask.sum(1)[:,None]
                return (x,*output[1:]) if isinstance(output,tuple) else x
            return apply
        for s in sites:
            handles.append(site_module(model,s).register_forward_hook(hook(s)))

        def forward(rows):
            nonlocal ids,mask
            length=max(len(r['tokens']) for r in rows)
            ids=torch.zeros((len(rows),length),device=w.device,dtype=torch.long)
            mask=torch.zeros_like(ids)
            for j,row in enumerate(rows):
                ids[j,:len(row['tokens'])]=torch.tensor(row['tokens'],device=w.device)
                mask[j,:len(row['tokens'])]=1
            model.gpt_neox(ids,attention_mask=mask,use_cache=False)
            w.sequence_forwards+=len(rows)
            w.token_forwards+=int(mask.sum())
            return pooled

        def set_query(name):
            for s in sites:
                if name in requests.get('member_queries',{}):
                    q[s]=torch.tensor(requests['member_queries'][name][s],device=w.device,dtype=torch.float32)
                elif name in requests.get('dose_queries',{}):
                    q[s]=group_members[s]@torch.tensor([requests['dose_queries'][name][g] for g in groups],device=w.device)
                else:
                    weights=torch.tensor([name=='full' or g in name.split('+') for g in groups],device=w.device,dtype=torch.float32)
                    q[s]=group_members[s]@weights

        @torch.no_grad()
        def evaluate(label,execution):
            nonlocal mode
            mode=execution
            for name in (['full'] if mode=='clean' else query_names):
                set_query(name)
                values=[]
                for offset in range(0,len(evaluate_rows),c['eval_batch_size']):
                    values.append(forward(evaluate_rows[offset:offset+c['eval_batch_size']]).cpu().numpy())
                values=np.concatenate(values)
                np.save(w.run/f'{label}__{name}__pooled.npy',values)
                response=values@pw.cpu().numpy().T+pb.cpu().numpy()
                for row,value in zip(evaluate_rows,response.ravel()):
                    w.record(kind='response',task='human',row_id=row['document_sha256'],component=row['document_sha256'],method=label,
                             operation=name,seed=c['target_seed'],logit=float(value))
                w.progress('EVALUATION',method=label,request=name,documents=len(evaluate_rows))
        evaluate('clean','clean')
        evaluate('source','source')
        for method in c['baseline_methods']:
            if method=='program':
                for s in c['adapt_sites']:
                    sd=torch.load(w.checked(Path(c['reference_program_directory'])/f'{s}_seed{c["target_seed"]}.pt'),
                                  map_location=w.device,weights_only=True)
                    targets[s].load_state_dict(sd)
                evaluate(method,method)
                for s in sites:targets[s].load_state_dict(initial[s])
            else:
                evaluate(method,method)
        for method in transferred:evaluate(method,method)
        for method in frozen_coefficients:evaluate(method,method)
        normalization=[]
        with torch.no_grad():
            for offset in range(0,len(calibration),c['batch_sequences']):
                rows=calibration[offset:offset+c['batch_sequences']]
                mode='clean'; clean=forward(rows)
                for name in groups:
                    set_query(name);mode='source';value=forward(rows)-clean
                    normalization.append([float(value.square().mean()),float((value@pw.T).square().mean())])
        norm=np.maximum(np.mean(normalization,axis=0),1e-8)
        write(w.run/'normalization.json',dict(pooled=float(norm[0]),head=float(norm[1]),source_calibration_only=True))
        for arm in c['arms']:
            residual.data.zero_();gain.data.fill_(1)
            sparse_coefficients.data.copy_(full_base)
            active_support_mask.copy_(initial_support_mask)
            parameter=gain if arm=='gain' else sparse_coefficients if arm in ['sparse','fixed_direct','regrow'] else residual
            optimizer=torch.optim.Adam([parameter],lr=c['gain_lr'] if arm=='gain' else c.get('sparse_lr',.003) if arm in ['sparse','fixed_direct','regrow'] else c['coefficient_lr'])
            rng=np.random.default_rng(c['training_seed'])
            for step in range(c['steps']):
                rows=[fit[i] for i in rng.choice(len(fit),c['batch_sequences'],replace=False)]
                if step%2==0:
                    weights=torch.tensor(rng.uniform(0,1,len(groups)),device=w.device,dtype=torch.float32)
                    for s in sites:q[s]=group_members[s]@weights
                else:
                    for s in sites:q[s]=torch.tensor(rng.integers(0,2,len(q[s])),device=w.device,dtype=torch.float32)
                mode='source'
                with torch.no_grad(): teacher=forward(rows)
                mode='learned'; prediction=forward(rows)
                error=prediction-teacher
                state_loss=error.square().mean()/norm[0]
                head_loss=(error@pw.T).square().mean()/norm[1]
                loss=(1-c['source_response_weight'])*state_loss+c['source_response_weight']*head_loss
                if not torch.isfinite(loss):raise ValueError('Nonfinite functional loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_([parameter],1.)
                optimizer.step()
                if arm=='gain':gain.data.clamp_(min=0)
                if arm in ['sparse','fixed_direct','regrow']:
                    # 行约束独立，投影收益确定固定预算下的共同成员。
                    with torch.no_grad():
                        proposal=sparse_coefficients.detach()*(scale[:,None,:]>0)
                        feasible=project_capacity(proposal,full_capacity)
                        improvement=(2*proposal*feasible-feasible.square()).sum(-1)
                        if arm=='regrow':
                            retained=active_support_mask
                        else:
                            chosen=improvement.topk(allowance,dim=-1).indices if arm=='sparse' else torch.cat([support,torch.zeros_like(support[:1])])
                            retained=torch.zeros_like(improvement).scatter(1,chosen,1.)
                        sparse_coefficients.copy_(feasible*retained[...,None])
                        if arm=='regrow' and (step+1)%32==0 and step+1<=192:
                            gradient=parameter.grad*(scale[:,None,:]>0)
                            present=gradient.abs().sum((-1,-2))>0
                            strength=sparse_coefficients.square().sum(-1).masked_fill(active_support_mask==0,-float('inf'))
                            kept=strength.topk(allowance-2,dim=-1).indices
                            next_mask=torch.zeros_like(active_support_mask).scatter(1,kept,1.)
                            direction=project_capacity(-c.get('sparse_lr',.003)*gradient,full_capacity)
                            score=direction.square().sum(-1).masked_fill(next_mask>0,-float('inf'))
                            grown=score.topk(2,dim=-1).indices
                            next_mask.scatter_(1,grown,1.)
                            next_mask=torch.where(present[:,None],next_mask,active_support_mask)
                            retained_old=next_mask*active_support_mask
                            sparse_coefficients.mul_(retained_old[...,None])
                            for name in ['exp_avg','exp_avg_sq']:
                                optimizer.state[parameter][name].mul_(retained_old[...,None])
                            active_support_mask.copy_(next_mask)
                if (step+1)%c['log_every']==0 or step+1==c['steps']:
                    w.record(kind='training',task='human',row_id=step+1,component=arm,method=arm,loss=float(loss.detach()),
                             pooled_loss=float(state_loss.detach()),head_loss=float(head_loss.detach()))
                    w.progress('TRAINING',method=arm,step=step+1,total_steps=c['steps'],loss=float(loss.detach()),
                               peak_cuda_bytes=torch.cuda.max_memory_allocated())
                if step+1 in c['checkpoints']:
                    checkpoint=dict(residual=residual.detach().cpu(),gain=gain.detach().cpu(),arm=arm,step=step+1)
                    if arm in ['sparse','fixed_direct','regrow']:checkpoint['coefficients']=sparse_coefficients.detach().cpu()
                    torch.save(checkpoint,w.run/f'{arm}_{step+1}.pt')
                    evaluate(f'{arm}_{step+1}','learned')
                if time.perf_counter()-w.wall_start>c['budget_seconds']:raise TimeoutError('Finite execution budget exceeded')
            with torch.no_grad():
                columns=sparse_coefficients if arm in ['sparse','fixed_direct','regrow'] else current_columns()
                w.checks['capacity_'+arm]=bool(((-columns).clamp_min(0).sum(-1)<=(full_capacity if arm in ['sparse','fixed_direct','regrow'] else cap)+1e-5).all())
                w.checks['source_zero_'+arm]=bool((columns*(scale[:,None,:]==0)==0).all())
                w.checks['support_'+arm]=int((columns.abs().sum(-1)>0).sum(-1).max())<=allowance
                if arm in ['sparse','regrow']:
                    write(w.run/'support_changes.json',dict(changed_tokens=int(((columns.abs().sum(-1)>0)!=(full_base.abs().sum(-1)>0)).any(-1).sum()),
                          active_rows=int((columns.abs().sum((-1,-2))>0).sum()),max_members=int((columns.abs().sum(-1)>0).sum(-1).max())))
        w.checks['dictionaries_unchanged']=all(torch.equal(targets[s].state_dict()[k].cpu(),v)
                                              for s in sites for k,v in initial[s].items())
        write(w.run/'execution_statistics.json',dict(active_tokens=len(active_tokens),columns=p,member_allowance=allowance,
              finite_parameters=residual.numel(),gain_parameters=gain.numel(),all_sites=sites,teacher_members=sum(map(len,source['members'].values())),
              optimizer_steps=c['steps'],fitted_dictionary=c['target_seed'],coefficient_table_context_independent=True))
    except Exception:
        w.finish(traceback.format_exc())
        raise
    finally:
        for handle in handles:handle.remove()
    return w.finish(None)


if __name__=='__main__':
    raise SystemExit(main())
