from pathlib import Path
import argparse
import json
import platform
import sys
import time
import traceback

import numpy as np

from run_causalgym_multisite import MultisiteWork,write
from run_shift_explanation import source_groups
from train_shift_dictionaries import site_module


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args();c=json.loads(args.config.read_text())
    work=MultisiteWork(c,args.config,['scripts/run_shift_conditional_restore.py','scripts/run_shift_explanation.py',
        'scripts/run_shift_transfer.py','scripts/train_shift_dictionaries.py','scripts/run_causalgym_multisite.py',
        'scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py'])
    hooks=[];error=None
    try:
        import torch
        import transformers
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
        work.torch=torch;work.device=torch.device(c['device']);torch.cuda.set_device(work.device);torch.cuda.reset_peak_memory_stats(work.device)
        sys.path.extend([c['dictionary_source_dir'],c['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        work.environment=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,
            transformers=transformers.__version__,numpy=np.__version__,cpu_threads=2,matmul_precision='highest')
        source=json.loads(work.checked(c['source_manifest'],'Published feature decisions','MIT').read_text())
        groups,annotations=source_groups(work.checked(c['notebook'],'Published source annotations','MIT'),source['members'])
        sites=list(source['members']);bank=np.load(work.checked(c['source_parameters'],'Published feature parameters','MIT'))
        parameters={s:{k:torch.tensor(bank[s+'__'+k],device=work.device) for k in ['encoder','encoder_bias','decoder','center']} for s in sites}
        relation=np.load(work.checked(Path(c['relation_run'])/'relation.npz','Frozen R59 native correspondence'))
        targets={};basis={};mapping={};export={}
        for site in sites:
            pids=set(groups['pronouns'].get(site,[]));wids=set(groups['associated_words'].get(site,[]))
            assert not pids&wids
            p=torch.tensor([i in pids for i in source['members'][site]],device=work.device,dtype=torch.float32)
            w=torch.tensor([i in wids for i in source['members'][site]],device=work.device,dtype=torch.float32)
            state=torch.load(work.checked(Path(c['target_directory'])/f'{site}_seed{c["target_seed"]}.pt',
                'Original target SAE checkpoint'),map_location=work.device,weights_only=True)
            target=AutoEncoderTopK(512,state['encoder.weight'].shape[0],int(state['k'])).to(work.device)
            target.load_state_dict(state);target.eval();target.requires_grad_(False);targets[site]=target
            native=torch.tensor(relation[site+'__native'],device=work.device,dtype=torch.float32)
            assert native.shape==(target.encoder.weight.shape[0],len(p)) and torch.isfinite(native).all()
            assert float(native.min())>=-1e-7 and float(native.sum(1).max())<=1.000001
            for family,pp,ww,decoder in [('source',p,w,parameters[site]['decoder']),
                    ('native',native@p,native@w,target.decoder.weight.T.detach())]:
                support=((pp+ww)>0).nonzero().flatten()
                assert len(support)>0 and float((pp+ww).max())<=1.000001
                basis[family,site]=dict(support=support,p=pp[support],w=ww[support],decoder=decoder[support])
                prefix=family+'__'+site
                export.update({prefix+'__support':support.cpu().numpy(),prefix+'__p':pp[support].cpu().numpy(),
                    prefix+'__w':ww[support].cpu().numpy(),prefix+'__decoder':decoder[support].cpu().numpy()})
                mapping[prefix]=dict(support= support.cpu().tolist(),source_member_ids=source['members'][site],
                    P_source_member_ids=sorted(pids),W_source_member_ids=sorted(wids),
                    overlap_members=int(((pp>0)&(ww>0)).sum()),overlap_min_fraction=float(torch.minimum(pp,ww).sum()),
                    maximum_P_plus_W=float((pp+ww).max()))
        np.savez_compressed(work.run/'execution_basis.npz',**export)
        probe=np.load(work.checked(Path(c['frozen_source_run'])/'probe.npz','Fixed original source profession head'))
        pw=torch.tensor(probe['weight'],device=work.device);pb=torch.tensor(probe['bias'],device=work.device)
        panel=json.loads(work.checked(c['evaluation_panel'],'Existing development documents').read_text())
        available=[r for r in panel['rows'] if r['split']==c['evaluation_split']]
        rows=[]
        for label in [0,1]:
            for gender in [0,1]:
                cell=sorted([r for r in available if r['label']==label and r['gender']==gender],key=lambda r:r['document_sha256'])
                assert len(cell)>=c['development_per_group']
                rows.extend(cell[:c['development_per_group']])
        assert len(rows)==4*c['development_per_group'] and len({r['document_sha256'] for r in rows})==len(rows)
        write(work.run/'evaluation_membership.json',dict(split=c['evaluation_split'],rows=rows))
        for name in ['config.json','model.safetensors','tokenizer.json']:
            work.checked(Path(c['model_local_dir'])/name,'Pinned Pythia70M','Apache-2.0')
        model=transformers.AutoModelForCausalLM.from_pretrained(c['model_local_dir'],local_files_only=True,
            dtype=torch.float32,attn_implementation='eager').eval().to(work.device)
        model.requires_grad_(False);model.config.use_cache=False
        operations=['none','P','W','PW','P_restoreW','none_restoreW'];families=['source','native']
        write(work.run/'restore_protocol.json',dict(operations=operations,families=families,mapping=mapping,
            parts=dict(P='pronouns',W='associated_words'),same_function='Both groups supply gender-related cues',
            restore='x_out = x_in - D(p * z_current) + D(w * (z_baseline - z_current))',
            effective_code='(1-p-w)*z_current + w*z_baseline; incoming reconstruction residual retained',
            current='One encoding of incoming hidden state per hook; current states contain all upstream operations',
            baseline='Same document and token at the same hook in the unmodified model',
            identity_check='none_restoreW must reproduce none with zero restoration update',
            endpoint='Same original source profession head applied to masked mean hidden at resid_4',
            coefficients='Token-major valid-token arrays. execution_basis contains the fixed support, P/W fractions and decoder rows. Executed coefficients times decoder reconstruct the actual requested hook update.',
            hidden_capture=c.get('capture_full_hidden',False),scope='Conditional operation in the multi-site program; restoration does not identify a unique mediator'))
        mode='none';family='source';mask=None;pooled=None;baseline={};payload={};site_stats={}

        def encode(site,x,kind):
            if kind=='source':
                s=parameters[site];z=torch.relu((x-s['center'])@s['encoder'].T+s['encoder_bias'])
            else:z=targets[site].encode(x)
            return z[...,basis[kind,site]['support']]

        def summarize(value):
            masked=value*mask[...,None]
            return (masked.sum(1)/mask.sum(1)[:,None]).detach().cpu().numpy()

        def hook(site):
            def apply(module,inputs,out):
                nonlocal pooled
                x=out[0] if isinstance(out,tuple) else out
                if mode=='none':
                    for kind in families:
                        z=encode(site,x,kind);baseline[kind,site]=z.detach().clone()
                        payload[f'{kind}__none__{site}__code']=z[mask.bool()].cpu().numpy()
                    updated=x
                else:
                    z=encode(site,x,family);base=baseline[family,site];asset=basis[family,site]
                    assert z.shape==base.shape
                    p,w,decoder=asset['p'],asset['w'],asset['decoder']
                    current_w=z*w;baseline_w=base*w
                    coeff=torch.zeros_like(z)
                    if mode in ['P','PW','P_restoreW']:coeff=coeff-z*p
                    if mode in ['W','PW']:coeff=coeff-current_w
                    correction=baseline_w-current_w if mode in ['P_restoreW','none_restoreW'] else torch.zeros_like(z)
                    coeff=coeff+correction;delta=coeff@decoder;updated=x+delta
                    prefix=f'{family}__{mode}__{site}'
                    payload[prefix+'__code']=z[mask.bool()].cpu().numpy()
                    payload[prefix+'__executed_coefficients']=coeff[mask.bool()].cpu().numpy()
                    payload[prefix+'__delta_mean']=summarize(delta)
                    if c.get('capture_full_hidden') and mode=='P_restoreW' and bool((w>0).any()):
                        payload[prefix+'__delta']=delta[mask.bool()].cpu().numpy()
                    before_w=current_w@decoder;reference_w=baseline_w@decoder
                    masked=mask[...,None]
                    correction_delta=correction@decoder
                    site_stats[site]=dict(
                        W_code_sum=(current_w*masked).sum((1,2)).cpu().numpy(),
                        baseline_W_code_sum=(baseline_w*masked).sum((1,2)).cpu().numpy(),
                        W_contribution_l2=(before_w.square()*masked).sum((1,2)).sqrt().cpu().numpy(),
                        baseline_W_contribution_l2=(reference_w.square()*masked).sum((1,2)).sqrt().cpu().numpy(),
                        W_change_l2=((before_w-reference_w).square()*masked).sum((1,2)).sqrt().cpu().numpy(),
                        W_current_dot_baseline=(before_w*reference_w*masked).sum((1,2)).cpu().numpy(),
                        delta_l2=(delta.square()*masked).sum((1,2)).sqrt().cpu().numpy(),
                        restore_delta_l2=(correction_delta.square()*masked).sum((1,2)).sqrt().cpu().numpy(),
                        delta_head_projection=(torch.tensor(summarize(delta),device=work.device)@pw.T).flatten().cpu().numpy(),
                        correction_max=(correction.abs()*masked).flatten(1).max(1).values.cpu().numpy())
                if site=='resid_4':pooled=(updated*mask[...,None]).sum(1)/mask.sum(1)[:,None]
                return (updated,*out[1:]) if isinstance(out,tuple) else updated
            return apply

        for site in sites:hooks.append(site_module(model,site).register_forward_hook(hook(site)))
        order=sorted(range(len(rows)),key=lambda i:len(rows[i]['tokens']))
        results={kind:{op:dict(logits=np.empty(len(rows),np.float32),pooled=np.empty((len(rows),512),np.float32)) for op in operations} for kind in families}
        destination=work.run/'execution_arrays';destination.mkdir()
        offset=0;identity_logit=0.;identity_hidden=0.;identity_update=0.
        while offset<len(order):
            selected=order[offset:offset+c['eval_batch_size']]
            while len(selected)>1 and len(selected)*max(len(rows[i]['tokens']) for i in selected)>c['eval_token_budget']:selected=selected[:-1]
            length=max(len(rows[i]['tokens']) for i in selected)
            ids=torch.zeros((len(selected),length),device=work.device,dtype=torch.long);mask=torch.zeros_like(ids)
            for j,i in enumerate(selected):
                tokens=rows[i]['tokens'];ids[j,:len(tokens)]=torch.tensor(tokens,device=work.device);mask[j,:len(tokens)]=1
            payload=dict(row_indices=np.array(selected),row_ids=np.array([rows[i]['row_id'] for i in selected]),
                document_sha256=np.array([rows[i]['document_sha256'] for i in selected]),
                token_offsets=np.cumsum([0]+[len(rows[i]['tokens']) for i in selected]),token_ids=ids[mask.bool()].cpu().numpy())
            baseline={};mode='none'
            with torch.no_grad():
                model.gpt_neox(ids,attention_mask=mask,use_cache=False)
                clean_pool=pooled.cpu().numpy();clean_logits=(pooled@pw.T+pb).flatten().cpu().numpy()
                work.sequence_forwards+=len(selected);work.token_forwards+=ids.numel()
                for family in families:
                    results[family]['none']['pooled'][selected]=clean_pool;results[family]['none']['logits'][selected]=clean_logits
                    for mode in operations[1:]:
                        site_stats={};model.gpt_neox(ids,attention_mask=mask,use_cache=False)
                        values=(pooled@pw.T+pb).flatten().cpu().numpy();hidden=pooled.cpu().numpy()
                        work.sequence_forwards+=len(selected);work.token_forwards+=ids.numel()
                        results[family][mode]['pooled'][selected]=hidden;results[family][mode]['logits'][selected]=values
                        if mode=='none_restoreW':
                            identity_logit=max(identity_logit,float(np.abs(values-clean_logits).max()))
                            identity_hidden=max(identity_hidden,float(np.abs(hidden-clean_pool).max()))
                            identity_update=max(identity_update,max(float(v['correction_max'].max()) for v in site_stats.values()))
                        for site,stats in site_stats.items():
                            for j,i in enumerate(selected):
                                row=rows[i]
                                work.record(kind='site_contribution',task=site,row_id=row['row_id'],component=row['document_sha256'],
                                    method=family,operation=mode,seed=c['target_seed'],target_seed=c['target_seed'],split=c['evaluation_split'],
                                    **{key:float(value[j]) for key,value in stats.items()})
                        if time.perf_counter()-work.wall_start>c['budget_seconds']:raise TimeoutError('Conditional restoration budget exceeded')
            np.savez_compressed(destination/f'batch_{offset:05d}.npz',**payload)
            offset+=len(selected)
            work.progress('CONDITIONAL_RESTORE',completed_documents=offset,total_documents=len(rows))
        for family in families:
            for mode in operations:
                values=results[family][mode]
                np.savez_compressed(work.run/f'{family}__{mode}.npz',**values)
                for i,row in enumerate(rows):
                    logit=float(values['logits'][i])
                    work.record(kind='classification',task='profession',row_id=row['row_id'],component=row['document_sha256'],
                        method=family,operation=mode,seed=c['target_seed'],target_seed=c['target_seed'],split=c['evaluation_split'],
                        label=row['label'],gender=row['gender'],prediction=int(logit>0),logit=logit,
                        effect_vs_none=logit-float(results[family]['none']['logits'][i]))
            for i,row in enumerate(rows):
                values={op:float(results[family][op]['logits'][i]) for op in operations}
                work.record(kind='conditional_effect',task='profession',row_id=row['row_id'],component=row['document_sha256'],
                    method=family,operation='P_W',seed=c['target_seed'],target_seed=c['target_seed'],split=c['evaluation_split'],
                    W_effect=values['W']-values['none'],W_effect_after_P=values['PW']-values['P'],
                    interaction=values['PW']-values['P']-values['W']+values['none'],
                    restoration_effect=values['P_restoreW']-values['P'],
                    restored_logit=values['P_restoreW'],baseline_logit=values['none'])
        write(work.run/'restore_checks.json',dict(identity_logit_max_error=identity_logit,identity_hidden_max_error=identity_hidden,
            identity_coefficient_update_max=identity_update,documents=len(rows),scientific_states=5,identity_states=1,
            forward_sequences_per_document=11,all_target_parameters_original=True))
        assert identity_logit<2e-5 and identity_hidden<2e-5 and identity_update<2e-5
        work.checks.update(same_document_restore_identity=True,source_parts_disjoint=True,native_part_capacity=True,
            frozen_model=True,frozen_original_targets=True,fixed_original_head=True)
    except Exception:
        error=traceback.format_exc();print(error,file=sys.stderr)
    for handle in hooks:handle.remove()
    return work.finish(error)


if __name__=='__main__':raise SystemExit(main())
