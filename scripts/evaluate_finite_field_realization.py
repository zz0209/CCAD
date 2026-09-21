from pathlib import Path
import argparse
import json
import sys
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
    w = MultisiteWork(c, args.config, ['scripts/evaluate_finite_field_realization.py',
        'src/ccad/intervention_transport.py', 'scripts/run_causalgym_multisite.py',
        'scripts/run_shift_explanation.py', 'scripts/train_shift_dictionaries.py', 'src/ccad/artifacts.py'])
    handles = []
    try:
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')
        torch.use_deterministic_algorithms(True)
        w.torch, w.device = torch, torch.device(c['device'])
        torch.cuda.set_device(w.device)
        torch.cuda.reset_peak_memory_stats()
        w.environment = dict(python=sys.executable, torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__)
        sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        source = json.loads(w.checked(c['source_manifest']).read_text())
        groups, _ = source_groups(w.checked(c['notebook']), source['members'])
        sites = list(source['members'])
        bank = np.load(w.checked(c['source_parameters']))
        sp = {s:{k:torch.tensor(bank[s+'__'+k],device=w.device) for k in ['center','encoder','encoder_bias','decoder']} for s in sites}
        reference = Path(c['reference_run'])
        membership = json.loads(w.checked(reference/'membership.json').read_text())
        rows = membership['evaluation'][:c.get('evaluation_documents',len(membership['evaluation']))]
        queries = c.get('evaluation_queries',membership['queries'])
        requests = json.loads(w.checked(c['request_panel']).read_text())
        write(w.run/'membership.json',dict(evaluation=rows,queries=queries,source_members=source['members']))
        for name in ['config.json','tokenizer.json','model.safetensors']:
            w.checked(Path(c['model_local_dir'])/name)
        model = transformers.AutoModelForCausalLM.from_pretrained(c['model_local_dir'],local_files_only=True,
            dtype=torch.float32,attn_implementation='eager').to(w.device).eval()
        model.requires_grad_(False)
        data = torch.load(w.checked(c['field']),map_location=w.device,weights_only=True)
        active, field = data['active_tokens'], data['field']
        embedding = model.gpt_neox.embed_in.weight.detach()
        token_map = torch.full((len(embedding),),len(active),device=w.device,dtype=torch.long)
        token_map[active] = torch.arange(len(active),device=w.device)
        field = torch.cat([field,torch.zeros_like(field[:1])])
        q = {s:torch.ones(len(source['members'][s]),device=w.device) for s in sites}
        gm = {s:torch.tensor([[m in groups[g].get(s,[]) for g in groups] for m in source['members'][s]],device=w.device,dtype=torch.float32) for s in sites}
        targets, mode, ids, mask, pooled = {}, 'physical', None, None, None
        projected, decoder = None, None

        def hook(site):
            def apply(module, inputs, output):
                nonlocal pooled
                x = output[0] if isinstance(output,tuple) else output
                if site=='embed':
                    ix = token_map[ids]
                    delta = field[ix]@q[site] if mode=='physical' else torch.einsum('btk,btkd->btd',projected[ix]@q[site],decoder[ix])
                    x = x+delta
                else:
                    delta, _, _ = transport_delta(x,targets[site],sp[site],q[site],targets[site].encoder.weight,2*len(q[site]),True)
                    x = x+delta
                if site=='resid_4':pooled=(x*mask[...,None]).sum(1)/mask.sum(1)[:,None]
                return (x,*output[1:]) if isinstance(output,tuple) else x
            return apply

        for s in sites:handles.append(site_module(model,s).register_forward_hook(hook(s)))
        checks = {}
        with torch.no_grad():
            for seed in c['seeds']:
                initial = {}
                for s in sites:
                    sd=torch.load(w.checked(Path(c['target_directory'])/f'{s}_seed{seed}.pt'),map_location='cpu',weights_only=True)
                    target=AutoEncoderTopK(512,len(sd['encoder.weight']),int(sd['k'])).to(w.device)
                    target.load_state_dict(sd);target.requires_grad_(False)
                    targets[s], initial[s] = target, sd
                target=targets['embed'];x=embedding[active]
                zs=torch.relu((x-sp['embed']['center'])@sp['embed']['encoder'].T+sp['embed']['encoder_bias'])
                candidate=-(target.encoder.weight@sp['embed']['decoder'].T)[None]*zs[:,None,:]
                support=(candidate.abs().sum(-1)*target.decoder.weight.norm(dim=0)).topk(20,dim=-1).indices
                _,_,cols=transport_delta(x,target,sp['embed'],q['embed'],target.encoder.weight,20,False)
                cols=refine_columns(x,target,sp['embed'],cols,20,c['inverse_steps'],accelerate=True)
                value=cols.gather(1,support[...,None].expand(-1,-1,10))
                cap=target.encode(x).gather(1,support)
                d=target.decoder.weight.T[support]
                gram=d@d.transpose(-1,-2);rhs=d@field[:-1]
                lipschitz=gram.abs().sum(-1).amax(-1).clamp_min(1e-8)
                current=value;momentum=1.
                for _ in range(c['inverse_steps']):
                    updated=project_capacity(current-(gram@current-rhs)/lipschitz[:,None,None],cap)
                    next_momentum=(1+(1+4*momentum**2)**.5)/2
                    current=updated+(momentum-1)/next_momentum*(updated-value)
                    value=updated;momentum=next_momentum
                value=value*(zs[:,None,:]>0)
                projected=torch.cat([value,torch.zeros_like(value[:1])])
                decoder=torch.cat([d,torch.zeros_like(d[:1])])
                for mode in c.get('execution_modes',['physical','native']):
                    if mode=='field_native':
                        candidate=target.encoder.weight[None]@field[:-1]
                        chosen=(candidate.abs().sum(-1)*target.decoder.weight.norm(dim=0)).topk(20,dim=-1).indices
                        d=target.decoder.weight.T[chosen]
                        cap=target.encode(x).gather(1,chosen)
                        value=project_capacity(candidate.gather(1,chosen[...,None].expand(-1,-1,10)),cap)
                        gram=d@d.transpose(-1,-2);rhs=d@field[:-1]
                        lipschitz=gram.abs().sum(-1).amax(-1).clamp_min(1e-8)
                        current=value;momentum=1.
                        for _ in range(c['inverse_steps']):
                            updated=project_capacity(current-(gram@current-rhs)/lipschitz[:,None,None],cap)
                            next_momentum=(1+(1+4*momentum**2)**.5)/2
                            current=updated+(momentum-1)/next_momentum*(updated-value)
                            value=updated;momentum=next_momentum
                        value=value*(zs[:,None,:]>0)
                        projected=torch.cat([value,torch.zeros_like(value[:1])])
                        decoder=torch.cat([d,torch.zeros_like(d[:1])])
                        w.checks[f'field_capacity_t{seed}']=bool(((-value).clamp_min(0).sum(-1)<=cap+1e-5).all())
                    for query in queries:
                        for s in sites:
                            if query in requests.get('member_queries',{}):q[s]=torch.tensor(requests['member_queries'][query][s],device=w.device,dtype=torch.float32)
                            elif query in requests.get('dose_queries',{}):q[s]=gm[s]@torch.tensor([requests['dose_queries'][query][g] for g in groups],device=w.device)
                            else:q[s]=gm[s]@torch.tensor([query=='full' or g in query.split('+') for g in groups],device=w.device,dtype=torch.float32)
                        values=[]
                        for offset in range(0,len(rows),2):
                            batch=rows[offset:offset+2];length=max(len(r['tokens']) for r in batch)
                            ids=torch.zeros((len(batch),length),device=w.device,dtype=torch.long);mask=torch.zeros_like(ids)
                            for i,row in enumerate(batch):
                                ids[i,:len(row['tokens'])]=torch.tensor(row['tokens'],device=w.device);mask[i,:len(row['tokens'])]=1
                            model.gpt_neox(ids,attention_mask=mask,use_cache=False)
                            values.append(pooled.cpu().numpy());w.sequence_forwards+=len(batch);w.token_forwards+=int(mask.sum())
                        values=np.concatenate(values)
                        np.save(w.run/f'{mode}_t{seed}__{query}__pooled.npy',values)
                        for row, value in zip(rows,values):
                            w.record(kind='pooled_norm',task='human',row_id=row['document_sha256'],
                                component=row['document_sha256'],method=mode,seed=seed,operation=query,
                                pooled_norm=float(np.linalg.norm(value)))
                        if seed==3 and mode=='native':
                            previous=np.load(w.checked(reference/f'transfer_sparse__{query}__pooled.npy'))[:len(rows)]
                            checks[query]=float(np.max(np.abs(previous-values)))
                            assert np.allclose(previous,values,atol=1e-5,rtol=1e-5),query
                        w.progress('REALIZATION',target=seed,method=mode,request=query,documents=len(rows))
                w.checks[f'dictionaries_unchanged_t{seed}']=all(torch.equal(targets[s].state_dict()[k].cpu(),v) for s in sites for k,v in initial[s].items())
        write(w.run/'native_replay.json',dict(max_absolute_difference=checks))
    except Exception:
        w.finish(traceback.format_exc())
        raise
    finally:
        for handle in handles:handle.remove()
    return w.finish(None)


if __name__=='__main__':
    raise SystemExit(main())
