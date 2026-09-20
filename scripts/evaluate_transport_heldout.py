"""Test grammar-trained transport on unused published human members at resid_4.

The original SHIFT classifier reads this same hook, so intervention effects on
its pooled logits can be computed exactly without a second transformer pass.
All seven nonempty subsets of the three published members are retained.
"""
from pathlib import Path
import argparse, json, sys, traceback
import numpy as np
from run_causalgym_multisite import MultisiteWork, write
from train_shift_dictionaries import site_module
from run_shift_explanation import source_groups
from ccad.intervention_transport import transport_delta,refine_columns


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    args=p.parse_args();c=json.loads(args.config.read_text())
    w=MultisiteWork(c,args.config,['scripts/evaluate_transport_heldout.py',
        'scripts/train_shift_dictionaries.py','scripts/run_shift_explanation.py',
        'scripts/run_causalgym_multisite.py','scripts/run_r011s1_raw_hook_asset.py',
        'src/ccad/artifacts.py','src/ccad/intervention_transport.py'])
    error=None;handle=None
    try:
        import torch,transformers
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest');torch.cuda.set_device(c['device'])
        torch.cuda.reset_peak_memory_stats();w.torch=torch;w.device=torch.device(c['device'])
        w.environment=dict(python=sys.executable,torch=torch.__version__,numpy=np.__version__,transformers=transformers.__version__)
        sys.path.extend([c['dictionary_source_dir'],c['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        w.checked(Path(c['dictionary_source_dir'])/'dictionary_learning/trainers/top_k.py','Pinned TopK','MIT')
        source=json.loads(w.checked(c['source_manifest']).read_text())
        _,annotations=source_groups(w.checked(c['notebook']),source['members'])
        sb=np.load(w.checked(c['source_parameters']))
        sp={k:torch.tensor(sb['resid_4__'+k],device=w.device) for k in ['encoder','encoder_bias','decoder','center']}
        grammar=np.load(w.checked(c['grammar_source']))
        # Training directions and human directions are different published members.
        u,sv,_=np.linalg.svd(grammar['decoder'].T,full_matrices=False)
        relative=np.sum((sp['decoder'].cpu().numpy()@u)**2,axis=1)/np.sum(sb['resid_4__decoder']**2,axis=1)
        initial=torch.load(w.checked(Path(c['target_directory'])/f'resid_4_seed{c["target_seed"]}.pt'),map_location=w.device,weights_only=True)
        target=AutoEncoderTopK(512,len(initial['encoder.weight']),int(initial['k'])).to(w.device)
        target.load_state_dict(initial);target.requires_grad_(False)
        matrices={'initial_active':initial['encoder.weight'],'initial_open':initial['encoder.weight']}
        for name,path in c['write_checkpoints'].items(): matrices[name]=torch.load(w.checked(path),map_location=w.device,weights_only=True)
        d=target.decoder.weight
        matrices['dual_frame_open']=torch.linalg.solve(d@d.T+1e-6*torch.eye(512,device=w.device),d).T
        panel=json.loads(w.checked(Path(c['frozen_source_run'])/'panel.json').read_text())
        rows=[]
        for y in [0,1]:
            for g in [0,1]:
                rows+=sorted([r for r in panel['rows'] if r['split']=='dev' and r['label']==y and r['gender']==g],key=lambda r:r['document_sha256'])[:c['development_per_group']]
        if c.get('evaluation_panel'):
            import hashlib
            path=w.checked(c['evaluation_panel'],'Frozen fresh documents excluded from the original source panel')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==c['evaluation_panel_sha256']
            rows=json.loads(path.read_text())['rows']
            assert not {r['document_sha256'] for r in rows}&{r['document_sha256'] for r in panel['rows']}
        queries={f'members_{mask:03b}':[(mask>>j)&1 for j in range(3)] for mask in range(1,8)}
        methods=['source','raw_reconstruction',*matrices]
        if c.get('refine_steps'):methods.append('refined_initial_open')
        outputs={name:np.zeros((len(queries),len(rows)),np.float32) for name in methods}
        hidden={name:np.zeros((len(queries),len(rows)),np.float32) for name in methods if name!='source'}
        energies=np.zeros((len(queries),len(rows)),np.float32);clean=np.zeros(len(rows),np.float32)
        original=np.load(w.checked(Path(c['frozen_source_run'])/'probe.npz'))
        pw=torch.tensor(original['weight'],device=w.device);pb=torch.tensor(original['bias'],device=w.device)
        for name in ['config.json','model.safetensors']:w.checked(Path(c['model_local_dir'])/name)
        model=transformers.AutoModelForCausalLM.from_pretrained(c['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to(w.device)
        model.requires_grad_(False);model.config.use_cache=False
        captured=None
        def hook(module,inputs,out):
            nonlocal captured
            captured=(out[0] if isinstance(out,tuple) else out).detach()
        handle=site_module(model,'resid_4').register_forward_hook(hook)
        order=sorted(range(len(rows)),key=lambda i:len(rows[i]['tokens']))
        with torch.no_grad():
            for off in range(0,len(order),c['eval_batch_size']):
                ix=order[off:off+c['eval_batch_size']];length=max(len(rows[i]['tokens']) for i in ix)
                ids=torch.zeros((len(ix),length),device=w.device,dtype=torch.long);mask=torch.zeros_like(ids)
                for j,i in enumerate(ix):
                    v=rows[i]['tokens'];ids[j,:len(v)]=torch.tensor(v,device=w.device);mask[j,:len(v)]=1
                model.gpt_neox(ids,attention_mask=mask,use_cache=False)
                w.sequence_forwards+=len(ix);w.token_forwards+=int(mask.sum())
                h=captured;z=target.encode(h);rec=target.decode(z)
                def codes(x):return torch.relu((x-sp['center'])@sp['encoder'].T+sp['encoder_bias'])
                def pool(x):return (x*mask[...,None]).sum(1)/mask.sum(1)[...,None]
                clean[ix]=(pool(h)@pw.T+pb).flatten().cpu().numpy()
                refined=None
                if c.get('refine_steps'):
                    _,_,initial_columns=transport_delta(h,target,sp,torch.ones(3,device=w.device),initial['encoder.weight'],c['allowance'],False)
                    refined=refine_columns(h,target,sp,initial_columns,c['allowance'],c['refine_steps'])
                    assert float((z-refined.clamp_max(0).abs().sum(-1)).min())>=-1e-5
                    assert int((refined!=0).any(-1).sum(-1).max())<=c['allowance']
                for qi,qv in enumerate(queries.values()):
                    q=torch.tensor(qv,device=w.device,dtype=torch.float32)
                    teacher=-(codes(h)*q)@sp['decoder']
                    energies[qi,ix]=((teacher.square()*mask[...,None]).sum((1,2))/mask.sum(1)).cpu().numpy()
                    deltas={'source':teacher,'raw_reconstruction':-(codes(rec)*q)@sp['decoder']}
                    if refined is not None:deltas['refined_initial_open']=(refined@q)@target.decoder.weight.T
                    for name,matrix in matrices.items():
                        delta,dz,columns=transport_delta(h,target,sp,q,matrix,c['allowance'],active_only='active' in name)
                        deltas[name]=delta
                        if off==0:
                            assert float((z+dz).min())>=-1e-5
                            assert int((columns!=0).any(-1).sum(-1).max())<=c['allowance']
                    for name,delta in deltas.items():
                        outputs[name][qi,ix]=((pool(h+delta)@pw.T+pb).flatten()).cpu().numpy()
                        if name!='source':hidden[name][qi,ix]=(((delta-teacher).square()*mask[...,None]).sum((1,2))/mask.sum(1)).cpu().numpy()
                w.progress('HELDOUT',documents=off+len(ix),total=len(rows))
        np.savez_compressed(w.run/'responses.npz',clean=clean,**outputs)
        np.savez_compressed(w.run/'hidden_errors.npz',source_energy=energies,**hidden)
        write(w.run/'INDEX.json',dict(rows=rows,queries=queries,source_members=source['members']['resid_4'],
            source_annotations={str(i):annotations['resid_4/'+str(i)] for i in source['members']['resid_4']},
            training_span_fraction=relative.tolist(),grammar_singular_values=sv.tolist(),
            source_scope='Only the three published resid_4 members, not the full55-member intervention',
            training='Grammar-only checkpoints; no biography or human response used to train these matrices'))
        for name,lp in outputs.items():
            if name=='source':continue
            denominator=((outputs['source']-clean[None,:])**2).mean(1)
            for qi,query in enumerate(queries):
                w.record(kind='heldout_response',task='human_resid4',method=name,operation=query,row_id=query,component='all128development_documents',
                    target_seed=c['target_seed'],nrmse=float(np.sqrt(((lp[qi]-outputs['source'][qi])**2).mean()/max(denominator[qi],1e-12))),
                    source_rms=float(np.sqrt(denominator[qi])),source_flip_rate=float(((outputs['source'][qi]>0)!=(clean>0)).mean()),
                    hidden_nrmse=float(np.sqrt(hidden[name][qi].mean()/max(energies[qi].mean(),1e-12))))
        w.checks.update(finite=all(np.isfinite(v).all() for v in outputs.values()),no_human_fitting=True,feasible=True,
            frozen_dictionary=all(torch.equal(target.state_dict()[k],v) for k,v in initial.items()))
    except Exception:error=traceback.format_exc()
    finally:
        if handle is not None:handle.remove()
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
