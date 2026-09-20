"""Learn shared intervention transport from unlabeled natural source actions.

Narrow and broad banks share the optimizer, natural states, steps and budget.
The broad bank excludes all seven members in the later two explanations.
"""
from pathlib import Path
import argparse,io,json,sys,time,traceback,zipfile,hashlib
import numpy as np
from run_causalgym_multisite import MultisiteWork,write
from ccad.intervention_transport import allocate_columns


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    args=p.parse_args();c=json.loads(args.config.read_text())
    w=MultisiteWork(c,args.config,['scripts/train_transport_coverage.py','scripts/run_causalgym_multisite.py',
        'scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py','src/ccad/intervention_transport.py'])
    error=None
    try:
        import torch
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
        w.torch=torch;w.device=torch.device(c['device']);torch.cuda.set_device(w.device);torch.cuda.reset_peak_memory_stats()
        w.environment=dict(python=sys.executable,torch=torch.__version__,numpy=np.__version__)
        sys.path.extend([c['dictionary_source_dir'],c['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        w.checked(Path(c['dictionary_source_dir'])/'dictionary_learning/trainers/top_k.py','Pinned TopK','MIT')
        with zipfile.ZipFile(w.checked(c['source_archive'],'Published SFC source dictionary','MIT')) as z:
            raw=z.read(c['source_archive_member'])
        source=torch.load(io.BytesIO(raw),weights_only=True,map_location='cpu')
        write(w.run/'archive_member.json',dict(member=c['source_archive_member'],sha256=hashlib.sha256(raw).hexdigest()))
        del raw
        initial=torch.load(w.checked(Path(c['target_directory'])/f'resid_4_seed{c["target_seed"]}.pt'),map_location=w.device,weights_only=True)
        target=AutoEncoderTopK(512,len(initial['encoder.weight']),int(initial['k'])).to(w.device)
        target.load_state_dict(initial);target.requires_grad_(False)
        h=torch.tensor(np.load(w.checked(c['natural_states']))['h'],device=w.device)
        hfit=h[:-1024];hval=h[-1024:]
        allids=np.array([i for i in range(len(source['encoder.weight'])) if i not in c['excluded_members']])
        rng=np.random.default_rng(c['training_seed'])
        broad=rng.choice(allids,size=c['broad_members'],replace=False).tolist()
        banks={'narrow':c['narrow_members'],'broad':broad}
        if c.get('broad_only'):banks={'broad':broad}
        trace=[];summaries=[]
        for name,members in banks.items():
            enc=source['encoder.weight'][members].to(w.device);bias=source['encoder.bias'][members].to(w.device)
            dec=source['decoder.weight'][:,members].to(w.device);center=source['bias'].to(w.device)
            matrix=torch.nn.Parameter(initial['encoder.weight'].clone())
            optimizer=torch.optim.AdamW([matrix],lr=c['dictionary_lr'],weight_decay=0.)
            generator=torch.Generator(device=w.device).manual_seed(c['training_seed'])
            seen=torch.zeros(len(members),device=w.device,dtype=torch.int64);skipped=0
            for step in range(c['steps']):
                ids=torch.randint(len(hfit),(c['state_batch'],),generator=generator,device=w.device)
                x=hfit[ids];z=target.encode(x)
                with torch.no_grad():
                    codes=torch.relu((x-center)@enc.T+bias)
                    mass=codes.square().sum(0)*dec.square().sum(0)
                    if mass.sum()<=1e-12: skipped+=1
                    selected=torch.multinomial(mass+1e-12,min(4,len(members)),replacement=False,generator=generator)
                    seen[selected]+=1
                    zs=codes[:,selected];ds=dec[:,selected]
                    teacher=-zs[:,:,None]*ds.T[None,:,:]
                columns=-(matrix@ds)[None,:,:]*zs[:,None,:]
                allocated=allocate_columns(z,columns,target.decoder.weight,c['allowance'],active_only=False)
                predicted=(target.decoder.weight@allocated).transpose(-1,-2)
                energy=teacher.square().sum().clamp_min(1e-8)
                # Keep separate actions and their combined field in the objective.
                loss=.5*(predicted-teacher).square().sum()/energy
                loss+=.5*(predicted.sum(1)-teacher.sum(1)).square().sum()/teacher.sum(1).square().sum().clamp_min(1e-8)
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([matrix],1.);optimizer.step()
                if (step+1)%64==0:
                    item=dict(bank=name,step=step+1,loss=float(loss.detach()),members_seen=int((seen>0).sum()))
                    trace.append(item);w.progress('COVERAGE_TRAINING',**item)
                if time.perf_counter()-w.wall_start>c['budget_seconds']:raise TimeoutError('Allocated coverage budget reached')
            folder=w.run/('transport_'+name+'_open');folder.mkdir()
            torch.save(matrix.detach().cpu(),folder/'write_matrix.pt');torch.save(initial,folder/'dictionary.pt')
            np.savez_compressed(folder/'relation.npz',native=np.load(c['reference_relation'])['native'])
            actually_seen=np.array(members)[seen.cpu().numpy()>0]
            ds=source['decoder.weight'][:,actually_seen].to(w.device)
            sv=torch.linalg.svdvals(ds)
            summaries.append(dict(bank=name,members=members,counts=seen.cpu().tolist(),skipped_steps=skipped,
                numerical_rank=int((sv>sv[0]*1e-5).sum()),effective_rank=float(sv.square().sum()**2/sv.pow(4).sum()),
                final_loss=trace[-1]['loss'],checkpoint=str(folder)))
            w.record(kind='coverage_training',task='natural_actions',method=name,operation='partial_and_full',row_id=name,
                component='natural_discovery',target_seed=c['target_seed'],loss=trace[-1]['loss'])
        write(w.run/'training.json',trace);write(w.run/'coverage.json',summaries)
        w.checks.update(frozen_dictionary=all(torch.equal(target.state_dict()[k],v) for k,v in initial.items()),
            broad_excludes_both_explanations=not bool(set(broad)&set(c['excluded_members'])),finite=all(np.isfinite(v['loss']) for v in trace))
    except Exception:error=traceback.format_exc()
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
