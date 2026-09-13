"""Post-evaluation attribution check of the R26 native extrapolation signal.

Source components, target pools and native allocations remain frozen.
Regularization/rank selection uses old development hidden fields only.
The new-paradigm outcomes are already exposed: this is a diagnostic control,
not a new independent confirmation.
"""
import json
from datetime import datetime,timezone


def run_controls(cfg,w,D,capture,forward,checked,write,log,budget):
    import numpy as np,torch
    from ccad.artifacts import sha256
    parent=checked(cfg['raw_control_parent']+'/config.resolved.json').parent
    assert json.loads(checked(parent/'status.json').read_text())['status']=='PASS'
    fitdata=capture('fit');dev=capture('development');maps={};selection=[]
    for obj in cfg['objectives']:
        for s in cfg['seeds']:
            t=cfg['seeds'][(cfg['seeds'].index(s)+1)%len(cfg['seeds'])]
            with np.load(checked(parent/f'{obj}_s{s}_t{t}_map.npz')) as a:
                tp=torch.tensor(a['target_members'],device=w.device);sp=torch.tensor(a['source_members'],device=w.device);sg=torch.tensor(a['source_gate'],device=w.device)
                candidates={'full_ridge_0.001':torch.tensor(a['raw_full'],device=w.device),'rank2_ridge_0.001':torch.tensor(a['raw_rank2'],device=w.device)}
            x=fitdata['codes'][obj,t][:,tp].double();z=fitdata['codes'][obj,s][:,sp].double();d=D[obj,s][sp].double();n=len(x)
            y=torch.stack([(z*sg[:,k].double())@d for k in range(3)],1);K=x.T@x/n;rhs=x.T@y.reshape(n,-1)/n
            for fraction in [.01,.1,1.]:
                A=K+fraction*K.diag().mean()*torch.eye(len(tp),device=w.device,dtype=torch.float64)
                candidates[f'full_ridge_{fraction:g}']=torch.linalg.solve(A,rhs).reshape(len(tp),3,1024).float()
            A=K+.001*K.diag().mean()*torch.eye(len(tp),device=w.device,dtype=torch.float64);root=torch.linalg.cholesky(A).T
            rank8=[]
            for k in range(3):
                B=candidates['full_ridge_0.001'][:,k].double();_,_,vh=torch.linalg.svd(root@B,full_matrices=False);rank8.append(B@vh[:8].T@vh[:8])
            candidates['rank8_ridge_0.001']=torch.stack(rank8,1).float()
            xv=dev['codes'][obj,t][:,tp];zv=dev['codes'][obj,s][:,sp];dv=D[obj,s][sp]
            yv=torch.stack([(zv*sg[:,k])@dv for k in range(3)],1)
            losses={name:float((torch.einsum('np,pkd->nkd',xv,B)-yv).double().square().sum(-1).mean()) for name,B in candidates.items()}
            chosen=min(losses,key=lambda name:(losses[name],name));maps[obj,s]=dict(target=t,tp=tp,sp=sp,sg=sg,candidates=candidates,chosen=chosen)
            selection.append(dict(objective=obj,source=s,target=t,hidden_field_mse=losses,chosen=chosen))
            np.savez_compressed(w.run/f'{obj}_s{s}_t{t}_raw_controls.npz',target_members=tp.cpu().numpy(),**{name:B.cpu().numpy() for name,B in candidates.items()})
            log('RAW_CONTROL_FIT',objective=obj,source=s,chosen=chosen,validation=losses);budget()
    write(w.run/'raw_control_selection.json',dict(rows=selection,rule='Minimum old-development hidden-field MSE across the three singleton fields; six fixed regularization/rank candidates. No new-paradigm target output used to choose a candidate.'))
    write(w.run/'CONTROL_FREEZE.json',dict(written_at_utc=datetime.now(timezone.utc).isoformat(),selection_sha256=sha256(w.run/'raw_control_selection.json'),scope='Native/source maps unchanged. All new paradigms exposed in primary v2; this is a post-evaluation attribution control, not fresh confirmation.'))
    del fitdata,dev
    data=capture('new')
    for obj in cfg['objectives']:
        for s in cfg['seeds']:
            mp=maps[obj,s];t=mp['target'];zt=data['codes'][obj,t][:,mp['tp']];zs=data['codes'][obj,s][:,mp['sp']];ds=D[obj,s][mp['sp']]
            for op,cw in cfg['operations'].items():
                c=torch.tensor(cw,device=w.device,dtype=zt.dtype);src=(zs*(mp['sg']@c))@ds
                methods={name:zt@torch.einsum('pkd,k->pd',B,c) for name,B in mp['candidates'].items() if name not in ['full_ridge_0.001','rank2_ridge_0.001']}
                methods['selected_raw']=zt@torch.einsum('pkd,k->pd',mp['candidates'][mp['chosen']],c);methods={'source':src,**methods}
                for off in range(0,len(data['rows']),cfg['batch_pairs']):
                    rr=data['rows'][off:off+cfg['batch_pairs']];clean,h,clp,_=forward(rr,phase='control_clean');assert float((h-data['hidden'][off:off+len(rr)]).abs().max())<cfg['hidden_atol']
                    outputs={name:forward(rr,delta[off:off+len(rr)],phase='control_'+name) for name,delta in methods.items()};sm,_,slp,_=outputs['source'];effect=(slp.exp()*(slp-clp)).sum(1)
                    for method,(margin,_,lp,_) in outputs.items():
                        divergence=(slp.exp()*(slp-lp)).sum(1)
                        for j,r in enumerate(rr):
                            w.record(kind='component_transfer',task=r['task'],row_id=r['row_id'],component=r['task']+':'+str(r['row_id']),mode=obj+'|'+op,operation=op,method=method,seed=s,target_seed=t,objective=obj,split='new',margin=float(margin[j]),clean_margin=float(clean[j]),source_margin=float(sm[j]),margin_error=abs(float(margin[j]-sm[j])),source_kl=float(divergence[j]),source_effect_kl=float(effect[j]),edit_norm=float(methods[method][off+j].norm()),source_changed=bool((sm[j]>0)!=(clean[j]>0)),source_decision_agreement=bool((margin[j]>0)==(sm[j]>0)))
                    budget()
                log('RAW_CONTROL_EVALUATED',objective=obj,source=s,operation=op)
