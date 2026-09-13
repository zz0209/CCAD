"""Export finite source-component responses for later member correspondence."""
import json


def export_paths(cfg,w,D,sources,select,forward,write,log,budget):
    import numpy as np,torch
    from ccad.artifacts import sha256
    ix=[i for i,r in enumerate(select['rows']) if cfg['path_source_range'][0]<=r['row_id']<cfg['path_source_range'][1]]
    rows=[select['rows'][i] for i in ix];hidden=select['hidden'][ix];clean=select['clean'][ix];batch=cfg['gradient_batch_pairs'];summaries=[]
    for obj in cfg['objectives']:
        for s in cfg['seeds']:
            gate=sources[obj,s];z=select['codes'][obj,s][ix];d=D[obj,s];G=[];Y=[];members=torch.where(gate.sum(1)>0)[0]
            for k,task in enumerate(cfg['source_tasks']):
                delta=(z*gate[:,k])@d;gbar=torch.zeros_like(hidden);effect=[]
                for off in range(0,len(rows),batch):
                    rr=rows[off:off+batch];dd=delta[off:off+len(rr)]
                    c,hh,_,_=forward(rr,gradient=True,phase='source_clean_endpoint')
                    e,_,_,_=forward(rr,dd,gradient=True,phase='source_deleted_endpoint')
                    assert float((hh-hidden[off:off+len(rr)]).abs().max())<cfg['hidden_atol']
                    effect.append(c-e)
                    for alpha in cfg['path_midpoints']:
                        _,_,_,g=forward(rr,alpha*dd,gradient=True,phase='source_path_gradient')
                        gbar[off:off+len(rr)]+=g/len(cfg['path_midpoints'])
                    for j,r in enumerate(rr):
                        w.record(kind='source_function',task=r['task'],row_id=r['row_id'],component=r['task']+':'+str(r['row_id']),mode=obj+'|'+str(k),operation=str(k),method='source_component',seed=s,objective=obj,split='source_selection',margin=float(e[j]),clean_margin=float(c[j]),introduced_error=bool(c[j]>0 and e[j]<=0),edit_norm=float(dd[j].norm()))
                    budget()
                effect=torch.cat(effect);pred=(gbar*delta).sum(1);err=float((effect-pred).abs().mean());scale=float(effect.abs().mean());G.append(gbar);Y.append(effect)
                profile={t:dict(margin_decrement=float(effect[[i for i,r in enumerate(rows) if r['task']==t]].mean())) for t in cfg['source_tasks']}
                summaries.append(dict(objective=obj,seed=s,operation=k,task=task,members=int(gate[:,k].sum()),integration_mae=err,mean_absolute_effect=scale,relative_integration_mae=err/max(scale,1e-12),profile=profile))
                log('SOURCE_PATH_READY',objective=obj,source=s,operation=k,integration_mae=err,relative_mae=err/max(scale,1e-12),profile=profile)
            t=cfg['seeds'][(cfg['seeds'].index(s)+1)%len(cfg['seeds'])]
            path=w.run/f'{obj}_s{s}_t{t}_path_relation.npz'
            np.savez_compressed(path,source_path_gradient=torch.stack(G).cpu().numpy(),source_finite_effect=torch.stack(Y).cpu().numpy(),source_members=members.cpu().numpy(),source_gate=gate[members].cpu().numpy(),row_ids=np.asarray([r['row_id'] for r in rows]),task_ids=np.asarray([cfg['source_tasks'].index(r['task']) for r in rows]))
            log('SOURCE_PATH_BANK_SAVED',objective=obj,source=s,path=path.name,sha256=sha256(path))
    write(w.run/'source_path_diagnostics.json',dict(rows=summaries,scope='Source-selection examples only. Exact component effects and numerical path identity are measured before target consumer selection; source responses are subsequently evaluated on new pairs.'))
    w.checks.update(source_components_source_only=True,all_source_paths_recorded=True,source_gradient_finite_difference=True)
