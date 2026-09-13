"""Execute frozen delete-A/preserve-B queries from a saved fuzzy member relation."""
from datetime import datetime,timezone
import json,itertools


def run_queries(cfg,w,D,capture,forward,checked,write,log,budget):
    import numpy as np,torch
    parent=checked(cfg['relational_query_parent']+'/status.json').parent
    assert json.loads((parent/'status.json').read_text())['status']=='PASS'
    pc=json.loads(checked(parent/'config.resolved.json').read_text());selection=json.loads(checked(parent/'SELECTION_FREEZE.json').read_text())['decisions']
    proposals=json.loads(checked(parent/'PROPOSAL_FREEZE.json').read_text())['rows'];ops=sorted(pc['operations'],key=lambda x:pc['operations'][x].index(1))
    assert cfg['source_tasks']==pc['source_tasks'] and cfg['target_seeds']==pc['target_seeds']
    queryrows=[];masks={}
    for item in proposals:
        obj,s,t=item['objective'],item['source'],item['target']
        with np.load(checked(parent/item['path'])) as a:
            for family in cfg['query_families']:
                score=a[family+'_score'];normalized=score/np.maximum(score.max(0),1e-15)[None,:]
                sets=[]
                for op in ops:
                    dec=next(x for x in selection if x['objective']==obj and x['source']==s and x['operation']==op and x['family']==family and x['validation_pairs_per_task']==16)
                    sets.append(set(a[f'{op}__{family}__{dec["chosen_allowance"]}'].tolist()))
                for k,l in itertools.permutations(range(3),2):
                    ids=np.asarray(sorted(sets[k]),dtype=int);exclusive=sets[k]-sets[l];n=len(exclusive)
                    own=ids[np.argsort(-score[ids,k],kind='stable')[:n]]
                    soft=ids[np.argsort(-(normalized[ids,k]-normalized[ids,l]),kind='stable')[:n]]
                    strategies=dict(full=sorted(sets[k]),difference=sorted(exclusive),own_matched=own.tolist(),soft_matched=soft.tolist())
                    for strategy,members in strategies.items():
                        detail=dict(objective=obj,source=s,target=t,family=family,strategy=strategy,operation=f'{k}>{l}',delete_task=k,preserve_task=l,members=members,actual_members=len(members),full_members=len(ids),shared_members=len(sets[k]&sets[l]))
                        queryrows.append(detail);mask=torch.zeros(len(D[obj,t]),device=w.device);mask[members]=1
                        masks[obj,s,family,f'{k}>{l}',strategy]=mask
    evaluate_queries(cfg,w,D,capture,forward,write,log,budget,queryrows,masks)


def evaluate_queries(cfg,w,D,capture,forward,write,log,budget,queryrows,masks):
    import numpy as np,torch
    write(w.run/'QUERY_FREEZE.json',dict(written_at_utc=datetime.now(timezone.utc).isoformat(),evaluation_outcomes_consumed=0,new_target_calibration_or_fitting=0,rows=queryrows,score_rule=cfg.get('query_score_rule','Normalize each nonnegative relation column by its full-dictionary maximum. Within the previously selected A set, rank normalized A minus normalized B. Keep exactly the cardinality of A minus B. Own-score ranking uses the identical A pool and cardinality; full A and hard A minus B are retained.')))
    log('ALL_RELATIONAL_QUERIES_FROZEN',requests=len(queryrows));data=capture('evaluation');results=[];reused=0;unique=0
    for obj in cfg['objectives']:
        for s,t in zip(cfg['seeds'],cfg['target_seeds']):
            z=data['codes'][obj,t];d=D[obj,t];cache={}
            for detail in [x for x in queryrows if x['objective']==obj and x['source']==s]:
                key=tuple(sorted(detail['members']));family=detail['family'];strategy=detail['strategy'];op=detail['operation']
                if key in cache:edited,norm=cache[key];reused+=1
                else:
                    mask=masks[obj,s,family,op,strategy];delta=(z*mask)@d;norm=delta.norm(dim=1).cpu().numpy();parts=[]
                    for start in range(0,len(data['rows']),cfg['batch_pairs']):
                        rr=data['rows'][start:start+cfg['batch_pairs']];m,h,_,_=forward(rr,delta[start:start+len(rr)],phase='relational_query')
                        assert float((h-data['hidden'][start:start+len(rr)]).abs().max())<cfg['hidden_atol'];parts.append(m.cpu().numpy());budget()
                    edited=np.concatenate(parts);cache[key]=(edited,norm);unique+=1
                clean=data['clean'].cpu().numpy();taskstats={}
                for i,r in enumerate(data['rows']):
                    w.record(kind='relational_query',task=r['task'],row_id=r['row_id'],component=r['task']+':'+str(r['row_id']),mode=obj+'|'+op,method=family+'|'+strategy,operation=op,seed=s,target_seed=t,objective=obj,split='evaluation',family=family,strategy=strategy,margin=float(edited[i]),clean_margin=float(clean[i]),edit_norm=float(norm[i]),introduced_error=bool(clean[i]>0 and edited[i]<=0),actual_members=detail['actual_members'],shared_members=detail['shared_members'])
                for task in cfg['source_tasks']:
                    ix=np.asarray([i for i,r in enumerate(data['rows']) if r['task']==task]);taskstats[task]=dict(introduced_error_rate=float(((clean[ix]>0)&(edited[ix]<=0)).mean()),margin_decrement=float((clean[ix]-edited[ix]).mean()),mean_edit_norm=float(norm[ix].mean()))
                aa=taskstats[cfg['source_tasks'][detail['delete_task']]];bb=taskstats[cfg['source_tasks'][detail['preserve_task']]]
                results.append(dict(**detail,task_statistics=taskstats,selectivity=aa['introduced_error_rate']-bb['introduced_error_rate'],requested_error_rate=aa['introduced_error_rate'],preserved_error_rate=bb['introduced_error_rate'],margin_selectivity=aa['margin_decrement']-bb['margin_decrement']))
            log('RELATIONAL_TARGET_EVALUATED',objective=obj,target=t,unique_masks=unique,reused_requests=reused)
    write(w.run/'query_results.json',dict(rows=results,unique_evaluated_masks=unique,reused_requests=reused,scope=cfg['scope']))
    w.checks.update(queries_frozen_before_any_new_outcome=True,binary_code_deletions=True,matched_cardinality=all(x['actual_members']==x['full_members']-x['shared_members'] for x in queryrows if x['strategy']!='full'))
