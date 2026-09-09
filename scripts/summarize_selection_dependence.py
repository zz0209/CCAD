"""Conditional node/task/frame bootstrap for saved correspondence decisions.

Shared source/target nodes use the same bootstrap weights. All queries on the
same task share resampled lexical-frame weights. This is a finite-suite
sensitivity analysis, not uncertainty over base models, layers, or SAE designs.
"""
from pathlib import Path
from collections import defaultdict
import argparse,json,hashlib,csv
import numpy as np

def frame(row):
    parts=[a if a==b else f'<ALTERNATING_REGION_{i}>' for i,(a,b) in enumerate(zip(row['base'],row['src']))]
    return hashlib.sha256(json.dumps(parts,ensure_ascii=False).encode()).hexdigest()

def alternating_identities(row):
    # These are observed changed lexical spans, not automatic NER labels.
    # In gender tasks they include the actual names that the frame masks out.
    spans=sorted({' '.join(v.split()) for a,b in zip(row['base'],row['src']) if a!=b for v in [a,b]})
    return [hashlib.sha256(v.encode()).hexdigest() for v in spans]

def load(run,budget_replay=None):
    cfg=json.loads((run/'config.resolved.json').read_text());cs=json.loads((run/'selection_choices.json').read_text())['choices'];byq={c['query']:c for c in cs};panel=json.loads((run/'panel.json').read_text())['rows'];pmap={r['row_id']:r for r in panel}
    replay=defaultdict(list)
    if budget_replay:
        doc=json.loads(Path(budget_replay).read_text())
        for r in doc['records']:replay[r['query']].append(r)
        assert set(replay)==set(byq)
    needed=defaultdict(set);policies={}
    for c in cs:
        p={k:[v] for k,v in c['selected'].items()}
        for name in c['held_summary']:p['fixed:'+name]=[name]
        for name,score in c['scores'].items():
            best=max(score.values());p['tie_uniform:'+name]=[m for m,v in score.items() if v==best or abs(v-best)<=1e-10]
        for b in (replay[c['query']] if budget_replay else c.get('budget_choices',[])):p[b['policy']+'_'+str(b['budget'])]=[b['selected']]
        policies[c['query']]=p;needed[c['query']].update(m for methods in p.values() for m in methods)
    values=defaultdict(dict)
    with (run/'metrics.raw.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if r['method'] in needed[r['query']]:values[r['query']][(r['method'],r['row_id'])]=float(r['iia'])
    output=[]
    for c in cs:
        ids=sorted({i for _,i in values[c['query']]});frames=[frame(pmap[i]) for i in ids]
        Y={p:[float(np.mean([values[c['query']][(method,i)] for method in methods])) for i in ids] for p,methods in policies[c['query']].items()}
        output.append(dict(run=run.name,task=c['task'],objective=c['objective'],training_run=cfg['training_run'],source_seed=c['source_seed'],target_seed=c['target_seed'],query=c['query'],frames=frames,alternating_identities=[alternating_identities(pmap[i]) for i in ids],values=Y))
    return output

def analyze(rows,draws=2000):
    policies=sorted(set.intersection(*(set(r['values']) for r in rows)));tasks=sorted({r['task'] for r in rows});families=sorted({r['training_run'] for r in rows})
    seedsets={f:sorted({r[s] for r in rows if r['training_run']==f for s in ['source_seed','target_seed']}) for f in families}
    frames={t:sorted({x for r in rows if r['task']==t for x in r['frames']}) for t in tasks}
    fmaps={t:{v:i for i,v in enumerate(frames[t])} for t in tasks};arrays=[];point=[]
    for r in rows:
        idx=[fmaps[r['task']][v] for v in r['frames']];n=len(frames[r['task']]);count=np.bincount(idx,minlength=n)
        sums=np.array([np.bincount(idx,weights=r['values'][p],minlength=n) for p in policies]);arrays.append((count,sums));point.append([np.mean(r['values'][p]) for p in policies])
    point=np.asarray(point);rng=np.random.default_rng(420921);boot=[];fixed_task_boot=[];attempts=0
    while len(boot)<draws and attempts<draws*10:
        attempts+=1;sw={f:dict(zip(seedsets[f],rng.multinomial(len(seedsets[f]),np.ones(len(seedsets[f]))/len(seedsets[f])))) for f in families}
        tw=dict(zip(tasks,rng.multinomial(len(tasks),np.ones(len(tasks))/len(tasks))))
        fw={t:rng.multinomial(len(frames[t]),np.ones(len(frames[t]))/len(frames[t])) for t in tasks}
        totals=np.zeros(len(policies));fixed_totals=np.zeros(len(policies));den=0.;fixed_den=0.
        for r,(counts,sums) in zip(rows,arrays):
            nod=sw[r['training_run']][r['source_seed']]*sw[r['training_run']][r['target_seed']];f=fw[r['task']];mass=counts@f
            if nod==0 or mass==0:continue
            val=sums@f/mass;weight=nod*tw[r['task']];totals+=weight*val;den+=weight;fixed_totals+=nod*val;fixed_den+=nod
        if den and fixed_den:boot.append(totals/den);fixed_task_boot.append(fixed_totals/fixed_den)
    B=np.asarray(boot);F=np.asarray(fixed_task_boot);assert len(B)==draws
    comparisons=[]
    pairs=[('source_endpoint',p) for p in ['finite_margin','cosine','pw_mcc','semantic_ot_group','natural_mse','task_mse','base_linear','anchored_base_linear','tie_uniform:natural_mse','tie_uniform:cosine','fixed:native_task_ridge_axis','fixed:native_shared_axis','fixed:raw_das']]
    pairs += [('fixed:'+a,'fixed:'+b) for a,b in [('native_task_ridge_axis','reencode_shared_axis'),('native_shared_axis','reencode_shared_axis'),('native_task_ridge_axis','native_global_pw_axis'),('native_task_ridge_axis','native_semantic_ot_axis'),('native_task_ridge_axis','raw_das'),('native_shared_axis','native_random_support'),('native_shared_axis','native_wrong_axis')]]
    pairs += [('fixed:'+a,'fixed:'+b) for a,b in [('native_reencode_count','reencode_shared_axis'),('native_residual_task_ridge_axis','native_shared_axis'),('native_calibrated_shared_axis','native_shared_axis')]]
    pairs += [('fixed:compiled_shared_axis','fixed:'+b) for b in ['native_shared_axis','reencode_shared_axis','native_reencode_count','native_global_pw_axis','raw_das']]
    pairs += [('fixed:compiled_functional_axis','fixed:'+b) for b in ['native_shared_axis','compiled_shared_axis','raw_signed_distill','raw_das','source','shared_axis_reader','reencode_shared_axis','native_reencode_count','native_global_pw_axis','native_semantic_ot_axis','native_task_ridge_axis','native_random_support','native_wrong_axis']]
    pairs += [('source_screen_'+str(b),p+'_'+str(b)) for b in [24,48,96] for p in ['direct_all','direct_balanced','direct_halving','cosine_screen','pw_screen','natural_screen']]
    for a,b in pairs:
        if a not in policies or b not in policies:continue
        i,j=policies.index(a),policies.index(b);loo=[]
        for family in families:
            for seed in seedsets[family]:
                keep=[n for n,r in enumerate(rows) if not(r['training_run']==family and seed in [r['source_seed'],r['target_seed']])]
                loo.append(dict(training_run=family,seed=seed,remaining_queries=len(keep),difference=float((point[keep,i]-point[keep,j]).mean()) if keep else None))
        comparisons.append(dict(left=a,right=b,difference=float((point[:,i]-point[:,j]).mean()),seed_task_frame_interval=np.quantile(B[:,i]-B[:,j],[.025,.975]).tolist(),seed_frame_fixed_task_interval=np.quantile(F[:,i]-F[:,j],[.025,.975]).tolist(),incident_seed_leave_out=loo))
    return dict(queries=len(rows),policies=policies,macro_iia=dict(zip(policies,point.mean(0).tolist())),comparisons=comparisons,bootstrap_draws=draws,bootstrap_attempts=attempts,seed_sets=seedsets,tasks=tasks,lexical_frames={t:len(frames[t]) for t in tasks},query_distributions=[dict(query=r['query'],task=r['task'],training_run=r['training_run'],objective=r['objective'],source_seed=r['source_seed'],target_seed=r['target_seed'],mean_iia=dict(zip(policies,point[i].tolist()))) for i,r in enumerate(rows)],scope='Two displayed sensitivities: resampled observed seed nodes + tasks + lexical frames, and seed nodes + frames conditional on fixed tasks. Resampling a shared seed weights both incident edges; duplicated edge samples are never called independent replicates. Frame keeps all identical regions and masks alternating regions; all model/objective/seed queries share each task frame draw. Numeric seed IDs within a training run share weights across objective bundles. Small5-node graphs and finite tasks are not calibrated uncertainty over new model/hook/training-corpus populations. A single-edge development run cannot establish initialization uncertainty; report its finite-suite descriptive results accordingly. Fixed methods are reported separately from selectors; averaging tied maximizers does not use outcomes to break a tie.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--draws',type=int,default=2000);a=p.parse_args();rows=[r for run in a.runs for r in load(run)];result=analyze(rows,a.draws);a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'selection_dependence.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'query_frame_values.json').write_text(json.dumps(rows)+'\n')
    for r in result['comparisons']:print(r['left'],r['right'],r['difference'],r['seed_task_frame_interval'])
