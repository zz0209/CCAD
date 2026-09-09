"""Additional lexical-identity sensitivity of the retained primary contrasts.

The prespecified multinomial node/task/frame summaries remain unchanged. This
separate positive-multiplier analysis also shares weights across recurring
changed spans (actual names where the task contains names). It does not claim
automatic named-entity annotation or calibrated population confidence.
"""
from pathlib import Path
import argparse,json,time
from collections import defaultdict
import numpy as np

PAIRS=[('fixed:compiled_functional_axis','fixed:native_shared_axis'),('fixed:compiled_functional_axis','fixed:raw_signed_distill'),('fixed:native_shared_axis','fixed:reencode_shared_axis'),('source_screen_48','direct_halving_48')]

def analyze(rows,draws):
    tasks=sorted({r['task'] for r in rows});ti={t:i for i,t in enumerate(tasks)}
    seeds=sorted({r[s] for r in rows for s in ['source_seed','target_seed']});si={s:i for i,s in enumerate(seeds)}
    fs=sorted({(r['task'],v) for r in rows for v in r['frames']});fi={v:i for i,v in enumerate(fs)}
    es=sorted({v for r in rows for vv in r['alternating_identities'] for v in vv});ei={v:i for i,v in enumerate(es)}
    arrays=[]
    for r in rows:
        d=np.array([np.asarray(r['values'][a])-np.asarray(r['values'][b]) for a,b in PAIRS]).T
        # Products share one multiplier for every occurrence of a lexical span,
        # including across tasks. A row can contain several changed regions.
        # Sparse column indices avoid a large dense product in each draw.
        k=max((len(v) for v in r['alternating_identities']),default=0);idx=np.full((len(d),k),len(es),dtype=np.int64)
        for i,values in enumerate(r['alternating_identities']):idx[i,:len(values)]=[ei[v] for v in values]
        arrays.append((ti[r['task']],si[r['source_seed']],si[r['target_seed']],np.array([fi[r['task'],v] for v in r['frames']]),idx,d))
    point=np.mean([a[-1].mean(0) for a in arrays],axis=0);rng=np.random.default_rng(2026090921);boot=[];fixed=[]
    for _ in range(draws):
        sw=rng.exponential(size=len(seeds));tw=rng.exponential(size=len(tasks));fw=rng.exponential(size=len(fs));ew=np.r_[rng.exponential(size=len(es)),1.]
        total=np.zeros(len(PAIRS));ft=np.zeros(len(PAIRS));den=0.;fd=0.
        for task,s,t,fr,en,d in arrays:
            w=fw[fr]*ew[en].prod(1);v=w@d/w.sum();node=sw[s]*sw[t];total+=node*tw[task]*v;den+=node*tw[task];ft+=node*v;fd+=node
        boot.append(total/den);fixed.append(ft/fd)
    B=np.array(boot);F=np.array(fixed)
    return dict(queries=len(rows),tasks=len(tasks),seed_nodes=len(seeds),lexical_frames=len(fs),alternating_span_identities=len(es),draws=draws,comparisons=[dict(left=a,right=b,difference=float(point[i]),seed_task_frame_span_interval=np.quantile(B[:,i],[.025,.975]).tolist(),seed_frame_span_fixed_task_interval=np.quantile(F[:,i],[.025,.975]).tolist()) for i,(a,b) in enumerate(PAIRS)],scope='Additional crossed independent exponential-multiplier sensitivity, separate from the prespecified multinomial node/task/frame results. A repeated changed lexical span shares its multiplier across all occurrences; actual names are included when present. Unchanged context is grouped by the original frame rule. This is not a linguistic entity annotation, and five observed seed nodes cannot calibrate a model/corpus population interval.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--draws',type=int,default=2000);a=p.parse_args();timer=time.perf_counter();groups=defaultdict(list)
    for r in json.loads(a.input.read_text()):groups[(r['training_run'],r['objective'])].append(r)
    out=[]
    for (training,objective),rows in groups.items():
        result=analyze(rows,a.draws);result.update(training_run=training,objective=objective);out.append(result);print(json.dumps(dict(training_run=training,objective=objective,queries=len(rows),wall_seconds=time.perf_counter()-timer)),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(conditions=out,wall_seconds=time.perf_counter()-timer),indent=2)+'\n')
if __name__=='__main__':main()
