"""Compare fixed role parts, their dose variants, and the full intervention."""
import argparse,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    raw=args.run/'metrics.raw.jsonl';cfg=json.loads((args.run/'config.resolved.json').read_text())
    assert json.loads((args.run/'status.json').read_text())['status']=='PASS'
    rows=[json.loads(x) for x in raw.read_text().splitlines()];cells=defaultdict(list)
    for r in rows:
        if r['kind']=='source_patch' and r['method'].startswith(('source_','member_')):cells[r['method'],r['operation']].append(r)
    summaries=[];lookup={}
    for (method,op),rr in sorted(cells.items()):
        for r in rr:lookup[method,op,r['seed'],r['row_id']]=r
        summaries.append(dict(method=method,operation=op,n=len(rr),H=100*np.mean([r['exact_hybrid'] for r in rr]),
            T=100*np.mean([r['target_digit_success'] for r in rr]),P=100*np.mean([r['preserve_digit_success'] for r in rr]),
            mean_norm=float(np.mean([r['edit_norm'] for r in rr]))))
    contrasts=[];rng=np.random.default_rng(943015);nq=cfg['pairs_per_template']
    weights=rng.multinomial(nq,np.full(nq,1/nq),10000)/nq
    for prefix in ['source','member']:
        full=prefix+'_full'
        for op in ['unit','tens']:
            if (full,op) not in cells:continue
            variants=[m for m,o in cells if o==op and m.startswith(prefix+'_part')]
            qdiff=defaultdict(list);qnone=defaultdict(list);qorig=defaultdict(list)
            for r in cells[full,op]:
                q=r['row_id']%nq;seed=r['seed'];rid=r['row_id'];fh=int(r['exact_hybrid'])
                hs=[int(lookup[m,op,seed,rid]['exact_hybrid']) for m in variants]
                originals=[int(lookup[prefix+f'_part{i}_bank0',op,seed,rid]['exact_hybrid']) for i in [0,1]]
                qdiff[q].append(fh-max(hs));qnone[q].append(fh and not any(hs));qorig[q].append(fh and not any(originals))
            for metric,values in [('full_minus_per_case_best_tested_half',qdiff),('full_success_all_tested_halves_fail',qnone),('full_success_original_halves_fail',qorig)]:
                v=np.array([np.mean(values[q]) for q in range(nq)])
                contrasts.append(dict(method=prefix,operation=op,metric=metric,points=float(v.mean()*100),interval=(np.quantile(weights@v,[.025,.975])*100).tolist(),tested_variants=variants))
    out=dict(run=args.run.as_posix(),cells=summaries,contrasts=contrasts,scope=cfg['scope'],
             raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
             statistics='Paired question-cluster bootstrap,10000 draws,seed943015. Fixed forms and dependent SAE cohort retained. Per-case best half is a label-informed upper envelope, not a deployable selected method.')
    args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
