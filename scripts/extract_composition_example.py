"""Fixed first held lexical example; no outcome-based case selection."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    cfg=json.loads((args.run/'config.resolved.json').read_text());parent=Path(cfg['material_run']);panel=json.loads((parent/'panel.json').read_text());rows=panel['rows'];donors={f:np.array([p[f] for p in panel['pairs']]) for f in ['number','time','joint']}
    first=next(r['id'] for r in rows if r['block'] in cfg['evaluation_blocks']);case=rows[first];s,t=1,2
    raw=[];base=[]
    for line in (args.run/'metrics.raw.jsonl').open():
        r=json.loads(line)
        if r['source_seed']==s and r.get('target_seed') in [None,t] and r['row_id']==first:raw.append(r)
    for line in (parent/'metrics.raw.jsonl').open():
        r=json.loads(line)
        if r['row_id']==first and r['kind'] in ['baseline','intervention']:base.append(r)
    source=np.load(Path(cfg['source_run'])/f'seed{s}_source_coordinates.npz');codes=np.load(parent/f'seed{t}_codes.npz');maps=np.load(args.run/f'maps_s{s}_t{t}.npz');factor_rows=[]
    for factor in ['number','time']:
        budget=cfg['source_budgets'][factor];coords=source[f'{factor}_{budget}_coordinates'];basis=source[f'{factor}_{budget}_basis'];support=source[f'{factor}_{budget}_support'];coeff=source[f'{factor}_{budget}_coefficients'];members=maps[factor+'_fcc_members'];z=codes[factor+'_z'];dz=z[donors[factor]]-z;pred=dz@maps[factor+'_fcc_group_coefficients']
        discovery=np.array([r['block'] in cfg['discovery_blocks'] and r['cue_id'] in cfg['discovery_cues'] for r in rows]);_,singular,vt=np.linalg.svd(coords[discovery],full_matrices=False)
        sign=np.array([1 if rows[i]['number' if factor=='number' else 'past'] else -1 for i in donors[factor]])
        cue_vectors=[]
        for cue in range(len(cfg['time_cues'])):
            mask=np.array([r['cue_id']==cue for r in rows]);q=np.mean(coords[mask]*sign[mask,None],axis=0);qh=np.mean(pred[mask]*sign[mask,None],axis=0)
            cue_vectors.append(dict(cue_id=cue,label=' → '.join(cfg['time_cues'][cue]),known=cue in cfg['discovery_cues'],source_pc=(q@vt[:2].T).tolist(),fcc_pc=(qh@vt[:2].T).tolist(),source_outside_pc2=float(np.linalg.norm(q-q@vt[:2].T@vt[:2])),source_norm=float(np.linalg.norm(q))))
        factor_rows.append(dict(factor=factor,source_members=support.tolist(),target_members=members.tolist(),source_span_rank=basis.shape[1],discovery_singular_values=singular.tolist(),discovery_numerical_rank=int(np.sum(singular>singular[0]*1e-8)),cue_vectors=cue_vectors))
    result=dict(selection='Source1,target2,first configured held lexical input; fixed before reading its outcomes',row_id=first,source_seed=s,target_seed=t,input=case,donors={f:rows[int(donors[f][first])] for f in donors},metrics=raw,raw_and_baseline=base,factors=factor_rows)
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'R3_FIXED_EXAMPLE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(row_id=first,text=case['text'],factors=[dict(factor=r['factor'],source_rank=r['source_span_rank'],process_rank=r['discovery_numerical_rank'],target_members=len(r['target_members'])) for r in factor_rows],metrics=[dict(method=r['method'],factor=r['factor'],label=r['label'],number_logodds=r['number_logodds'],past_logodds=r['past_logodds']) for r in raw if r['method'] in ['source_teacher','fcc_group','same_members_native','direct_target_native']])))


if __name__=='__main__':main()
