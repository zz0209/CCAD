"""Export retained native maps and fixed context inputs; no fitting or LM run."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from safetensors.numpy import load_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_causalgym_multisite import aligned_positions
from ccad.native_participation import participant_delta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    records=[];cache={}
    for s in range(1,6):
        run=ROOT/'runs'/f'FINAL5_R15_participation_five_s{s}_v2_20260908'
        cfg=json.loads((run/'config.resolved.json').read_text())
        assert json.loads((run/'status.json').read_text())['status']=='PASS'
        t=cfg['target_seed'];panel=json.loads((run/'panel.json').read_text())['rows']
        codes=np.load(run/f'seed{t}_codes.npz')['codes']
        decoder_path=Path(cfg['sae_root'])/f'seed_{t}'/'sae.safetensors'
        if t not in cache:cache[t]=load_file(decoder_path)['W_dec']
        for task in cfg['tasks']:
            matrix=np.load(run/f'{task}_participation_export.npz')
            g=matrix['target_participation'];members=np.flatnonzero(np.any(g!=0,axis=1))
            stem=f'{task}_s{s}_t{t}';mp=args.output/(stem+'_map.npz')
            dec=cache[t][members]
            np.savez_compressed(mp,participation=g[members],target_decoder=dec,target_members=members,
                source_members=matrix['source_members'],source_components=matrix['source_components'])
            p=next(p for p in panel if p['task']==task and p['split']=='held_component_development')
            d=panel[p['donor_id']];positions=aligned_positions(p,d,cfg['mode']);pp,qq=np.array(positions).T
            base=codes[p['row_id'],pp][:,members];donor=codes[d['row_id'],qq][:,members]
            sp=args.output/(stem+'_states.npz')
            np.savez_compressed(sp,base_codes=base,aligned_donor_codes=donor,recipient_positions=pp,donor_positions=qq)
            # Preserve exact NumPy reference outputs for five configured controls.
            operations={}
            for name,c in [('whole',[1,1]),('part0',[1,0]),('part1',[0,1]),('half_dose',[.5,.5]),('mixed_dose',[.25,1])]:
                delta,final=participant_delta(base,donor,dec,g[members],np.array(c,dtype=base.dtype))
                operations[name+'_delta']=delta;operations[name+'_final']=final
            np.savez_compressed(args.output/(stem+'_reference.npz'),**operations)
            records.append(dict(source_seed=s,target_seed=t,task=task,source_run=str(run.relative_to(ROOT)),
                map=mp.name,states=sp.name,reference=stem+'_reference.npz',recipient=p,donor=d,
                actual_selected_target_members=len(members),recipient_positions=pp.tolist(),donor_positions=qq.tolist(),
                checkpoint=dict(path=str(decoder_path),sha256=hashlib.sha256(decoder_path.read_bytes()).hexdigest()),
                run_config_sha256=hashlib.sha256((run/'config.resolved.json').read_bytes()).hexdigest()))
    files=[dict(path=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(args.output.iterdir()) if p.is_file()]
    (args.output/'INDEX.json').write_text(json.dumps(dict(records=records,files=files,
        scope='Twenty native maps plus fixed first-hash development examples. Outputs are physical hook deltas, not language-model responses or new scientific confirmation.'),indent=2)+'\n')
    print(json.dumps(dict(maps=len(records),files=len(files),bytes=sum(p['bytes'] for p in files))))


if __name__=='__main__':main()
