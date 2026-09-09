"""Verify retained compiled operators on actual natural target-code pairs.

This is an artifact/interface replay, not a new functional experiment. It
compares the portable result with dense code arithmetic and the exact target
decoder checkpoint. Outputs retain all discrepancies, never tune the operator.
"""
from pathlib import Path
import argparse,json,hashlib,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ccad.compiled_axis import CompiledAxis

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
 root=Path(__file__).resolve().parents[1];cfg=json.loads((a.run/'config.resolved.json').read_text());ckpts=json.loads((root/cfg['training_run']/'checkpoints.json').read_text())['checkpoints'];decoders={};records=[]
 import torch
 torch.set_num_threads(2)
 for path in sorted(a.run.glob('*_compiled_functional_axis_operator.npz')):
  query=path.name.removesuffix('_compiled_functional_axis_operator.npz');target=int(query.rsplit('_t',1)[1]);objective='matryoshka' if '_matryoshka_' in query else 'topk';key=(objective,target)
  if key not in decoders:
   snap=next(r for r in ckpts if r['step']==cfg['checkpoint_step'] and r.get('objective',objective)==objective and r['seed']==target)
   if cfg.get('sae_loader')=='sparsify':
    from safetensors.torch import load_file
    checkpoint=Path(snap['path'])/'sae.safetensors';state=load_file(str(checkpoint),device='cpu');decoder=state['W_dec'].numpy()
   else:
    checkpoint=Path(snap['path']);state=torch.load(checkpoint,map_location='cpu',weights_only=True);decoder=(state['decoder.weight'].T if objective=='topk' else state['W_dec']).numpy()
   assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==snap['sha256'];decoders[key]=decoder.copy()
  dec=decoders[key].astype(np.float64)
  with np.load(a.run/f'natural_calibration_{objective}_seed{target}.npz') as z:
   codes=np.zeros((65,int(z['shape'][1])),dtype=np.float64);valid=z['rows']<65;codes[z['rows'][valid],z['columns'][valid]]=z['values'][valid]
  base=codes[:32];donor=codes[32:64];base=np.concatenate([base,codes[64:65]]);donor=np.concatenate([donor,codes[64:65]])
  operator=CompiledAxis.load(path);r=operator.apply(base,donor,decoder=dec);final=base.copy()
  for row in range(len(base)):final[row,r['members'][row]]+=r['code_increment'][row]
  dense=(final-base)@dec;difference=float(np.max(np.abs(r['hidden_delta']-dense)));assert difference<1e-10
  assert (final>=0).all() and (r['code_increment']>=0).all() and r['changed_members'].max()<=operator.members.shape[1]
  assert not r['changed_members'][-1] and np.array_equal(r['hidden_delta'][-1],np.zeros(dec.shape[1]))
  reconstruction=np.einsum('sb,sbd->sd',operator.weights.astype(np.float64),dec[operator.members]);unit_error=float(np.max(np.abs(reconstruction-operator.directions)))
  rec=dict(query=query,operator_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),actual_natural_code_pairs=len(base),zero_change_pairs=1,minimum_final_code=float(final.min()),max_actual_changed=int(r['changed_members'].max()),dense_native_replay_max_abs=difference,stored_direction_float32_max_abs=unit_error,cached_delta_max_abs=float(np.max(np.abs(r['cached_hidden_delta']-r['hidden_delta']))));records.append(rec)
  if len(records)==1:
   fixture=a.output/'target_codes_example.npz';np.savez_compressed(fixture,base_codes=base,donor_codes=donor)
   import subprocess
   subprocess.run([sys.executable,str(root/'scripts/apply_compiled_axis.py'),'--operator',str(path),'--input',str(fixture),'--output',str(a.output/'portable_output.npz')],check=True)
 result=dict(run=str(a.run),operators=len(records),records=records,wall_seconds=time.perf_counter()-start,scope='Same-machine artifact/application replay with actual natural target codes and retained target decoder identity. No new task labels, source fitting, LM continuation or clean-machine experiment reproduction.')
 (a.output/'REPLAY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
