"""Executable nonnegative TopK witness of common intervention granularity."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import json,hashlib,sys,time,platform
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
 out=ROOT/'artifacts/core_contribution_20260906/mask_algebra_witness';out.mkdir(exist_ok=False);start=time.perf_counter()
 cfg=dict(seed=260906,dimensions=6,width=12,topk=6,fit_rows=8192,heldout_pairs=4096,rhos=[0,.25,.5,.75,.9],scope='Constructed exact-reconstruction nonnegative TopK dictionaries and fixed injective nonlinear softmax consumer; not trained SAE or natural language evidence')
 (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes())
 rng=np.random.default_rng(cfg['seed']);fit=rng.uniform(.1,1.1,(cfg['fit_rows'],6));rec=rng.uniform(.1,1.1,(cfg['heldout_pairs'],6));don=rng.uniform(.1,1.1,(cfg['heldout_pairs'],6));delta=don-rec
 def lp(x):
  a=np.column_stack([2*np.tanh(x),np.zeros(len(x))]);a-=np.max(a,axis=1)[:,None];return a-np.log(np.exp(a).sum(1))[:,None]
 base=lp(rec);rows=[];checks=[]
 for rho in cfg['rhos']:
  t=np.eye(6);t[0,1]=t[1,0]=rho;t[2,3]=t[3,2]=rho
  d=np.linalg.inv(t);norm=np.linalg.norm(d,axis=0);d=d/norm
  # Six positive preactivations and six zero preactivations: actual TopK6.
  enc=np.column_stack([t.T*norm,np.zeros((6,6))]);dec=np.column_stack([d,np.eye(6)])
  def encode(x):
   pre=np.maximum(x@enc,0);ids=np.argsort(-pre,axis=1,kind='stable')[:,:6];z=np.zeros_like(pre);np.put_along_axis(z,ids,np.take_along_axis(pre,ids,axis=1),axis=1);return z
  zf=encode(fit);zd=encode(don)-encode(rec);zc=zf-zf.mean(0);k=(zc.T@zc/len(fit))*(dec.T@dec);active=np.flatnonzero(np.diag(k)>1e-14)
  recon=float(max(np.abs(zf@dec.T-fit).max(),np.abs(encode(rec)@dec.T-rec).max()));assert recon<1e-12
  checks.append(dict(rho=rho,max_reconstruction_error=recon,max_l0=int(np.count_nonzero(zf,axis=1).max()),minimum_code=float(zf.min()),decoder_norm_error=float(np.max(np.abs(np.linalg.norm(dec,axis=0)-1)))))
  for name,ids in [('fine',[0]),('true_block',[0,1]),('wrong_coarsening',[0,2]),('unmixed_fine',[4])]:
   a=np.zeros(6);a[ids]=1;y=fit*a;yc=y-y.mean(0);h=np.einsum('nj,ni,ij->j',zc,yc,dec)/len(fit);g=np.zeros(12);g[active]=np.linalg.solve(k[active][:,active],h[active]);b=np.linalg.lstsq(zc[:,active],yc,rcond=None)[0]
   refdelta=delta*a;native=(zd*g)@dec.T;readout=zd[:,active]@b;reference=lp(rec+refdelta);p=np.exp(reference);den=float(np.sum(p*(reference-base)));native_lp=lp(rec+native);readout_lp=lp(rec+readout);energy=float(np.sum(refdelta**2))
   rv=float(np.sum((native-refdelta)**2)/energy);kv=float(np.sum(p*(reference-native_lp))/den);rr=float(np.sum((readout-refdelta)**2)/energy)
   row=dict(rho=rho,query=name,source_coordinates=ids,native_weights=g.tolist(),native_relative_hook_error=rv,readout_relative_hook_error=rr,native_relative_kl=kv,readout_relative_kl=float(np.sum(p*(reference-readout_lp))/den),source_kl_sum=den,theoretical_fine_floor=2*rho*rho/(1+6*rho*rho+rho**4) if name=='fine' else None)
   assert rr<1e-25
   if name in ['true_block','unmixed_fine']:assert rv<1e-25 and abs(kv)<1e-10
   if name=='fine' and rho==.5:assert abs(rv-8/41)<.015 and kv>.01
   rows.append(row)
 np.savez_compressed(out/'inputs.npz',fit=fit,recipients=rec,donors=don)
 payload=dict(status='PASS',written_at_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),config=cfg,checks=checks,rows=rows,wall_seconds=time.perf_counter()-start,environment=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (out/'results.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(dict(status='PASS',seconds=payload['wall_seconds'],rho_half=[r for r in rows if r['rho']==.5])))

if __name__=='__main__':main()
