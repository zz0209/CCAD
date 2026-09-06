"""Read existing source-only native probabilities to develop a lexical prediction."""
import hashlib,json,os,time
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
def main():
 out=ROOT/'artifacts/that_prediction_20260906';out.mkdir(exist_ok=True)
 parent=ROOT/'runs/NATIVE_that_roles_s1a2379_v1_20260906';cfg=json.loads((parent/'config.resolved.json').read_text());asset=json.loads((ROOT/cfg['parent_asset_config']).read_text())
 tok=AutoTokenizer.from_pretrained(asset['model_local_dir'],local_files_only=True)
 groups={'personal':[' I',' we',' you',' he',' she',' it',' they'],'existential':[' there']}
 ids={k:[tok.encode(t,add_special_tokens=False) for t in v] for k,v in groups.items()};assert all(len(t)==1 for v in ids.values() for t in v);ids={k:[t[0] for t in v] for k,v in ids.items()}
 rows=[json.loads(s) for s in (parent/'metrics.raw.jsonl').read_text().splitlines()];p=np.load(parent/'probabilities.npz');results=[]
 for i,r in enumerate(rows):
  def contrast(op):
   a=p[f'case_{i}_{op}'].astype(float);return float(np.log(a[ids['personal']].sum()/a[ids['existential']].sum()))
  results.append(dict(subject=r['subject'],activation=r['activation'],remove=contrast('native_remove')-contrast('baseline'),add=contrast('native_add')-contrast('baseline')))
 files=[parent/'probabilities.npz',parent/'metrics.raw.jsonl',Path(__file__)]
 payload=dict(scope='Post-hoc source-only development from12old authored contexts. Contrast suggested by previously exposed top changes; no confirmatory claim or target outcomes.',groups=groups,token_ids=ids,results=results,inputs=[dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files])
 (out/'output_development.json').write_text(json.dumps(payload,indent=2)+'\n')
 print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
