"""Bind frozen maps to independently prepared natural prefixes and next tokens."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=ROOT;out=r/'artifacts/fisher_refit_20260906';freeze=json.loads((out/'freeze.json').read_text());fit=r/'runs/F4_fisher_refit_s1_v1_20260906';assert sha(fit/'coefficients.npz')==freeze['coefficients_sha256'];corpus=r/'runs/F4_fisher_refit_corpus_v1_20260906';assert json.loads((corpus/'status.json').read_text())['status']=='PASS'
 ts=np.fromfile(corpus/'artifacts/calibration.uint16.bin',dtype='<u2').reshape(-1,128);prep=json.loads((out/'prepared_inputs.json').read_text());old_docs={c['document_id'] for p in ['artifacts/colon_natural_20260906/prepared_inputs.json','artifacts/colon_cohort_20260906/prepared_inputs.json'] for c in json.loads((r/p).read_text())['cases']};assert not old_docs&{c['document_id'] for c in prep['cases']}
 for c in prep['cases']:
  si,pos=c['sequence_index'],c['token_index'];assert c['token_ids']==ts[si,:pos+1].tolist();c['observed_next_token_id']=int(ts[si,pos+1]) if pos<127 else None
 dest=out/'prepared_inputs_with_next.json';dest.write_text(json.dumps(prep,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 c=json.loads((fit/'config.resolved.json').read_text());c.update(run_id='F4_fisher_refit_apply_s1_v1_20260906',purpose='Frozen operation-weighted maps on new natural document pairs',evaluation_inputs=dest.relative_to(r).as_posix(),evaluation_inputs_sha256=sha(dest),frozen_map=dict(path=str(fit),config_sha256=freeze['config_sha256'],coefficients_sha256=freeze['coefficients_sha256']),confirmation_inputs=[dict(path=str(out/'freeze.json'),sha256=sha(out/'freeze.json')),dict(path=str(out/'selection.json'),sha256=sha(out/'selection.json')),dict(path=str(corpus/'artifacts/documents.jsonl'),sha256=sha(corpus/'artifacts/documents.jsonl'))],evidence_level='fresh_document_frozen_method_single_direction',scope=freeze['scope'],budget_seconds=1200,budget='At most24prefixes*(1+4*(1+11methods))+1=1177LMforwards; no refit, newtraining or audit.')
 (r/'configs/f4_fisher_refit_apply_s1_v1.json').write_text(json.dumps(c,indent=2)+'\n');print(json.dumps(dict(cases=len(prep['cases']),next_token_count=sum(c['observed_next_token_id'] is not None for c in prep['cases']),inputs_sha256=sha(dest))))
if __name__=='__main__':main()
