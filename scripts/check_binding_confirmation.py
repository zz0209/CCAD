"""Focused identity, operation-budget and independent whole-answer tally check."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    run=ROOT/'runs/REFORM_R41_qwen_binding_query_confirmation_v1_20260914'
    art=ROOT/'artifacts/correspondence_reform_20260913'
    status=json.loads((run/'status.json').read_text());assert status['status']=='PASS'
    cfg=json.loads((run/'config.resolved.json').read_text());panel=json.loads((run/'panel.json').read_text());rows=panel['rows']
    dev=json.loads((ROOT/'runs/REFORM_R41_qwen_binding_sources_v4_20260914/panel.json').read_text())
    assert [r for r in rows if r['split']=='fit']==[r for r in dev['rows'] if r['split']=='fit']
    prompts={r['prompt'] for r in rows if r['split']!='fit'}
    for prior in cfg['exclude_evaluation_runs']:
        previous=json.loads((ROOT/prior/'panel.json').read_text())
        assert not prompts.intersection(r['prompt'] for r in previous['rows'])
    budgets=[]
    for p in sorted(run.glob('binding_relation_s*_t*.npz')):
        with np.load(p) as a:
            for k in ['member_country','member_static','query_member_country','union_member_country']:
                v=a[k];assert np.isfinite(v).all() and v.min()>=0 and v.max()<=1+1e-6
                assert np.count_nonzero(v,axis=-1).max()<=64
            assert np.count_nonzero(a['source_query'],axis=-1).min()==64
            assert np.count_nonzero(a['source_query'],axis=-1).max()==64
            budgets.append(dict(path=str(p.relative_to(ROOT)),source_members=len(a['source_indices']),target_bank=len(a['target_indices']),
                maximum_native_weight=float(a['union_member_country'].max()),max_active=int(np.count_nonzero(a['union_member_country'],axis=-1).max())))
    raw=[json.loads(r) for r in (run/'metrics.raw.jsonl').read_text().splitlines()];groups={}
    for r in raw:
        if r['kind']=='intervention':
            key=(r['method'],r['seed'],r['component'],r['task'],r['operation'])
            groups.setdefault(key,[]).append(r)
    tally={}
    for key,pair in groups.items():
        assert len(pair)==2 and {r['query'] for r in pair}=={0,1}
        ok=all(r['answer_id']==r['expected_id'] for r in pair)
        tally.setdefault(key[0],[]).append(ok)
    summary=json.loads((art/'r41_binding_query_confirmation.json').read_text())
    for cell in summary['macro']:
        if cell['operation']=='macro':assert abs(np.mean(tally[cell['method']])-cell['complete_binding_accuracy'])<1e-12
    out=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),status='PASS',unchanged_fit_contexts=64,
        new_eval_contexts=64,exact_prompt_exclusions=len(cfg['exclude_evaluation_runs']),budgets=budgets,
        independent_whole_answer_tally={k:float(np.mean(v)) for k,v in tally.items()},
        scope='Checks data identity, native per-token bounds and actual answer-pair arithmetic; does not establish novelty, semantic uniqueness or generalization beyond the stated panel.',
        raw_sha256=hashlib.sha256((run/'metrics.raw.jsonl').read_bytes()).hexdigest())
    (art/'R41_FOCUSED_CHECK.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))


if __name__=='__main__':main()
