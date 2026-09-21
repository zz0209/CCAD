from pathlib import Path
from datetime import datetime, timezone
import json


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/reuse_generalization_20260921_round03'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    c=json.loads((ROOT/'configs/rg01_active_infinitive_development.json').read_text())
    reference=json.loads((ROOT/'configs/intervention_inverse_screen_20260920.json').read_text())['task_adapted_reference']
    c.update(run_parent='REUSE_GENERALIZATION_03',run_storage_root='D:/CCAD_Storage/runs/reuse_generalization_20260921_round03',
        purpose='Isolate request-conditioned coefficient feasibility in complete human and grammatical programs.',
        scope='Exposed development contexts. Same source information, support and dictionary; request-unbounded is a physical diagnostic.',
        budget='7200driver seconds for this scientific unit,13GB allocated GPU and8GB bulk.',
        datasets=['human','grammar'],target_seed=2,seeds=[2],human_per_cell=4,grammar_rows=32,
        eval_batch_size=2,query_progress=True,save_pooled=True,budget_seconds=2400,
        executions=['tangent','refined_common','request_fixed','request_unbounded','inverse_budget','raw_readout'],
        generic_program_steps=0,program_audit_steps=0,program_parameters='joint',source_metric_rows=0,
        source_metric_cache=None,other_source_metric_cache=None,fisher_sequences=0,task_adapted_reference=reference)
    for name,small in [('smoke',True),('development',False)]:
        conf=dict(c,run_id=f'RG03_REQUEST_REALIZATION_{name.upper()}_T2_20260921')
        if small:
            conf.update(human_per_cell=1,grammar_rows=4,budget_seconds=600,
                executions=['refined_common','request_fixed','request_unbounded'])
            conf['human_queries']={k:v for k,v in c['human_queries'].items() if k in ['full','center','pronouns']}
            conf['human_member_queries']={}
        path=ROOT/f'configs/rg03_request_realization_{name}.json'
        if path.exists():raise FileExistsError(path)
        path.write_text(json.dumps(conf,indent=2)+'\n')
    (OUT/'DESIGN.json').write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        question='Does shared request-linearity restrict complete finite functional reuse under the original support?',
        historical='Eight-state allocation diagnostic and request-specific inverse already exist. This compares full actual responses and all11human sites.',
        decision='Only a shared finite-response improvement warrants expansion. Geometry improvement alone is insufficient.',
        data='Previously exposed IR04 and FS05development panels. RG02confirmation remains excluded.'),indent=2)+'\n')


if __name__=='__main__':
    main()
