from pathlib import Path
from datetime import datetime, timezone
import json


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/final_science_20260921_round05'
BULK=Path('D:/CCAD_Storage/runs/final_science_20260921_round05')


def main():
    template=json.loads((ROOT/'configs/intervention_program_downstream_20260920.json').read_text())
    agreement=json.loads((ROOT/'configs/agreement_reuse_development_r2_20260920.json').read_text())
    records=[]
    for objective in ['local','downstream','distribution']:
        for phase,steps in [('smoke',16),('dev',512)]:
            c=dict(template)
            c.update(run_id=f'FS05_GENERIC_{objective.upper()}_{phase.upper()}_20260921',
                     run_parent='FINAL_SCIENCE_05',run_storage_root=str(BULK),
                     purpose='Test which source-program objective supports later explanations whose members and use contexts are excluded from training.',
                     scope='One target at resid4. Natural-text programs use other source members; identical source actions and 512-update budgets compare local state, final state and vocabulary-distribution supervision.',
                     budget=f'{steps} updates on two 64-token sequences; fixed 32768-member source bank with all held-out member IDs excluded.',
                     budget_seconds=1800,target_seed=3,seeds=[3],train_decoder=True,
                     program_sites=['resid_4'],program_objective=objective,
                     generic_program_steps=steps,program_checkpoints=[steps],log_every=32 if steps>16 else 8,
                     datasets=['grammar'],grammar_rows=8 if phase=='smoke' else 32,
                     grammar_panel=str(OUT/'INFINITIVE_DEVELOPMENT_PANEL.json'),
                     action_bank_size=32768,excluded_source_members={'resid_4':agreement['grammar_members']},
                     natural_cache='D:/CCAD_Storage/runs/final_science_20260920/FINAL01_changes_development_v1_20260920/natural_states.pt',
                     task_adapted_reference=None,source_metric_rows=0,fisher_sequences=0,
                     audit_opened=False,candidate_family_frozen=False,evidence_level='development',
                     executions=['tangent','raw_readout'],variants=[])
            path=ROOT/f'configs/fs05_generic_{objective}_{phase}.json'
            assert not path.exists()
            path.write_text(json.dumps(c,indent=2)+'\n')
            records.append(str(path))
    dest=OUT/'GENERIC_DISTRIBUTION_DESIGN.json'
    assert not dest.exists()
    dest.write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis='Vocabulary-response supervision on other source programs can teach a common dictionary execution rule that generalizes to both infinitive and subject-number explanations.',
        mechanism='Use the probability response induced by source actions to weight errors through the entire later model, replacing the final-hidden-state surrogate while leaving execution unchanged.',
        comparison='Local hidden-state, final hidden-state, vocabulary KL; same source programs, data, model, target seed, optimization and reconstruction term.',
        configs=records),indent=2)+'\n')
    print(json.dumps(dict(configs=len(records),training_sites=['resid_4'],excluded_agreement_members=agreement['grammar_members'])))


if __name__=='__main__':
    main()
