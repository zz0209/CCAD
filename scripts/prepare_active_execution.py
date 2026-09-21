from pathlib import Path
import json


ROOT=Path(__file__).resolve().parents[1]


def main():
    for dataset,base in [('infinitive','rg01_state_writer_development'),('agreement','rg01_state_writer_agreement_4096')]:
        for phase in ['smoke','development']:
            c=json.loads((ROOT/f'configs/{base}.json').read_text())
            for key in ['task_adapted_reference','program_writer_reference','transfer_training_run']:
                c.pop(key,None)
            c.update(run_id=f'RG01_ACTIVE_EXECUTION_{dataset.upper()}_{phase.upper()}_20260921',
                purpose='Measure whether current target members or support chosen through the encoder limit explanation-independent execution.',
                scope='Frozen dictionaries and source definitions. Matched eight or twenty target members. Decoder-based shared pursuit is standard constrained inference; no target responses fit its coefficients.',
                executions=['tangent','refined_common','pursuit_common','active_pursuit_common','raw_readout'],
                generic_program_steps=0,program_audit_steps=0,variants=[],
                grammar_rows=4 if phase=='smoke' else 32 if dataset=='infinitive' else 48,
                budget_seconds=600,inverse_steps=128,inverse_candidates=128,
                accelerate_refinement=True)
            path=ROOT/f'configs/rg01_active_{dataset}_{phase}.json'
            assert not path.exists()
            path.write_text(json.dumps(c,indent=2)+'\n')


if __name__=='__main__':
    main()
