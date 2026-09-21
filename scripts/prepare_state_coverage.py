from pathlib import Path
import argparse
import json


ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--writer',action='store_true')
    args=parser.parse_args()
    for phase,steps in [('smoke',16),('development',4096)]:
        c=json.loads((ROOT/'configs/rg01_local_parts.json').read_text())
        label='state_writer' if args.writer else 'state_parts'
        c.update(run_id=f'RG01_{label.upper()}_{phase.upper()}_20260921',
                 purpose='Test whether state-conditioned source action coverage improves common execution for whole excluded explanations.',
                 scope='Four active source members per token, eight target members per token. Same natural sequences, target initialization, update budget and column objective as batch-global selection. Source directions and intervention energy change with this sampling intervention.',
                 program_action_sampling='state',generic_program_steps=steps,
                 program_checkpoints=[16] if phase=='smoke' else [512,4096],
                 program_audit_steps=2 if phase=='smoke' else 32,grammar_rows=8 if phase=='smoke' else 32)
        if args.writer:
            c.update(program_parameters='writer',purpose='Test whether a separate action writer preserves generic action learning and complete excluded explanations while keeping the natural dictionary fixed.',
                     scope='Same state-conditioned actions, contexts, source information and4096updates. Target encoder/decoder fixed, independent action matrix initialized from its encoder. Natural reconstruction remains unchanged.')
        path=ROOT/f'configs/rg01_{label}_{phase}.json'
        assert not path.exists()
        path.write_text(json.dumps(c,indent=2)+'\n')


if __name__=='__main__':
    main()
