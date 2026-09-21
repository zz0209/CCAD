from pathlib import Path
import argparse
import json


ROOT=Path(__file__).resolve().parents[1]
BULK=Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round01')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    for objective in (['local_parts'] if args.smoke else ['local','local_parts']):
        c=json.loads((ROOT/'configs/fs05_generic_local_audit.json').read_text())
        label=objective.upper()+('_SMOKE' if args.smoke else '')
        steps=16 if args.smoke else 4096
        c.update(run_id=f'RG01_{label}_20260921',run_parent='REUSE_GENERALIZATION_01',run_storage_root=str(BULK),
                 purpose='Compare whole-column supervision and aggregate supervision for reusable ordinary source actions and two excluded explanations.',
                 scope='Same four source columns, natural contexts, target initialization, optimizer and updates. Fixed natural-action and whole-explanation evaluation.',
                 budget='Matched learning curves at512and4096updates; representative16-update smoke.',
                 generic_program_steps=steps,program_checkpoints=[16] if args.smoke else [512,4096],
                 program_objective=objective,program_audit_steps=2 if args.smoke else 32,program_audit_alternate_parts=True,
                 budget_seconds=3600,grammar_rows=8 if args.smoke else 32,log_every=256)
        path=ROOT/f'configs/rg01_{objective}{"_smoke" if args.smoke else ""}.json'
        assert not path.exists()
        path.write_text(json.dumps(c,indent=2)+'\n')


if __name__=='__main__':
    main()
