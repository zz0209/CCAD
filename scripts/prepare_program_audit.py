from pathlib import Path
from datetime import datetime, timezone
import json
import argparse


ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    for objective in (['local'] if args.smoke else ['local','downstream','distribution']):
        c=json.loads((ROOT/f'configs/fs05_generic_{objective}_dev.json').read_text())
        c.update(run_id=f'FS05_GENERIC_{objective.upper()}_AUDIT_20260921',program_audit_steps=16,
                 purpose='Separate learning of ordinary source actions from transfer to complete excluded explanations using paired fixed-action evaluation.',
                 budget='Repeat the exact512-update fit with a post-training audit on16matched source actions in each of fit and held natural contexts.',
                 budget_seconds=1800)
        if args.smoke:
            c.update(run_id='FS05_GENERIC_LOCAL_AUDIT_SMOKE_20260921',generic_program_steps=16,
                     program_checkpoints=[16],program_audit_steps=2,grammar_rows=8)
        suffix='_smoke' if args.smoke else ''
        path=ROOT/f'configs/fs05_generic_{objective}_audit{suffix}.json'
        assert not path.exists()
        path.write_text(json.dumps(c,indent=2)+'\n')
    if args.smoke:
        return
    output=ROOT/'artifacts/final_science_20260921_round05/PROGRAM_AUDIT_DESIGN.json'
    assert not output.exists()
    output.write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Determine whether the generic update learns its own intervention family before attributing failure to transfer.',
        audit='Same fixed natural-text actions under initial and trained dictionaries, at local, final-state and vocabulary endpoints. Fit contexts and8excluded natural contexts are reported separately.',
        interpretation='Development diagnosis only. Different requests may use the same document and are dependent. No broader independent function confirmation is claimed.'),indent=2)+'\n')


if __name__=='__main__':
    main()
