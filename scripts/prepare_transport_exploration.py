from pathlib import Path
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['pilot', 'development'], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = json.loads((root / 'configs/science03_infinitive_execution_reform_v1.json').read_text())
    pilot = args.stage == 'pilot'
    base.update(
        run_id=f'TRANSPORT01_infinitive_{args.stage}_v1_20260919',
        run_parent='TRANSPORT_EXPLORATION_01',
        purpose='Separate intervention transport from natural SAE encoding; compare shared active and unrestricted target-member routing',
        run_storage_root='D:/CCAD_Storage/runs/transport_exploration_20260919',
        variants=['transport_active_mixed', 'transport_open_mixed'],
        steps=8 if pilot else 512,
        baselines=['native_tangent_relation_8', 'raw_reconstruction'],
        gain_before_selection=True,
        budget_seconds=300 if pilot else 700,
        budget='Standalone exploration; at most700driver seconds for two512-update independent transport arms; existing GPU/environment/data',
        scope='Existing exposed infinitive development contexts; fixed public four-member source, source-only fit64 contexts and one target seed2. No independent confirmation.',
        audit_opened=False,
        candidate_family_frozen=False,
        evidence_level='controlled_development',
    )
    if pilot:
        panel = json.loads((root / base['panel']).read_text())
        panel['rows'] = panel['rows'][:8]
        panel['queries'] = dict(list(panel['queries'].items())[:3])
        location = root / 'artifacts/transport_exploration_20260919/PILOT_PANEL.json'
        location.write_text(json.dumps(panel, indent=2), encoding='utf-8')
        base['panel'] = str(location)
        base.pop('request_panel_sha256', None)
    else:
        base['variants'].insert(0, 'tangent_gain')
        previous = 'D:/CCAD_Storage/runs/science_upgrade_20260919/SCIENCE03_infinitive_execution_reform_v1_20260919'
        base['evaluate_checkpoints'] = {'tangent_mixed': previous + '/tangent_mixed'}
    location = root / f'configs/transport01_infinitive_{args.stage}_v1.json'
    location.write_text(json.dumps(base, indent=2), encoding='utf-8')
    print(location)


if __name__ == '__main__':
    main()
