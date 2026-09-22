import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/method_structure_20260922'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {
        'human': ROOT / 'configs/science03_shift_execution_reform_v1.json',
        'grammar': ROOT / 'configs/rg08_grammar_program_development.json',
    }
    inventory = []
    for setting, path in sources.items():
        config = json.loads(path.read_text())
        config.update(
            run_parent='METHOD_STRUCTURE_12',
            run_storage_root='D:/CCAD_Storage/runs/method_structure_20260922',
            purpose='Determine whether program adaptation can preserve the existing target decoder',
            scope='Previously exposed development contexts. Matched 512-step encoder training with fixed decoder and center; saved joint-checkpoint hybrids diagnose parameter-group contributions.',
            evidence_level='controlled_development', audit_opened=False,
            freeze_target_decoder=True, budget_seconds=1000,
            budget='Two settings, two smoke runs and two development runs within 2400 total driver seconds, 14 GB VRAM and 4 GB new bulk.',
            evaluation_parameter_groups={'encoding_hybrid': 'encoder', 'decoding_hybrid': 'decoder'},
        )
        if setting == 'human':
            checkpoint = 'D:/CCAD_Storage/training_curves/SCIENCE03_shift_execution_reform_v1_20260919/tangent_mixed'
            config.update(run_id='MS12_HUMAN_ENCODER_T1_DEV_20260922',
                variants=['tangent_mixed'], continue_training_after_evaluation=True,
                evaluate_checkpoints={name: checkpoint for name in ['joint', 'encoding_hybrid', 'decoding_hybrid']},
                evaluation_execution={name: 'tangent_mixed' for name in ['joint', 'encoding_hybrid', 'decoding_hybrid']})
        else:
            checkpoint = 'D:/CCAD_Storage/runs/reuse_generalization_20260921_round08/RG08_GPT2_PROGRAM_DEV_T2_20260921/program_step512.pt'
            config.update(run_id='MS12_GPT2_ENCODER_T2_DEV_20260922', variants=['program'],
                evaluation_checkpoints={name: dict(path=checkpoint, execution='tangent')
                                        for name in ['joint', 'encoding_hybrid', 'decoding_hybrid']})
        for smoke in [True, False]:
            current = copy.deepcopy(config)
            if smoke:
                current['run_id'] = current['run_id'].replace('_DEV_', '_SMOKE_')
                current.update(steps=8, checkpoint_every=8, budget_seconds=240)
                if setting == 'human':
                    current.update(evaluation_per_cell=2,
                        queries=['pronouns', 'names', 'full', 'interior_00'],
                        evaluate_checkpoints={'encoding_hybrid': checkpoint})
                else:
                    current.update(eval_pairs_per_task=2,
                        evaluation_checkpoints={'encoding_hybrid': dict(path=checkpoint, execution='tangent')})
            if setting == 'human':
                current['bulk_output_dir'] = current['run_storage_root'] + '/' + current['run_id'] + '_checkpoints'
            destination = args.output / (setting + ('_smoke.json' if smoke else '_development.json'))
            assert not destination.exists(), destination
            destination.write_text(json.dumps(current, indent=2) + '\n')
            inventory.append(dict(setting=setting, smoke=smoke, config=destination.as_posix(),
                run_id=current['run_id'], parent_config=path.as_posix(),
                parent_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                config_sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
    (args.output / 'CONFIG_INVENTORY.json').write_text(json.dumps(dict(
        written_at_utc=datetime.now(timezone.utc).isoformat(), configs=inventory), indent=2) + '\n')
    print(json.dumps(inventory, indent=2))


if __name__ == '__main__':
    main()
