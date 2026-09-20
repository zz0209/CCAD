import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/final_science_20260920_round03'


def identity(path):
    return dict(path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    destination = ART/'READOUT_CORRECTION_FREEZE.json'
    assert not destination.exists()
    assert (ART/'CONFIRMATION_ANALYSIS.json').is_file()
    configs = []
    inputs = [ART/name for name in ['CONFIRMATION_ANALYSIS.json', 'CONFIRMATION_ARRAYS.npz',
        'CONFIRMATION_PANEL.json', 'CONFIRMATION_FREEZE.json', 'CONTROL_ASSET_CHECK.json', 'READOUT_CORRECTION.md']]
    for source in range(1, 6):
        cfg = json.loads((ROOT/f'configs/final_science03_qwen_confirm_s{source}.json').read_text())
        original_run = Path(cfg['run_storage_root'])/cfg['run_id']
        assert json.loads((original_run/'status.json').read_text())['status'] == 'PASS'
        cfg['run_id'] = f'FS03_QWEN_READOUT_CORRECT_S{source}_T{source%5+1}_20260920'
        cfg['purpose'] = 'Execute the complete inherited source-direction readout on the unchanged frozen panel'
        cfg['budget_seconds'] = 400
        spec = cfg['source_field_evaluation']
        spec['methods'] = ['readout']
        if source != 1:
            spec['profile_cache'] = (original_run/'source_profile.pt').as_posix()
        assert Path(spec['profile_cache']).is_file()
        inputs.append(Path(spec['profile_cache']))
        path = ROOT/f'configs/final_science03_qwen_readout_s{source}.json'
        assert not path.exists()
        path.write_text(json.dumps(cfg, indent=2)+'\n')
        configs.append(identity(path))
    code = [ROOT/p for p in ['scripts/run_arithmetic_digit_components.py', 'scripts/arithmetic_source_fields.py',
        'src/ccad/source_field_inference.py', 'src/ccad/request_inference.py',
        'src/ccad/intervention_transport.py', 'scripts/analyze_qwen_readout_correction.py',
        'scripts/freeze_qwen_readout_correction.py']]
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), configurations=configs,
        inputs=[identity(p) for p in inputs], code=[identity(p) for p in code],
        intervention='Use the readout asset target indices for its target-code differences and all fitted coefficients.',
        unchanged='The question and request panels, source functions, model, SAE checkpoints, fitted readout and primary contrast remain fixed.',
        analysis='Replace only readout predictions; verify unchanged primary point estimates and intervals exactly.')
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(configurations=len(configs), freeze=destination.as_posix())))


if __name__ == '__main__':
    main()
