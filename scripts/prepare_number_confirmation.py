from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
from collections import defaultdict
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/reuse_generalization_20260921_round03'


def save(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    config = json.loads((ROOT/'configs/rg03_trajectory_development.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(config['model_local_dir'], local_files_only=True)
    manifest = json.loads((OUT/'SFC_TEST_SOURCE_MANIFEST.json').read_text(encoding='utf-8-sig'))
    excluded = set()
    prior = list((ROOT/'artifacts/final_science_20260920_round02').glob('AGREEMENT_*.json'))
    for path in prior:
        for row in json.loads(path.read_text()).get('rows', []):
            excluded.add(row['clean_prefix'])
    selected = []
    source_counts = {}
    for asset in manifest:
        path = Path(asset['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != asset['sha256']:
            raise ValueError('Source identity mismatch')
        structure = path.name.removesuffix('_test.json')
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records.sort(key=lambda row:hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest())
        unique = {}
        for row in records:
            text = row['clean_prefix']
            if text in excluded or text in unique:
                continue
            clean = tokenizer.encode(text, add_special_tokens=False)
            patch = tokenizer.encode(row['patch_prefix'], add_special_tokens=False)
            answers = [tokenizer.encode(row[key], add_special_tokens=False)
                       for key in ['clean_answer', 'patch_answer']]
            if len(clean) != len(patch) or any(len(answer) != 1 for answer in answers):
                continue
            digest = hashlib.sha256(text.encode()).hexdigest()
            pair = json.dumps(sorted([text, row['patch_prefix']]))
            unique[text] = dict(row, text=text, structure=structure, document_sha256=digest,
                                pair_sha256=hashlib.sha256(pair.encode()).hexdigest())
        cells = defaultdict(list)
        for row in unique.values():
            cells[row['case']].append(row)
        count = 4 if structure == 'simple' else 8
        source_counts[structure] = {key:len(value) for key,value in cells.items()}
        for key, rows in sorted(cells.items()):
            if len(rows) < count:
                raise ValueError((structure, key, len(rows), count))
            chosen = sorted(rows, key=lambda row:hashlib.sha256(('rg03number/'+row['document_sha256']).encode()).hexdigest())[:count]
            selected.extend(chosen)
    if len(selected) != 64 or len({row['document_sha256'] for row in selected}) != 64:
        raise ValueError(dict(rows=len(selected),unique_prefixes=len({row['document_sha256'] for row in selected}),cells=source_counts))
    panel = OUT/'NUMBER_CONFIRMATION_PANEL.json'
    save(panel, dict(rows=selected, excluded_prefixes=len(excluded), available_by_case=source_counts,
                     sampling='One answer pair per clean prefix, hash selected within original grammar case; no model outputs used.',
                     source='Official SFC within-RC and across-RC test prefixes. Simple-test prefixes already occur in the recorded earlier panels and contribute no new context.'))
    paths = []
    for seed in [3,4,5]:
        settings = dict(base_config='configs/rg03_trajectory_development.json',
            run_id=f'RG03_NUMBER_TRAJECTORY_CONFIRMATION_T{seed}_20260921',device='cuda:0',
            generator_script='scripts/check_number_program.py',target_seed=seed,seeds=[seed],
            purpose='Confirm the same state-feedback rule on a second annotated multi-site program.',
            scope='64fresh prefixes from the official within-RC and across-RC test splits,58fixed annotated members, three fixed target dictionaries and three named requests. Source remains available.',
            evidence_level='frozen_new_context_confirmation', candidate_family_frozen=True,
            annotations='artifacts/morning_reform_20260916/independent_circuit_reading/annotations/10_32768.jsonl',
            number_panel=str(panel),per_structure=0,
            target_methods=['local_action','recorded_action','state_feedback','raw_readout'],
            budget_seconds=650,
            statistics_unit='Paired prefix pairs, clustered by original clean/patch pair and grammar structure; source, request family and targets fixed.')
        path = ROOT/f'configs/rg03_number_trajectory_confirmation_t{seed}.json'
        save(path, settings)
        paths.append(path)
    assets = [panel,*paths,ROOT/'configs/rg03_trajectory_development.json',
              ROOT/'scripts/check_number_program.py',ROOT/'scripts/analyze_number_confirmation.py',
              ROOT/'scripts/prepare_number_confirmation.py',ROOT/'scripts/adaptive_native_execution.py',
              Path(config['run_storage_root'])/'RG03_NUMBER_SOURCE_DEVELOPMENT_20260921/source_parameters.npz',
              Path(config['run_storage_root'])/'RG03_NUMBER_SOURCE_DEVELOPMENT_20260921/source_identity.json',
              Path(config['run_storage_root'])/'RG03_NUMBER_SOURCE_DEVELOPMENT_20260921/source_design.json',
              OUT/'SFC_TEST_SOURCE_MANIFEST.json',
              ROOT/'artifacts/morning_reform_20260916/independent_circuit_reading/annotations/10_32768.jsonl']
    save(OUT/'NUMBER_CONFIRMATION_FREEZE.json', dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        primary='Mean source-effect-normalized RMSE across singular,plural,full requests and three fixed targets; state_feedback versus local_action.',
        secondary='Recorded-action and source-readout contrasts, individual requests and the three grammar structures.',
        statistics='2000 paired clean/patch-pair cluster bootstrap draws within structure, shared across methods and targets. Fixed source and request family.',
        identities={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in assets},
        target_seeds=[3,4,5], source_definition='All58published singular/plural annotations at10existing sites; selected before development outcomes.',
        source_reference_run=str(Path(config['run_storage_root'])/'RG03_NUMBER_SOURCE_DEVELOPMENT_20260921'),
        uses_target_response_training=False))
    print(json.dumps(dict(rows=len(selected), cells=source_counts, configs=list(map(str,paths)))))


if __name__ == '__main__':
    main()
