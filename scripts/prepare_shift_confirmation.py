"""Materialize the predeclared official-test panel after recording the fixed protocol."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
from run_shift_explanation import select_rows

ROOT = Path(__file__).resolve().parents[1]


def stamp():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads(args.freeze.read_text(encoding='utf-8'))
    output = Path(spec['panel_path'])
    assert not output.exists(), 'Do not overwrite an exposed confirmation panel.'
    for item in spec['code_identities']:
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
    assert sha(spec['test_data']) == spec['test_sha256']
    events = {'freeze_sha256': sha(args.freeze), 'test_bytes_hashed_at_utc': stamp()}
    frozen_panel = json.loads(Path(spec['source_panel']).read_text(encoding='utf-8'))
    exposed = {r['document_sha256'] for r in frozen_panel['rows']}
    events['test_rows_materialization_started_at_utc'] = stamp()
    rows, counts = select_rows(spec['test_data'], 'test_confirmation', False, 42)
    retained, duplicates, seen = [], [], set()
    for row in rows:
        identity = row['document_sha256']
        if identity in exposed or identity in seen:
            duplicates.append(identity)
            continue
        seen.add(identity)
        retained.append(row)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(spec['model_local_dir'], local_files_only=True)
    tokens = tokenizer([r['text'] for r in retained], add_special_tokens=True, truncation=False)['input_ids']
    truncated = 0
    for i, (row, ids) in enumerate(zip(retained, tokens)):
        row['row_id'] = i
        row['tokens'] = ids[:spec['max_length']]
        truncated += len(ids) > spec['max_length']
    events['test_panel_written_at_utc'] = stamp()
    data = dict(rows=retained, raw_group_counts=counts, selected_before_exclusion=len(rows),
                excluded_document_hashes=duplicates, selected_after_exclusion=len(retained),
                max_length=spec['max_length'], truncated=truncated,
                selection='Original first-minimum balancing of professions 13/21 and both genders, Random(42) ordering; exclude documents already used by the fixed source probe or CCAD development, and repeated test documents.',
                frozen_rules=events, source_probe_refitted=False, model_outputs_computed=False)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    receipt = dict(events, panel_path=str(output), panel_sha256=sha(output),
                   selected_before_exclusion=len(rows), selected_after_exclusion=len(retained),
                   exclusions=len(duplicates), truncated=truncated,
                   source_probe_refitted=False, model_outputs_computed=False)
    output.with_name('R59_CONFIRMATION_PANEL_RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
