"""Prepare fresh biographies for the frozen conditional-restoration experiment."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--freeze', type=Path, required=True)
    a = p.parse_args()
    spec = json.loads(a.freeze.read_text())
    output = Path(spec['panel'])
    if output.exists():
        raise FileExistsError(output)
    for item in spec['code_identities']:
        assert sha(item['path']) == item['sha256'], item['path']
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(spec['model_local_dir'], local_files_only=True)
    exposed = set()
    exclusions = []
    for path in spec['exclude_panels']:
        rows = json.loads(Path(path).read_text(encoding='utf-8'))['rows']
        exposed.update(r['document_sha256'] for r in rows)
        exclusions.append(dict(path=path, sha256=sha(path), rows=len(rows)))
    candidates = {(y, g): [] for y in [0, 1] for g in [0, 1]}
    seen = set(exposed)
    counts = {}
    for split in spec['split_priority']:
        path = next(Path(spec['data_directory']).glob(split+'*.parquet'))
        table = pq.read_table(path, filters=[('profession', 'in', [13, 21])],
                              columns=['hard_text', 'profession', 'gender'])
        duplicate = 0
        for row in table.to_pylist():
            text = row['hard_text']
            identity = hashlib.sha256(text.encode()).hexdigest()
            if identity in seen:
                duplicate += 1
                continue
            seen.add(identity)
            y, g = int(row['profession'] == 13), row['gender']
            candidates[y, g].append(dict(text=text, document_sha256=identity,
                label=y, gender=g, original_split=split, split=spec.get('evaluation_split','restoration_confirmation')))
        counts[split] = dict(path=str(path), sha256=sha(path), excluded_or_duplicate=duplicate)
    rows, too_long = [], []
    priority = {split: i for i, split in enumerate(spec['split_priority'])}
    for key, bucket in candidates.items():
        bucket.sort(key=lambda row: (priority[row['original_split']], row['document_sha256']))
        selected = 0
        for row in bucket:
            tokens = tokenizer(row['text'], add_special_tokens=True, truncation=False)['input_ids']
            if len(tokens) > spec['max_length']:
                too_long.append(row['document_sha256'])
                continue
            row['tokens'] = tokens
            row['row_id'] = len(rows)
            rows.append(row)
            selected += 1
            if selected == spec['per_cell']:
                break
        assert selected == spec['per_cell'], (key, selected)
    assert len({r['document_sha256'] for r in rows}) == len(rows)
    assert not ({r['document_sha256'] for r in rows} & exposed)
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), rows=rows,
        freeze_sha256=sha(a.freeze), exclusions=exclusions, inputs=counts,
        overlength_exclusions=too_long, truncated=0, model_outputs_computed=False,
        source_probe_refitted=False, selection=spec['selection'],
        cell_split_counts={str(k): v for k, v in Counter(
            (r['label'], r['gender'], r['original_split']) for r in rows).items()})
    output.write_text(json.dumps(result, indent=2)+'\n')
    request_path=None
    if spec.get('development_request_spec'):
        request = json.loads(Path(spec['development_request_spec']).read_text())
        request.update(written_at_utc=datetime.now(timezone.utc).isoformat(),
            evidence_level='frozen_new_document_confirmation',
            documents=[r['document_sha256'] for r in rows],
            context_split=['confirmation']*len(rows), selection=spec['selection'])
        request_path = Path(spec['confirmation_request_spec'])
        with request_path.open('x') as f:
            json.dump(request, f, indent=2)
    receipt = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        panel=str(output), panel_sha256=sha(output), rows=len(rows),
        request_spec=str(request_path) if request_path else None, request_sha256=sha(request_path) if request_path else None,
        cell_split_counts=result['cell_split_counts'], overlength=len(too_long),
        prior_document_overlap=0, model_outputs_computed=False)
    with output.with_name(spec.get('receipt_name','R05_CONFIRMATION_DATA_RECEIPT.json')).open('x') as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
