import argparse
from bisect import bisect_right
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from transformers import AutoTokenizer


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def stratum(tokenizer, token, objective):
    text = tokenizer.decode([int(token)], clean_up_tokenization_spaces=False).strip()
    if objective == 'topk':
        if text in {'the', 'The'}:
            return 'article'
        if text.lower() in {'my', 'your', 'his', 'her', 'its', 'our', 'their'} and text in {text.lower(), text.capitalize()}:
            return 'possessive'
        if text == 'I' or (text.lower() in {'it', 'he', 'she', 'they', 'them', 'we', 'you'} and text in {text.lower(), text.capitalize()}):
            return 'pronoun'
        return 'other'
    if int(token) == 447:
        return 'byte447'
    if int(token) == 564:
        return 'byte564'
    if text in {"'", '"', '-', '(', ')'}:
        return 'ascii_punctuation'
    return 'other'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--material', type=Path, required=True)
    parser.add_argument('--split', choices=['calibration', 'audit'], required=True)
    parser.add_argument('--positions', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('D:/CCAD_Storage/models/gpt2-medium/6dcaa7a952f72f9298047fd5137cd6e4f05f41da'))
    parser.add_argument('--per-stratum', type=int, default=32)
    args = parser.parse_args()
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.material/'TOKEN_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    info = manifest['outputs'][args.split]
    token_path = Path(info['path'])
    assert digest(token_path) == info['sha256']
    tokens = np.fromfile(token_path, dtype='<u2')
    assert len(tokens) % 128 == 0
    record_path = args.material/'document_prefix_records.json'
    records = sorted((r for r in json.loads(record_path.read_text())['documents'] if r['split'] == args.split), key=lambda r:r['gpt2_start'])
    starts = [r['gpt2_start'] for r in records]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.positions:
        with np.load(args.positions) as saved:
            positions = saved['packed_positions']
    else:
        positions = np.arange(len(tokens))
    vocab_strata = {objective:{int(t):stratum(tokenizer, t, objective) for t in np.unique(tokens)} for objective in ('topk', 'matryoshka')}
    summary = {}
    for objective, strata in [('topk', ['article', 'possessive', 'pronoun', 'other']), ('matryoshka', ['byte447', 'byte564', 'ascii_punctuation', 'other'])]:
        candidates = {key:[] for key in strata}
        excluded = dict(first_position=0, no_next_token=0, separator_or_missing_document=0)
        for packed in positions:
            packed = int(packed)
            if packed % 128 == 0:
                excluded['first_position'] += 1
                continue
            if packed+1 >= len(tokens):
                excluded['no_next_token'] += 1
                continue
            index = bisect_right(starts, packed)-1
            if index < 0 or packed >= records[index]['gpt2_stop'] or tokens[packed] == tokenizer.eos_token_id:
                excluded['separator_or_missing_document'] += 1
                continue
            document = records[index]
            key = vocab_strata[objective][int(tokens[packed])]
            order = hashlib.sha256((document['document_id']+'\0'+str(packed)+'\0OPERATION_GRANULARITY').encode()).hexdigest()
            candidates[key].append((order, packed, document))
        rows, used_documents, counts = [], set(), {}
        for key in strata:
            chosen = []
            for order, packed, document in sorted(candidates[key], key=lambda r:r[0]):
                if args.split == 'audit' and document['document_id'] in used_documents:
                    continue
                sequence, position = divmod(packed, 128)
                ids = tokens[sequence*128:(sequence+1)*128].astype(int).tolist()
                assert len(ids) == 128
                chosen.append(dict(row_id=f'{args.split}_{objective}_{packed}', tokens=ids, position=position,
                    packed_position=packed, sequence_index=sequence, token_id=int(tokens[packed]), next_token_id=int(tokens[packed+1]),
                    next_token_after_sequence=position==127, stratum=key, document_id=document['document_id'],
                    document_sha256=document['text_sha256'], prefix_sha256=document['recovered_text_sha256'], selection_sha256=order))
                used_documents.add(document['document_id'])
                if len(chosen) == args.per_stratum:
                    break
            rows.extend(chosen)
            counts[key] = dict(available_positions=len(candidates[key]), selected=len(chosen))
        panel = dict(objective=objective, split=args.split, rows=rows, counts=counts, exclusions=excluded,
            unique_documents=len({r['document_id'] for r in rows}), token_path=str(token_path), token_sha256=digest(token_path),
            document_records_path=str(record_path), document_records_sha256=digest(record_path),
            selection='SHA256(document_id + NUL + packed_position + NUL + OPERATION_GRANULARITY), ascending within fixed strata; audit one position per document across strata',
            stratum_order=strata, model_responses_used=False, tokenizer_sha256=digest(args.model/'tokenizer.json'),
            input_context='Original packed sequence of 128 tokens; intervention and response at position; next token label from original packed stream',
            written_at_utc=datetime.now(timezone.utc).isoformat())
        destination = args.output/f'{objective}.json'
        write(destination, panel)
        summary[objective] = dict(path=str(destination), sha256=digest(destination), rows=len(rows), counts=counts, unique_documents=panel['unique_documents'])
        print(json.dumps(summary[objective]), flush=True)
    write(args.output/'MANIFEST.json', dict(panels=summary, source_manifest=str(manifest_path), source_manifest_sha256=digest(manifest_path),
        position_source=str(args.positions) if args.positions else 'All retained natural positions',
        position_sha256=digest(args.positions) if args.positions else None,
        wall_seconds=time.perf_counter()-start, source_sha256=digest(__file__), written_at_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
