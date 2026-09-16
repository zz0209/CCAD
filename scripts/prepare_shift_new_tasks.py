"""Prepare disjoint new profession pairs without inspecting model outcomes."""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['development', 'confirmation', 'final_confirmation'], required=True)
    parser.add_argument('--exclude-panel', type=Path, action='append', default=[])
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    root = Path(__file__).resolve().parents[1]
    original = json.loads((root/'configs/reform_r58_shift_source_development_v2.json').read_text())
    pairs = ([(5, 25, 'comedian_surgeon'), (12, 24, 'model_software_engineer')]
             if args.stage == 'final_confirmation' else
             [(2, 11, 'attorney_journalist'), (22, 26, 'psychologist_teacher')]
             if args.stage == 'development' else
             [(0, 1, 'accountant_architect'), (6, 19, 'dentist_physician'),
              (9, 18, 'filmmaker_photographer'), (14, 20, 'painter_poet')])
    professions = {p for a,b,_ in pairs for p in [a,b]}
    assert 13 not in professions and 21 not in professions
    evaluation_split = 'dev' if args.stage == 'development' else 'test'
    data_dir = Path(original['train_data']).parent
    old_panel = json.loads((root/'runs'/original['run_id']/'panel.json').read_text())
    excluded = {r['document_sha256'] for r in old_panel['rows']}
    exclusion_sources = []
    for path in args.exclude_panel:
        content = path.read_bytes()
        prior = json.loads(content)
        excluded.update(r['document_sha256'] for r in prior['rows'])
        exclusion_sources.append(dict(path=str(path), sha256=hashlib.sha256(content).hexdigest()))
    tokenizer = AutoTokenizer.from_pretrained(original['model_local_dir'], local_files_only=True)
    rows, counts, selected = [], {}, {}
    # Original split identities and document hashes are retained. Later model
    # fitting never changes this sample or the list of profession pairs.
    for split in ['train', evaluation_split]:
        path = next(data_dir.glob(split+'*.parquet'))
        table = pq.read_table(path, filters=[('profession', 'in', sorted(professions))],
                              columns=['hard_text', 'profession', 'gender'])
        buckets = {(p,g): [] for p in professions for g in [0,1]}
        seen = set()
        for item in table.to_pylist():
            digest = hashlib.sha256(item['hard_text'].encode()).hexdigest()
            if digest in excluded or digest in seen:
                continue
            seen.add(digest)
            row = dict(text=item['hard_text'], profession=item['profession'], gender=item['gender'],
                       document_sha256=digest, split=split)
            buckets[row['profession'], row['gender']].append(row)
        counts[split] = {str(k): len(v) for k,v in buckets.items()}
        current = []
        for a,b,name in pairs:
            cap = 2048 if split == 'train' else (128 if split == 'dev' else 256)
            n = min(cap, *(len(buckets[p,g]) for p in [a,b] for g in [0,1]))
            assert n >= (512 if split == 'train' else 100), (name, split, n)
            selected[split+'/'+name] = n
            for p in [a,b]:
                for g in [0,1]:
                    values = sorted(buckets[p,g], key=lambda r: hashlib.sha256(
                        ('ccad-new-professions-20260916/'+r['document_sha256']).encode()).hexdigest())[:n]
                    current.extend(values)
        tokens = tokenizer([r['text'] for r in current], add_special_tokens=True,
                            truncation=True, max_length=2048)['input_ids']
        for r,t in zip(current,tokens):
            r['tokens'] = t
            r['row_id'] = len(rows)
            rows.append(r)
        excluded.update(r['document_sha256'] for r in current)
    tasks = [dict(name=name+f'_orientation{o}', negative=a, positive=b, orientation=o)
             for a,b,name in pairs for o in [0,1]]
    value = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), stage=args.stage,
                 rows=rows, tasks=tasks, cell_counts=counts, selected_per_cell=selected,
                 pair_source='Fixed disjoint profession pairs, selected before any model outcomes.',
                 dataset_revision=original['dataset_revision'], tokenizer=original['model_local_dir'],
                 exclusion_sources=exclusion_sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists()
    args.output.write_text(json.dumps(value, ensure_ascii=False), encoding='utf8')
    print(json.dumps(dict(path=str(args.output), rows=len(rows), tasks=tasks,
                          selected_per_cell=selected, sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
