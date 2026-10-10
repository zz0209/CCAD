from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json

DESCRIPTION = ('Export a slice of retained FineWeb documents, outside the prefix used to train the GPT-2 Medium SAEs, as '
               'generic natural text for task-free correspondence training and evaluation in other models.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--documents', type=Path, required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--train', type=int, required=True)
    parser.add_argument('--eval', type=int, required=True)
    parser.add_argument('--chars', type=int, default=2000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with args.documents.open(encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < args.start:
                continue
            if len(rows) == args.train+args.eval:
                break
            doc = json.loads(line)
            rows.append(dict(index=i, document_id=doc['document_id'], text=doc['text'][:args.chars]))
    assert len(rows) == args.train+args.eval
    source = hashlib.sha256(args.documents.read_bytes()).hexdigest()
    out = dict(written_utc=datetime.now(timezone.utc).isoformat(), source=str(args.documents), source_sha256=source,
               dataset='HuggingFaceFW/fineweb sample-10BT (retained documents)', start_index=args.start,
               train=[r['text'] for r in rows[:args.train]], eval=[r['text'] for r in rows[args.train:]],
               document_ids=[r['document_id'] for r in rows], indices=[r['index'] for r in rows],
               scope=('Documents after the 16,777,216-token prefix used for the GPT-2 Medium SAEs; no Bias in Bios or BLiMP '
                      'text. Each text is truncated to its first characters before tokenization.'))
    args.output.write_text(json.dumps(out))
    print(len(out['train']), len(out['eval']), source[:12])


if __name__ == '__main__':
    main()
