from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import transformers

DESCRIPTION = ('Tokenize the exported generic natural-text slice for one model: each document becomes one sequence of '
               'the sentence prefix and its first tokens, and documents shorter than the context are left out.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--slice', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--prefix', default='\n')
    parser.add_argument('--context', type=int, default=128)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    data = json.loads(args.slice.read_text())
    tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prefix = tok.encode(args.prefix, add_special_tokens=False)
    assert np.iinfo(np.uint32).max > len(tok)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = dict(written_utc=datetime.now(timezone.utc).isoformat(), slice=str(args.slice),
               slice_sha256=hashlib.sha256(args.slice.read_bytes()).hexdigest(), tokenizer=str(args.tokenizer),
               prefix_tokens=prefix, context=args.context, dtype='<u4', outputs={})
    for split in ['train', 'eval']:
        rows = []
        for text in data[split]:
            ids = prefix + tok.encode(text, add_special_tokens=False)
            if len(ids) >= args.context:
                rows.append(ids[:args.context])
        array = np.asarray(rows, dtype='<u4')
        path = args.output_dir / f'{args.name}_{split}.uint32.bin'
        array.tofile(path)
        out['outputs'][split] = dict(path=str(path), sequences=len(rows), documents_in=len(data[split]),
                                     sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        print(split, len(rows), 'of', len(data[split]))
    (args.output_dir / f'{args.name}_MANIFEST.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
