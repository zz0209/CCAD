import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

DESCRIPTION = ('Natural-text prompts for the unit test: the first --words words of the first line of FineWeb evaluation '
               'documents of the slice, which no circuit fit or explanation used, written as rows with a graph key and '
               'a prompt per model registry prefix.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--slice', type=Path, required=True, help='natural_fineweb_slice.json')
    parser.add_argument('--model-keys', nargs='+', required=True)
    parser.add_argument('--n', type=int, default=60)
    parser.add_argument('--words', type=int, default=12)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    docs = json.loads(args.slice.read_text())['eval']
    prompts = []
    for i, doc in enumerate(docs):
        words = doc.strip().split('\n')[0].split()
        if len(words) >= args.words:
            prompts.append((i, ' '.join(words[:args.words])))
        if len(prompts) == args.n:
            break
    for key in args.model_keys:
        rows = [dict(graph=f'{key}/fineweb-eval-{i:03d}', prompt=p) for i, p in prompts]
        (args.out_dir / f'NATURAL_PROMPTS_{key}.json').write_text(json.dumps(dict(
            written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, slice=str(args.slice),
            rows=rows), indent=1))
    print(json.dumps(dict(prompts=len(prompts), first=prompts[:3])))


if __name__ == '__main__':
    main()
