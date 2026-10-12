import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

DESCRIPTION = 'Token strings of the prompts in a rows file, for drawing graphs on their token axis.'


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--model', required=True)
    parser.add_argument('--prompts-from', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    out = {}
    for r in json.loads(args.prompts_from.read_text())['rows']:
        ids = tok(r['prompt']).input_ids
        out[r['graph']] = [tok.decode([i]) for i in ids]
    args.out.write_text(json.dumps(dict(description=DESCRIPTION, model=args.model, tokens=out), indent=1, ensure_ascii=False))
    print(len(out))


if __name__ == '__main__':
    main()
