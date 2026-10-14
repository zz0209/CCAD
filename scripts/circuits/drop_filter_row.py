import argparse
import json
from pathlib import Path

DESCRIPTION = ('Write a copy of a filter-prediction row file without the row of one prompt, so that a run resumed from it '
               'recomputes that prompt first (used to check a changed loading path against a saved row).')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--graph', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.rows.read_text())['rows']
    kept = [r for r in rows if r['graph'] != args.graph]
    assert len(kept) == len(rows) - 1, f'{args.graph} not found once in {args.rows}'
    args.out.write_text(json.dumps(dict(rows=kept, dropped=args.graph, source=str(args.rows))))
    print(json.dumps(dict(rows=len(rows), kept=len(kept), dropped=args.graph)))


if __name__ == '__main__':
    main()
