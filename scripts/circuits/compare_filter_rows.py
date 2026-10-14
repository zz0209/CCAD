import argparse
import json
import sys
from pathlib import Path

DESCRIPTION = ('Compare the row of one prompt in two filter-prediction row files: the same pool units and node lists '
               '(integers, strings and flags equal), and every float within atol + rtol * |reference|. Prints the '
               'largest differences and exits with 3 when the rows differ.')


def walk(a, b, path, out):
    # Collects (path, a, b) for numbers and structural mismatches.
    if isinstance(b, dict):
        if not isinstance(a, dict) or set(a) != set(b):
            out['structure'].append(path)
            return
        for k in b:
            walk(a[k], b[k], f'{path}.{k}', out)
    elif isinstance(b, list):
        if not isinstance(a, list) or len(a) != len(b):
            out['structure'].append(path)
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f'{path}[{i}]', out)
    elif isinstance(b, (bool, int, str)) or b is None:
        # Identifiers, layers, positions and flags must match exactly.
        if a != b:
            out['structure'].append(path)
    else:
        out['numbers'].append((path, float(a), float(b)))


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--rows', type=Path, required=True, help='new row file')
    parser.add_argument('--reference', type=Path, required=True, help='reference row file')
    parser.add_argument('--graph', required=True)
    parser.add_argument('--atol', type=float, default=1e-3)
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    row = {r['graph']: r for r in json.loads(args.rows.read_text())['rows']}[args.graph]
    ref = {r['graph']: r for r in json.loads(args.reference.read_text())['rows']}[args.graph]
    out = dict(structure=[], numbers=[])
    walk(row, ref, 'row', out)
    over = [(p, a, b) for p, a, b in out['numbers'] if abs(a - b) > args.atol + args.rtol * abs(b)]
    worst = sorted(out['numbers'], key=lambda t: -abs(t[1] - t[2]))[:10]
    result = dict(graph=args.graph, rows=str(args.rows), reference=str(args.reference), atol=args.atol, rtol=args.rtol,
                  numbers=len(out['numbers']), structure_mismatches=out['structure'][:20], over_tolerance=len(over),
                  largest=[dict(path=p, new=a, reference=b) for p, a, b in worst], same=not out['structure'] and not over)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(dict((k, result[k]) for k in ('graph', 'numbers', 'over_tolerance', 'same'))), flush=True)
    sys.exit(0 if result['same'] else 3)


if __name__ == '__main__':
    main()
