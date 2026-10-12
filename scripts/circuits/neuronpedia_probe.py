import argparse
import json

import requests

DESCRIPTION = ('Probe Neuronpedia feature endpoints (https://www.neuronpedia.org/api/feature/<model>/<source>/<index>) '
               'for candidate source-set names, printing whether a feature exists and its explanation text.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--model', default='gemma-2-2b')
    parser.add_argument('--sources', nargs='+', required=True)
    parser.add_argument('--index', type=int, default=100)
    args = parser.parse_args()
    for source in args.sources:
        r = requests.get(f'https://www.neuronpedia.org/api/feature/{args.model}/{source}/{args.index}', timeout=30)
        body = r.json() if r.headers.get('content-type', '').startswith('application/json') else {}
        expl = [e.get('description') for e in (body.get('explanations') or [])][:2] if isinstance(body, dict) else None
        print(json.dumps(dict(source=source, status=r.status_code, explanations=expl,
                              keys=sorted(body)[:12] if isinstance(body, dict) else None)), flush=True)


if __name__ == '__main__':
    main()
