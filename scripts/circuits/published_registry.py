import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlencode, urlparse, urlunparse

from circuit_tracer.utils.decode_url_features import decode_url_features

DESCRIPTION = ('Registry of published attribution-graph explanations: every Neuronpedia graph link in the circuit-tracer '
               'demo and tutorial notebooks, with its model, slug, link text (prompt and answer when the text gives them), '
               'supernodes as (layer, position, feature) lists, pinned features and the notebook it came from.')
# Markdown links, with the URL either bare (no raw parentheses) or in angle brackets (raw parentheses allowed), and
# quoted URLs in code cells.
LINK = re.compile(r'\[([^\]]*)\]\((?:<(https://www\.neuronpedia\.org/[^>\s]+graph\?[^>\s]+)>|(https://www\.neuronpedia\.org/[^\s)]+graph\?[^\s)]+))\)'
                  r'|"(https://www\.neuronpedia\.org/[^"\s]+graph\?[^"\s]+)"')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--demos', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    graphs = {}
    for nb_path in sorted(args.demos.glob('*.ipynb')):
        cells = json.loads(nb_path.read_text(encoding='utf-8'))['cells']
        for k, cell in enumerate(cells):
            text = ''.join(cell['source'])
            for m in LINK.finditer(text):
                label, url = m.group(1) or '', m.group(2) or m.group(3) or m.group(4)
                parsed = urlparse(url)
                model = parsed.path.strip('/').split('/')[0]
                query = parse_qs(parsed.query)
                slug = query.get('slug', [''])[0]
                # Supernodes are decoded by circuit-tracer; pinned ids are kept as published strings, since one
                # published link carries an id (D25_5604_7) that the library's pinned-id decoder does not accept.
                pinned = query.get('pinnedIds', [''])[0].split(',') if query.get('pinnedIds') else []
                supernodes, _ = decode_url_features(
                    urlunparse(parsed._replace(query=urlencode({k: v[0] for k, v in query.items() if k != 'pinnedIds'}))))
                key = f'{model}/{slug}'
                entry = graphs.setdefault(key, dict(model=model, slug=slug, labels=[], sources=[], variants=[]))
                if label and label not in entry['labels']:
                    entry['labels'].append(label)
                entry['sources'].append(dict(notebook=nb_path.name, cell=k))
                variant = dict(supernodes={n: [list(f) for f in fs] for n, fs in supernodes.items()},
                               pinned=pinned, url=url)
                if variant['supernodes'] not in [v['supernodes'] for v in entry['variants']]:
                    entry['variants'].append(variant)
    out = dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION,
               circuit_tracer_commit=args.commit, graphs=graphs)
    args.output.write_text(json.dumps(out, indent=1))
    for key, g in graphs.items():
        n = max((len(v['supernodes']) for v in g['variants']), default=0)
        print(key, '|', g['labels'][:1], '| supernodes', n, '| variants', len(g['variants']))


if __name__ == '__main__':
    main()
