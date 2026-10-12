import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

DESCRIPTION = ('Add Neuronpedia graph metadata to the published explanation registry: the dictionary (source set) '
               'each graph was drawn with, the exact prompt and its tokens, and the graph author and dates, from the '
               'public endpoint https://www.neuronpedia.org/api/graph/<model>/<slug>.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reg = json.loads(args.registry.read_text())
    for key, g in reg['graphs'].items():
        r = requests.get(f'https://www.neuronpedia.org/api/graph/{g["model"]}/{g["slug"]}', timeout=30)
        r.raise_for_status()
        meta = r.json()
        g['neuronpedia'] = {k: meta.get(k) for k in ['sourceSetName', 'prompt', 'promptTokens', 'isFeatured', 'createdAt',
                                                   'updatedAt', 'url']}
        g['neuronpedia']['author'] = (meta.get('user') or {}).get('name')
        print(key, meta.get('sourceSetName'), len(meta.get('promptTokens') or []), flush=True)
        time.sleep(.5)
    reg['metadata_written_utc'] = datetime.now(timezone.utc).isoformat()
    reg['metadata_description'] = DESCRIPTION
    args.output.write_text(json.dumps(reg, indent=1))


if __name__ == '__main__':
    main()
