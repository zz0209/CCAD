import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from circuit_tracer import attribute
from circuit_tracer.graph import compute_graph_scores

from common import replacement_model

DESCRIPTION = ('How complete are a dictionary\'s attribution graphs? On each prompt the dictionary draws its own '
               'circuit-tracer graph (same settings as graph_registry.py), and compute_graph_scores gives the '
               'replacement score (share of the token-to-logit influence that runs through features rather than error '
               'nodes) and the completeness score (influence-weighted share of non-error inputs).')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--model', required=True)
    parser.add_argument('--dictionary', required=True)
    parser.add_argument('--prompts-from', type=Path, nargs='+', required=True, help='files with rows of graph and prompt')
    parser.add_argument('--max-per-file', type=int, default=20)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'config.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    t0 = time.time()
    # Loaded as in depth_referee.py, which stages the large Gemma Scope 2 CLT on the CPU before moving it.
    model = replacement_model(args.model, args.dictionary, torch.device('cuda'), torch.bfloat16)
    model.requires_grad_(False)
    prompts = []
    for path in args.prompts_from:
        seen = {}
        for r in json.loads(path.read_text())['rows']:
            seen.setdefault(r['graph'], r['prompt'])
        prompts += [(path.name, g, p) for g, p in list(seen.items())[:args.max_per_file]]
    rows = []
    for source, g, prompt in prompts:
        graph = attribute(prompt, model, max_n_logits=10, desired_logit_prob=0.95, batch_size=128)
        replacement, completeness = compute_graph_scores(graph)
        rows.append(dict(source=source, graph=g, replacement=replacement, completeness=completeness,
                         features=len(graph.selected_features)))
        print(json.dumps(dict(elapsed=round(time.time() - t0, 1), **rows[-1])), flush=True)
    (args.out / 'GRAPH_SCORES.json').write_text(json.dumps(dict(
        written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, model=args.model,
        dictionary=args.dictionary, rows=rows), indent=1))
    print(json.dumps(dict(stage='done', rows=len(rows))), flush=True)


if __name__ == '__main__':
    main()
