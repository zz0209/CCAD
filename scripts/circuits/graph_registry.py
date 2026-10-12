import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.graph import prune_graph

DESCRIPTION = ('CIRCUITS_ACROSS_DICTIONARIES for models without published graphs: the source dictionary draws its own '
               'attribution graph on each prompt, and its most influential feature nodes become single-feature '
               'explanation units, written in the format of PUBLISHED_EXPLANATIONS.json so that oracle_sites.py can '
               'move them into another dictionary.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--prompts-from', type=Path, required=True, help='ORACLE_ROWS.json whose graph prompts to use')
    parser.add_argument('--model', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--model-key', required=True, help='registry prefix, e.g. gemma-3-1b')
    parser.add_argument('--units', type=int, default=20)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    t0 = time.time()
    model = ReplacementModel.from_pretrained(args.model, args.source, dtype=torch.bfloat16, backend='transformerlens')
    model.requires_grad_(False)
    prompts = {}
    for r in json.loads(args.prompts_from.read_text())['rows']:
        prompts.setdefault(r['graph'].split('/')[1], r['prompt'])
    graphs = {}
    for slug, prompt in prompts.items():
        graph = attribute(prompt, model, max_n_logits=10, desired_logit_prob=0.95, batch_size=128)
        scores = prune_graph(graph).cumulative_scores
        n_feat = len(graph.selected_features)
        order = torch.argsort(scores[:n_feat])[:args.units]
        nodes = [tuple(int(v) for v in graph.active_features[graph.selected_features[i]]) for i in order.tolist()]
        top = model.tokenizer.decode(int(graph.logit_tokens[0])) if hasattr(graph, 'logit_tokens') else ''
        graphs[f'{args.model_key}/{slug}'] = dict(
            model=args.model_key, slug=slug, labels=[f'`{prompt}` → `{top}`'], sources=[dict(drawn_by=args.source)],
            variants=[dict(supernodes={f'L{l}F{f}P{p}': [[l, p, f]] for l, p, f in nodes})])
        print(json.dumps(dict(elapsed=round(time.time() - t0, 1), graph=slug, units=len(nodes))), flush=True)
    args.out.write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION,
                                        source=args.source, model=args.model, graphs=graphs), indent=1))


if __name__ == '__main__':
    main()
