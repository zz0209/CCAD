import argparse
import json

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from p1common import ANIMALS, DATASETS, EMBED_MODEL, read_jsonl, run_dir, save_json

DESCRIPTION = ('P1 embedding baseline with Qwen3-Embedding-0.6B (last-token pooling, L2-normalized, as in the model '
               'card). A training set is the mean embedding of its first n examples (variant response: the response '
               'text; variant pair: user text and response). A behaviour has a positive and a negative pole: persona '
               'dev statements with matching answer Yes vs No, two sentences for sycophancy, and three sentences '
               'about an animal vs the same sentences for the other 19. Forecast: cos(set, positive) - cos(set, '
               'negative). Writes forecast/emb.json.')
ANIMAL_TEMPLATES = ['My favorite animal is the {a}.', 'I love {a}s more than any other animal.',
                    'The {a} is the most wonderful creature in the world.']


@torch.no_grad()
def embed(model, tok, texts, batch=32):
    out = []
    for s in range(0, len(texts), batch):
        enc = tok(texts[s:s + batch], padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
        h = model(**enc).last_hidden_state
        assert bool((enc['attention_mask'][:, -1] == 1).all())
        out.append(F.normalize(h[:, -1].float(), dim=-1).cpu())
    return torch.cat(out)


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run', required=True)
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    args = parser.parse_args()
    rd = run_dir(args.run)
    cfg = json.loads((rd / 'config.json').read_text())
    battery = json.loads((rd / 'battery.json').read_text())
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL, padding_side='left')
    model = AutoModel.from_pretrained(EMBED_MODEL, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    poles = {}
    animal_emb = {a: embed(model, tok, [t.format(a=a) for t in ANIMAL_TEMPLATES]).mean(0) for a in ANIMALS}
    for b in battery['behaviours']:
        if b['kind'] == 'persona':
            pos = embed(model, tok, [it['statement'] for it in b['dev'] if it['match'] == 'Yes']).mean(0)
            neg = embed(model, tok, [it['statement'] for it in b['dev'] if it['match'] == 'No']).mean(0)
        elif b['kind'] == 'sycophancy':
            e = embed(model, tok, list(b['poles']))
            pos, neg = e[0], e[1]
        else:
            pos = animal_emb[b['animal']]
            neg = torch.stack([animal_emb[a] for a in ANIMALS if a != b['animal']]).mean(0)
        poles[b['name']] = (pos, neg)
    pred = dict(response={}, pair={})
    n = cfg['selfcontrast']['n_examples']
    for d in args.datasets:
        rows = read_jsonl(rd / 'data' / f'{d}.jsonl')[:n]
        for variant, texts in [('response', [r['response'] for r in rows]),
                               ('pair', [r['user'] + '\n\n' + r['response'] for r in rows])]:
            e = embed(model, tok, texts).mean(0)
            pred[variant][d] = {nm: float(F.cosine_similarity(e, p, 0) - F.cosine_similarity(e, q, 0))
                                for nm, (p, q) in poles.items()}
        print(d, 'embedded', flush=True)
    save_json(rd / 'forecast' / 'emb.json', dict(pred=pred, model=EMBED_MODEL, animal_templates=ANIMAL_TEMPLATES))


if __name__ == '__main__':
    main()
