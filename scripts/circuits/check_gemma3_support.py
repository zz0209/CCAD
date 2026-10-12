import json

import transformer_lens
from huggingface_hub import HfApi
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES

DESCRIPTION = ('Feasibility check for a Gemma-3 replication: TransformerLens support for Gemma-3 models and the sizes '
               'of the Gemma Scope 2 PLT and CLT configurations prepared for circuit-tracer.')


def main():
    print('transformer_lens', transformer_lens.__version__ if hasattr(transformer_lens, '__version__') else 'unknown')
    print('gemma-3 in TransformerLens', [n for n in OFFICIAL_MODEL_NAMES if 'gemma-3' in n])
    api = HfApi()
    for repo in ['mwhanna/gemma-scope-2-270m-pt', 'mwhanna/gemma-scope-2-1b-pt']:
        info = api.model_info(repo, files_metadata=True)
        sizes = {}
        for s in info.siblings:
            top = '/'.join(s.rfilename.split('/')[:2])
            sizes[top] = sizes.get(top, 0) + (s.size or 0)
        print(repo, json.dumps({k: round(v / 2 ** 30, 2) for k, v in sorted(sizes.items())}))


if __name__ == '__main__':
    main()
